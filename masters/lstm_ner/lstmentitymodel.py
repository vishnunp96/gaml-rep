import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertConfig, BertForMaskedLM

from annotations.datasets import AnnotationDataset

from annotations.models.base import BaseANNEntityModule
from masters.berts.checkpointing import load_checkpoint

# Adapting annotations.models.lstm_entity_model to use BERT embeddings

class TokenisedAnnotations(AnnotationDataset):
    def __init__(self, tokenizer, tokenizer_max_length, ann_list,
                 bert_model,
                 label_encoder,
                 sentence_split_params,
                 cuda=False,
                 outside_class="outside",
                 begin_class="begin",
                 inside_class="inside"):
        super(TokenisedAnnotations, self).__init__(ann_list, cuda)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.class_template = {"O": outside_class, "B": begin_class,
                               "I": inside_class}
        self.cuda = cuda
        self.label_encoder = label_encoder
        self.bert_model = bert_model

        self.sentence_split_length, self.sentence_overlap = sentence_split_params

        self.words, self.annotations, self.classes = self.load_annotations(
            ann_list)
        print(f"Finished reading annotations. Classes: {self.classes}")
        # sentence_tensor = num_sentences * (['input_ids', 'token_type_ids', 'attention_mask'] X 1 * tokenizer_max_length)
        self.sentence_tensors, self.labels = self.tokenize_annotations(
            self.words,
            self.annotations)
        if self.cuda:
            print("Moving vars to GPU", flush=True)
            self.sentence_tensors = [t.to('cuda') for t in self.sentence_tensors]
        print(f"Now to get embeddings.. Found sentences: {len(self.sentence_tensors)}", flush=True)
        # num_sentences * num_tokens * 768
        self.embeddings = self.get_embeddings()
        print(f"Finished making tokens and labels. Length: {self.__len__()}", flush=True)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sentence_embeds, labels = self.embeddings[idx], self.labels[idx]
        return sentence_embeds.float(), torch.from_numpy(labels).long()

    def collate_fn(self, sequences):
        embeds, labels = zip(*sequences)
        embeds, labels = (rnn.pack_sequence(embeds,
                                            enforce_sorted=False),
                          rnn.pack_sequence(
                              labels, enforce_sorted=False))
        if self.cuda:
            return embeds.cuda(), labels.cuda()
        else:
            return embeds, labels


    def load_annotations(self, standoff_list):
        # need to use standoffs instead of directory.
        # need to split appropriately as well.
        words = []
        annotations = []
        classes = set()

        for standoff in standoff_list:
            for i in range(0, len(standoff.tokens),
                           self.sentence_split_length - self.sentence_overlap):
                words.append(standoff.tokens[i:i + self.sentence_split_length])
                annotations.append(
                    standoff.labels[i:i + self.sentence_split_length])
                classes.update(annotations[-1])

        return words, annotations, classes

    def tokenize_annotations(self, words, annotations):
        token_tensor = []
        label_list = []
        for word_line, ann_line in zip(words, annotations):
            # get tokens
            sentence = " ".join(word_line)
            tokenized_tensor = self.tokenizer(sentence, return_tensors='pt',
                                              max_length=self.tokenizer_max_length,
                                              padding='max_length',
                                              truncation=True)
            labels = ([self.class_template["O"]] *
                      torch.sum(tokenized_tensor['attention_mask'],
                                dim=1).item())

            label_index = 0
            for word, ann in zip(word_line, ann_line):
                tokens = self.tokenizer.tokenize(word)
                for i, token in enumerate(tokens):
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if label_index >= len(labels):
                        print("Warning: Labels are not fitting when mapping to tokenized versions of sentences.")
                        break
                    while tokenized_tensor['input_ids'][0][
                        label_index] != token_id:
                        label_index += 1
                    if i == 0:
                        labels[label_index] = ann
                    else:
                        labels[label_index] = ann.replace(
                            self.class_template["B"], self.class_template["I"])
                    # labels[label_index] = ann.replace("begin", "inside")
                    label_index += 1
            token_tensor.append(tokenized_tensor)
            label_list.append(self.label_encoder.transform(labels))


        return token_tensor, label_list

    def get_embeddings(self):
        embeddings = []
        for sentence_tensor in self.sentence_tensors:
            with torch.no_grad():
                output = self.bert_model(sentence_tensor['input_ids'],
                                         sentence_tensor['attention_mask'])
            # final hidden state = 1 * tokenizer_max_length * 768
            # labels = flat( num_sentences * tokenizer_max_length )
            length = torch.sum(sentence_tensor['attention_mask'], dim=1).item()
            embeddings.append(output.hidden_states[-1][0][:length].float())
        return embeddings


# Define model
class LSTMEntityBertEmbedModel(BaseANNEntityModule):
    def __init__(self, hidden_size, num_layers, output_labels, bert_name,
                 embed_model,
                 tokenizer_max_length=512):
        super(LSTMEntityBertEmbedModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bert_name = bert_name
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        self.tokenizer_max_length = tokenizer_max_length
        # sentence splitting of standoffs based on tokenizer_max_length/4
        self.sentence_split_params = (tokenizer_max_length // 4, 16)

        self.embed_model = embed_model

        # labels <-> ids
        self.labels = LabelEncoder().fit(output_labels)
        # num labels
        self.output_num = len(self.labels.classes_)

        # Embedding - object containing embeddings - num rows
        # LSTM goes from embedding of a token to output
        # our embedding matrix = flat( num_sentences * tokenizer_max_length  ) * 768
        # this dimension is usually 768 for BERT
        self.embedding_dim = self.embed_model.config.hidden_size
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size,
                            self.num_layers, bidirectional=True, dropout=0.2)

        # for linear transformation of output - hidden_size*2 - to output_num
        self.output_dense = nn.Linear(self.hidden_size * 2, self.output_num)

    def make_dataset(self, ann_list):
        return TokenisedAnnotations(self.tokenizer, self.tokenizer_max_length,
                                    ann_list,
                                    self.embed_model,
                                    self.labels,
                                    self.sentence_split_params,
                                    cuda=next(self.output_dense.parameters()).is_cuda
                                    )

    def forward(self, sentence_batch):
        """ Accepts PackedSequence, representing batch of token index sequences. """
        # PACKED => means zeros are truncated
        # UNPACKED => means zeros are added to make it square

        # LSTM LAYERS
        # pack it back into PackedSequence
        # perform LSTM operation - bidirectional, so output is 2*hidden_size
        sentence_batch, _ = self.lstm(
            sentence_batch)  # (batch_size, sequence_length, hidden_size*2)
        # unpack the sequence
        sentence_batch, lengths = rnn.pad_packed_sequence(sentence_batch,
                                                          padding_value=0,
                                                          batch_first=True)
        # RELU activation for classification
        relu_activated = F.relu(sentence_batch)
        # linear transformation to output space
        relu_transformed = self.output_dense(
            relu_activated)  # (batch_size, sequence_length, output_num)
        # pack it back into PackedSequence
        relu_packed = rnn.pack_padded_sequence(relu_transformed, lengths,
                                               batch_first=True,
                                               enforce_sorted=False)
        return relu_packed

    def compute_class_weight(self, class_weight, dataset):
        return compute_class_weight(class_weight,
        classes=self.labels.classes_,
        y=[tag for a in dataset.anns for tag in
        a.labels])








if __name__ == '__main__':

    from utilities import StopWatch
    stopwatch = StopWatch(memory=True)

    from utilities.torchutils import save_figs # predict_from_dataloader

    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction,DirectoryAction,ListAction
    from utilities.mlutils import split_data
    import os

    from annotations.annmlutils import open_anns

    import sklearn.metrics

    from annotations.models.training import perform_training,evaluate_entities

    def parse_tuple(s):
        return tuple(int(i) for i in s.split('-'))

    parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
    parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
    parser.add_argument('bertmodel',action=FileAction, mustexist=True,help='Bert fine-tuned model.')
    parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
    parser.add_argument('-w','--class-weight',action='store_const',const='balanced',help='Flag to indicate that the loss function should be class-balanced for training.')
    parser.add_argument('--train-fractions',action='store_true',help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
    parser.add_argument('--eval',action='store_true',help='Flag to indicate that the model in modeldir should be loaded and evaluated, rather than a new model be trained.')
    parser.add_argument('--hidden',type=int,default=32,help='Number of neurons in hidden layers.')
    parser.add_argument('--layers',type=int,default=2,help='Number of layers of LSTM cells.')
    parser.add_argument('--seed', type=int, default=42, help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.8,0.1,0.1), help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
    parser.add_argument('--types',action=ListAction, help='Annotation types to consider.')
    args = parser.parse_args()

    print("Starting..", flush=True)

    torch.manual_seed(args.seed)
    modelname = os.path.basename(args.modeldir)

    # Read in data
    if not args.types:
        args.types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Definition']
    anns = open_anns(args.ann,types=args.types,use_labelled=True)
    ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)
    output_labels = list(set(l for a in anns for t,l in a))

    stopwatch.tick('Opened all files',report=True)

    # Open up BERT stuff
    bert_model_name = 'bert-base-cased'
    config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
    bert_model = BertForMaskedLM.from_pretrained(bert_model_name, config=config)
    _, _, _, loaded_model_base = load_checkpoint(args.bertmodel, bert_model)

    stopwatch.tick('Loaded the bert model', report=True)
    print("Loaded the bert model", flush=True)

    # Make test/train split
    # Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics

    # Model parameters

    # Training parameters
    batch_size = 64
    epochs = 150
    patience = 25
    min_delta = 0.001

    # Generating functions
    def make_model():
        return LSTMEntityBertEmbedModel(args.hidden, args.layers, output_labels, bert_model_name,
                                        bert_model).float().cuda()
    def make_opt(model):
        # learning rate is defined here.
        adam_lr = 0.001
        return optim.Adam(model.parameters(), lr=adam_lr)
    def make_loss_func(model, train_dataset):
        ignore_index = 999
        class_weights = model.compute_class_weight(args.class_weight, train_dataset)
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=torch.tensor(class_weights).float().cuda() if args.class_weight else None)
        return model.make_loss_func(criterion,ignore_index=ignore_index)
    def make_metrics(model):
        average = 'macro'
        f1_score = model.make_metric(lambda o,t: sklearn.metrics.f1_score(t,o,average=average,zero_division=0))
        precision_score = model.make_metric(lambda o,t: sklearn.metrics.precision_score(t,o,average=average,zero_division=0))
        recall_score = model.make_metric(lambda o,t: sklearn.metrics.recall_score(t,o,average=average,zero_division=0))
        return {'f1': f1_score, 'precision': precision_score, 'recall': recall_score}
    model, _, history = perform_training(
            make_model, (ann_train,ann_dev,ann_test), args.modeldir,
            make_metrics, make_opt, make_loss_func,
            batch_size=batch_size, epochs=epochs,
            patience=patience, min_delta=min_delta,
            modelname= modelname,
            train_fractions=args.train_fractions,
            stopwatch=stopwatch)

    save_figs(history, modelname, args.modeldir)

    ### TEST MODEL
    class_metrics,overall_metrics = evaluate_entities(model, ann_test, batch_size=batch_size)
    class_metrics.to_csv(os.path.join(args.modeldir,'class_metrics.csv'))
    overall_metrics.to_csv(os.path.join(args.modeldir,'overall_metrics.csv'))

    stopwatch.tick('Finished evaluation',report=True)

    stopwatch.report()



