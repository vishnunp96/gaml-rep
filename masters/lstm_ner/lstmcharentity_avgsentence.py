import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertConfig, BertForMaskedLM

from annotations.datasets import EntityIndexesCharsDataset, AnnotationDataset

from annotations.models.base import BaseANNEntityModule
from masters.berts.checkpointing import load_checkpoint

'''
- Using BERT embeddings for training an LSTM char entity model.
- Each words character embedding added to all tokens from that word. 
- Attempting whether BERT embeddings averaged across sentence splitting is actually worth it.
    - Worse than LSTM character entity with sentence splitting.
'''

class AvgdSentenceEmbedCharDataset(AnnotationDataset):
    def __init__(self, tokenizer, tokenizer_max_length,
                 char_embedding_dim,
                 char_indexes,
                 ann_list,
                 bert_model,
                 label_encoder,
                 sentence_split_params,
                 cuda=False,
                 outside_class="outside",
                 begin_class="begin",
                 inside_class="inside"):
        super(AvgdSentenceEmbedCharDataset, self).__init__(ann_list, cuda)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.class_template = {"O": outside_class, "B": begin_class,
                               "I": inside_class}
        self.cuda = cuda
        self.label_encoder = label_encoder
        self.char_indexes = char_indexes
        self.bert_model = bert_model

        self.beg_token_char = numpy.array([self.char_indexes.get(c, self.char_indexes[None])
                                      for c in self.tokenizer.cls_token])
        self.end_token_char = numpy.array([self.char_indexes.get(c, self.char_indexes[None])
                                      for c in self.tokenizer.sep_token])
        self.sentence_split_length, self.sentence_overlap = sentence_split_params

        (self.abstracts,
         self.abstract_anns,
         self.abstract_char_indexes,
         self.classes) = self.load_annotations(
            ann_list)
        # abstracts = num_abstract * (num_tokens * 768)
        # abstract_anns = num_abstract * (num_tokens) [0-NUM_CLASSES]
        # abstract_char_indexes = num_abstract * (num_tokens * num_chars)


        print(f"Finished reading annotations. Classes: {self.classes}")

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        abstract_embeds, char_indices, labels = self.abstracts[idx], \
            self.abstract_char_indexes[idx], self.abstract_anns[idx]
        # return ((sentence_embeds.float(), char_indices.long()), torch.from_numpy(labels).long())
        return ((abstract_embeds.float(), rnn.pack_sequence([torch.from_numpy(i).long() for i in char_indices],False)), torch.from_numpy(labels).long())


    def collate_fn(self, sequences):
        xs, labels = zip(*sequences)
        abstract_embeds, char_indexes = tuple(zip(*xs))
        if self.cuda:
            return (rnn.pack_sequence(abstract_embeds, False).cuda(),
                    [i.cuda() for i in char_indexes]), rnn.pack_sequence(labels, False).cuda()
        else:
            return (
            rnn.pack_sequence(abstract_embeds, False), [i for i in char_indexes]), rnn.pack_sequence(labels,
                                                                              False)


    def load_annotations(self, standoff_list):
        # need to use standoffs instead of directory.
        # need to split appropriately as well.
        abstracts = []
        abstract_char_indexes = []
        abstract_anns = []
        classes = set()

        for standoff in standoff_list:
            standoff_embed = torch.empty(0, self.bert_model.config.hidden_size)
            if self.cuda:
                standoff_embed = standoff_embed.to('cuda')
            standoff_anns = []
            standoff_char_indexes = []
            prev_embed_overlap = None
            for i in range(0, len(standoff.tokens),
                           self.sentence_split_length - self.sentence_overlap):
                word_array = standoff.tokens[i:i + self.sentence_split_length]
                # abstracts[-1].append(cur_sentence)
                label_array = standoff.labels[i:i + self.sentence_split_length]
                # annotations[-1].append(cur_labels)
                classes.update(label_array)
                sentence_tensor, token_aligned_labels, char_embeds, overlap_index = self.tokenize_sentence_and_labels(
                    word_array, label_array)
                if self.cuda:
                    sentence_tensor = sentence_tensor.to('cuda')
                sentence_embed = self.get_sentence_embedding(sentence_tensor)
                if prev_embed_overlap is not None:
                    # need to average embeddings
                    overlap_length = len(prev_embed_overlap)
                    overlap_embed = (sentence_embed[:overlap_length] + prev_embed_overlap) / 2
                    sentence_embed = torch.cat((overlap_embed, sentence_embed[overlap_length:]))
                    # trimming existing standoff to replace with averaged
                    # trimming labels too, no need to average though.
                    standoff_embed = standoff_embed[:-overlap_length]
                    standoff_anns = standoff_anns[:-overlap_length]
                    standoff_char_indexes = standoff_char_indexes[:-overlap_length]
                standoff_embed = torch.cat((standoff_embed, sentence_embed), dim=0)
                standoff_anns.extend(token_aligned_labels)
                standoff_char_indexes.extend(char_embeds)
                # (n-16)*768 + ~128*768
                prev_embed_overlap = sentence_embed[overlap_index:]
            abstracts.append(standoff_embed)
            abstract_anns.append(np.array(standoff_anns))
            abstract_char_indexes.append(standoff_char_indexes)

        return abstracts, abstract_anns, abstract_char_indexes, classes

    def tokenize_sentence_and_labels(self, word_array, ann_array):
        tokenized = self.tokenizer(' '.join(word_array), return_tensors='pt',
                                   max_length=self.tokenizer_max_length,
                                   padding='max_length',
                                   truncation=True)
        token_labels = ([self.class_template["O"]] *
                  torch.sum(tokenized['attention_mask'],
                            dim=1).item())

        label_index = 0
        overlap_index = 0
        sent_char_embeds = [self.end_token_char] * len(token_labels)
        sent_char_embeds[0] = self.beg_token_char
        for word_index, (word, ann) in enumerate(zip(word_array, ann_array)):
            if word_index == self.sentence_split_length - self.sentence_overlap:
                overlap_index = label_index
            word_char_index = numpy.array(
                [self.char_indexes.get(c, self.char_indexes[None]) for c in
                 word])
            tokens = self.tokenizer.tokenize(word)
            for i, token in enumerate(tokens):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if label_index >= len(token_labels):
                    print(
                        "Warning: Labels are not fitting when mapping to tokenized versions of sentences.")
                    break
                while tokenized['input_ids'][0][label_index] != token_id:
                    label_index += 1
                if i == 0:
                    token_labels[label_index] = ann
                else:
                    token_labels[label_index] = ann.replace(
                        self.class_template["B"], self.class_template["I"])
                sent_char_embeds[label_index] = word_char_index
                label_index += 1
        # label_list.append(self.label_encoder.transform(labels))
        transformed_labels = self.label_encoder.transform(token_labels)

        return tokenized, transformed_labels, sent_char_embeds, overlap_index

    def get_sentence_embedding(self, sentence_tensor):
        with torch.no_grad():
            output = self.bert_model(sentence_tensor['input_ids'],
                                     sentence_tensor['attention_mask'])
        # final hidden state = 1 * tokenizer_max_length * 768
        # labels = flat( num_sentences * tokenizer_max_length )
        length = torch.sum(sentence_tensor['attention_mask'], dim=1).item()
        return output.hidden_states[-1][0][:length].float()

class CharacterEncoder(nn.Module):
    def __init__(self, embedding_dim, bidirectional=False):
        super(CharacterEncoder, self).__init__()

        # added hash as tokenized works tend to include hash
        self.char_indexes = {c: (i + 1) for i, c in enumerate(
            "$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~\'\"#")}
        self.char_indexes[None] = 0  # Unknown character (fallback)
        ## Replacements for some common non-ascii Unicode characters
        self.char_indexes['\u2018'] = self.char_indexes[
            "'"]  # Left single quote
        self.char_indexes['\u2019'] = self.char_indexes[
            "'"]  # Right single quote
        self.char_indexes['\u2032'] = self.char_indexes[
            "'"]  # Unicode prime mark (`)
        self.char_indexes['\u201c'] = self.char_indexes[
            '"']  # Left double quote
        self.char_indexes['\u201d'] = self.char_indexes[
            '"']  # Right double quote
        self.char_indexes['\u2013'] = self.char_indexes['-']  # En dash
        self.char_indexes['\u2014'] = self.char_indexes['-']  # Em dash

        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(len(set(self.char_indexes.values())),
                                      embedding_dim)

        self.lstm = nn.LSTM(self.embedding.embedding_dim, embedding_dim, 1,
                            bidirectional=bidirectional)

    def forward(self, idxs):
        ### Data is provided as a PackedSequence of character tokens,
        ### where each sequence in the batch represents a word.

        ## Unpack embeddings
        idxs, lengths = rnn.pad_packed_sequence(idxs, padding_value=0,
                                                batch_first=True)  # (batch_size, sequence_length)
        x = self.embedding(idxs)  # (batch_size, sequence_length, embedding_dim)
        x = rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                     enforce_sorted=False)

        ## LSTM
        _, (h_n, _) = self.lstm(
            x)  # h_n: (num_layers * num_directions, batch, hidden_size)

        if self.lstm.bidirectional:
            # h_n = h_n.view(1, 2, -1, self.embedding_dim) # (num_layers, num_directions, batch_size, hidden_size)
            ## h_n[-1] = last layer, h_n[-1,0] = last layer forward, h_n[-1,1] = last layer backward
            # x = torch.cat((h_n[-1,0],h_n[-1,1]),dim=1) # (batch_size, hidden_size * 2)

            x = torch.cat((h_n[-2], h_n[-1]),
                          dim=1)  # (batch_size, hidden_size * 2)
        else:
            x = h_n[-1]  # (batch_size, hidden_size)

        return x


# Define model
class LSTMCharsEntityBertEmbedModel(BaseANNEntityModule):
    def __init__(self, hidden_size, num_layers, char_embedding_dim,
                 output_labels,
                 bert_name,
                 embed_model,
                 tokenizer_max_length=512):
        super(LSTMCharsEntityBertEmbedModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bert_name = bert_name
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        self.tokenizer_max_length = tokenizer_max_length
        # sentence splitting of standoffs based on tokenizer_max_length/4
        self.sentence_split_params = (tokenizer_max_length // 4, 16)

        self.embed_model = embed_model
        self.labels = LabelEncoder().fit(output_labels)
        self.output_num = len(self.labels.classes_)

        self.char_embedding_dim = char_embedding_dim
        self.char_encoder = CharacterEncoder(self.char_embedding_dim,
                                             bidirectional=True)
        self.embedding_dim = self.embed_model.config.hidden_size
        self.lstm = nn.LSTM(
            self.embedding_dim + 2 * self.char_encoder.embedding_dim,
            self.hidden_size, self.num_layers, bidirectional=True, dropout=0.2)

        self.output_dense = nn.Linear(self.hidden_size * 2, self.output_num)

    def forward(self, inputs):
        ''' Accepts PackedSequence, representing batch of token index sequences. '''

        sentence_batch, batch_char_idxs = inputs

        ## Deal with character embeddings
        ### self.char_encoder(char_idxs) -> (num_words, char_embedding_dim)
        char_emb = rnn.pad_sequence(
            [self.char_encoder(char_idxs) for char_idxs in batch_char_idxs],
            padding_value=0,
            batch_first=True)  # (batch_size, sequence_length, char_embedding_dim)

        ## Deal with word embeddings
        sentence_batch, lengths = rnn.pad_packed_sequence(sentence_batch, padding_value=0,
                                                     batch_first=True)  # (batch_size, sequence_length)
        x = torch.cat((sentence_batch, char_emb),
                      dim=2)  # (batch_size, sequence_length, 768+char_embedding_dim)

        ## LSTM layers
        x = rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                     enforce_sorted=False)
        x, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size*2)
        x, lengths = rnn.pad_packed_sequence(x, padding_value=0,
                                             batch_first=True)
        x = F.relu(x)
        x = self.output_dense(x)  # (batch_size, sequence_length, output_num)
        x = rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                     enforce_sorted=False)

        ## CRF??

        return x

    def make_dataset(self, ann_list):
        return AvgdSentenceEmbedCharDataset(self.tokenizer, self.tokenizer_max_length,
                                            self.char_encoder.embedding_dim,
                                            self.char_encoder.char_indexes,
                                            ann_list,
                                            self.embed_model,
                                            self.labels,
                                            self.sentence_split_params,
                                            cuda=next(self.output_dense.parameters()).is_cuda
                                            )
    def compute_class_weight(self, class_weight, dataset):
        return compute_class_weight(class_weight, classes=self.labels.classes_,
                                    y=[tag for a in dataset.anns for tag in
                                       a.labels])


if __name__ == '__main__':

    from utilities import StopWatch

    stopwatch = StopWatch(memory=True)

    from utilities.torchutils import save_figs  # predict_from_dataloader

    import matplotlib.pyplot as plt

    plt.switch_backend('agg')

    from utilities.argparseactions import ArgumentParser, IterFilesAction, \
        FileAction, DirectoryAction, ListAction
    from utilities.mlutils import split_data
    import os

    from annotations.annmlutils import open_anns

    import sklearn.metrics

    from annotations.models.training import perform_training, evaluate_entities


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
    parser.add_argument('--char-emb',type=int,default=64,help='Length of character embeddings.')
    parser.add_argument('--seed', type=int, default=42, help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.8,0.1,0.1), help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
    parser.add_argument('--types',action=ListAction, help='Annotation types to consider.')
    args = parser.parse_args()

    print("Starting..", flush=True)

    torch.manual_seed(args.seed)
    modelname = os.path.basename(args.modeldir)

    # Read in data
    if not args.types:
        args.types = ['MeasuredValue', 'Constraint', 'ParameterSymbol',
                      'ParameterName', 'ConfidenceLimit', 'ObjectName',
                      'Definition']
    anns = open_anns(args.ann, types=args.types, use_labelled=True)
    ann_train, ann_dev, ann_test = split_data(anns, args.split,
                                              random_state=args.seed)
    output_labels = list(set(l for a in anns for t, l in a))

    stopwatch.tick('Opened all files',report=True)    # Open up BERT stuff
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
        return (LSTMCharsEntityBertEmbedModel(args.hidden, args.layers, args.char_emb,
                                             output_labels, bert_model_name,
                                             bert_model
                                             )
                .float().cuda())


    def make_opt(model):
        adam_lr = 0.001
        return optim.Adam(model.parameters(), lr=adam_lr)


    def make_loss_func(model, train_dataset):
        ignore_index = 999
        class_weights = model.compute_class_weight(args.class_weight,
                                                   train_dataset)
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                        weight=torch.tensor(
                                            class_weights).float().cuda() if args.class_weight else None)
        return model.make_loss_func(criterion, ignore_index=ignore_index)


    def make_metrics(model):
        average = 'macro'
        f1_score = model.make_metric(
            lambda o, t: sklearn.metrics.f1_score(t, o, average=average,
                                                  zero_division=0))
        precision_score = model.make_metric(
            lambda o, t: sklearn.metrics.precision_score(t, o,
                                                         average=average,
                                                         zero_division=0))
        recall_score = model.make_metric(
            lambda o, t: sklearn.metrics.recall_score(t, o, average=average,
                                                      zero_division=0))
        return {'f1': f1_score, 'precision': precision_score,
                'recall': recall_score}


    model, _, history = perform_training(
        make_model, (ann_train, ann_dev, ann_test), args.modeldir,
        make_metrics, make_opt, make_loss_func,
        batch_size=batch_size, epochs=epochs,
        patience=patience, min_delta=min_delta,
        modelname=modelname,
        train_fractions=args.train_fractions,
        stopwatch=stopwatch)

    save_figs(history, modelname, args.modeldir)

    ### TEST MODEL
    class_metrics, overall_metrics = evaluate_entities(model, ann_test,
                                                       batch_size=batch_size)
    class_metrics.to_csv(os.path.join(args.modeldir, 'class_metrics.csv'))
    overall_metrics.to_csv(os.path.join(args.modeldir, 'overall_metrics.csv'))

    stopwatch.tick('Finished evaluation', report=True)

    stopwatch.report()