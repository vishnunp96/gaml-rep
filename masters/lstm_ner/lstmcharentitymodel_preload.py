from copy import deepcopy

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

from utilities.torchutils import unpack_sequence
from annotations.brattowindow import StandoffLabels
from annotations.models.predictionutils import token_labels_to_entities_map
from utilities.torchutils import predict_from_dataloader

'''
separate bits for numeric, rarity
'''
class SentenceEmbedCharBoostedPreloadDataset(AnnotationDataset):
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
        super(SentenceEmbedCharBoostedPreloadDataset, self).__init__(ann_list, cuda)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.class_template = {"O": outside_class, "B": begin_class,
                               "I": inside_class}
        self.cuda = cuda
        self.label_encoder = label_encoder
        self.char_indexes = char_indexes
        self.bert_model = bert_model

        self.sentence_split_length, self.sentence_overlap = sentence_split_params

        words, annotations, self.classes = self.load_annotations(
            ann_list)
        print(f"Finished reading annotations. Classes: {self.classes}")
        # sentence_tensor = num_sentences * (['input_ids', 'token_type_ids', 'attention_mask'] X 1 * tokenizer_max_length)
        (self.sentence_tensors, self.sentence_char_indices, self.numeric_list,
         self. rarity_list, self.labels, self.index_maps) = self.tokenize_annotations(
            words,
            annotations)
        if self.cuda:
            print("Moving vars to GPU", flush=True)
            self.sentence_tensors = [t.to('cuda') for t in
                                     self.sentence_tensors]
        print(
            f"Now to get embeddings.. Found sentences: {len(self.sentence_tensors)}",
            flush=True)
        # num_sentences * num_tokens * 768
        self.embeddings = self.get_embeddings()
        # self.embeddings = []
        print(f"Finished making tokens and labels. Length: {self.__len__()}",
              flush=True)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sentence_embeds, char_indices, labels = self.embeddings[idx], \
            self.sentence_char_indices[idx], self.labels[idx]
        numeric_flags, rarity_vals = self.numeric_list[idx], self.rarity_list[idx]
        # return ((sentence_embeds.float(), char_indices.long()), torch.from_numpy(labels).long())
        return ((sentence_embeds.float(), rnn.pack_sequence([torch.from_numpy(i).long() for i in char_indices],False),
                 torch.from_numpy(numeric_flags).long(), torch.from_numpy(rarity_vals).float()),
                torch.from_numpy(labels).long())


    def collate_fn(self, sequences):
        xs, labels = zip(*sequences)
        sent_embeds, char_indexes, numeric_flags, rarity_vals = tuple(zip(*xs))
        if self.cuda:
            return ((rnn.pack_sequence(sent_embeds, False).cuda(),
                    [i.cuda() for i in char_indexes],
                     rnn.pack_sequence(numeric_flags, False).cuda(),
                     rnn.pack_sequence(rarity_vals, False).cuda() ),
                    rnn.pack_sequence(labels, False).cuda())
        else:
            return ((
            rnn.pack_sequence(sent_embeds, False),
            [i for i in char_indexes],
            rnn.pack_sequence(numeric_flags, False),
            rnn.pack_sequence(rarity_vals, False)),
                    rnn.pack_sequence(labels,False))


    def load_annotations(self, standoff_list):
        # need to use standoffs instead of directory.
        # need to split appropriately as well.
        words = []
        annotations = []
        classes = set()

        for standoff in standoff_list:
            words.append(standoff.tokens)
            annotations.append(standoff.labels)
            classes.update(annotations[-1])

        return words, annotations, classes


    def tokenize_annotations(self, words, annotations):
        def is_numeric(words):
            return any([word.replace('.','').isnumeric() for word in words])

        def rarity_array(words):
            counts = dict()
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            rarity = numpy.array([1/counts[word] for word in words])
            # e_x = np.exp(rarity - np.max(rarity))
            # return e_x / e_x.sum()
            return rarity


        sentence_tensors = []
        label_list = []
        sentence_char_indexes = []
        numeric_list = []
        rarity_list = []
        beg_token_char = numpy.array([self.char_indexes.get(c, self.char_indexes[None])
                                      for c in self.tokenizer.cls_token])
        end_token_char = numpy.array([self.char_indexes.get(c, self.char_indexes[None])
                                      for c in self.tokenizer.sep_token])
        index_maps = []
        for word_line, ann_line in zip(words, annotations):
            # get tokens
            sentence = " ".join(word_line)
            tokenized_tensor = self.tokenizer(sentence, return_tensors='pt',
                                              padding='max_length')
            labels = ([self.class_template["O"]] *
                      torch.sum(tokenized_tensor['attention_mask'],
                                dim=1).item())

            numeric_flags = numpy.array([0 for i in range(len(labels))])
            rarity_vals = numpy.array([0 for i in range(len(labels))])
            label_index = 0
            index_map = {}
            sent_char_index = [beg_token_char]
            rarity = rarity_array(word_line)
            # adding character embeddings for each word after tokenizing
            # adding along with label
            for wi, (word, ann) in enumerate(zip(word_line, ann_line)):
                word_char_index = numpy.array([self.char_indexes.get(c, self.char_indexes[None]) for c in word])
                number_flag = False
                if '.' in word:
                    number_flag = is_numeric(word_line[max(wi-1,0):wi+2])
                tokens = self.tokenizer.tokenize(word)
                for i, token in enumerate(tokens):
                    # token_char_index = numpy.array(
                    #     [self.char_indexes.get(c, self.char_indexes[None]) for c
                    #      in token])
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if label_index >= len(labels):
                        print(
                            "Warning: Labels are not fitting when mapping to tokenized versions of sentences.")
                        break
                    while tokenized_tensor['input_ids'][0][
                        label_index] != token_id:
                        label_index += 1
                    if i == 0:
                        labels[label_index] = ann
                        numeric_flags[label_index] = number_flag
                        rarity_vals[label_index] = rarity[wi]
                        index_map[wi] = label_index
                    else:
                        labels[label_index] = ann.replace(
                            self.class_template["B"], self.class_template["I"])
                    # labels[label_index] = ann.replace("begin", "inside")
                    sent_char_index.append(word_char_index)
                    # token wise character embedding did worse in all classes
                    # sent_char_index.append(token_char_index)
                    label_index += 1
            while len(sent_char_index) < len(labels):
                sent_char_index.append(end_token_char)
            sentence_char_indexes.append(sent_char_index)
            sentence_tensors.append(tokenized_tensor)
            label_list.append(self.label_encoder.transform(labels))
            rarity_list.append(rarity_vals)
            numeric_list.append(numeric_flags)
            index_maps.append(index_map)

        return sentence_tensors, sentence_char_indexes, numeric_list, rarity_list, label_list, index_maps

    def get_embeddings(self):
        embeddings = []
        for sentence_tensor in self.sentence_tensors:
            length = torch.sum(sentence_tensor['attention_mask'], dim=1).item()
            embed = []
            for i in range(0, length, self.tokenizer_max_length - self.sentence_overlap):
                with torch.no_grad():
                    output = self.bert_model(
                        sentence_tensor['input_ids'][:, i:i + self.tokenizer_max_length],
                        sentence_tensor['attention_mask'][:, i:i + self.tokenizer_max_length])
                # final hidden state = 1 * tokenizer_max_length * 768
                cur_len = torch.sum(sentence_tensor['attention_mask'][:, i:i + self.tokenizer_max_length], dim=1).item()
                if i == 0:
                    embed.append(output.hidden_states[-1][0][:cur_len].float())
                else:
                    embed.append(output.hidden_states[-1][0][self.sentence_overlap:cur_len].float())
            embeddings.append(torch.cat(embed, dim=0))
        return embeddings


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
class LSTMCharsEntityBertEmbedNodupModel(BaseANNEntityModule):
    def __init__(self, hidden_size, num_layers, char_embedding_dim,
                 output_labels,
                 bert_name,
                 embed_model,
                 numeric_flags=False,
                 rarity_flags=False,
                 tokenizer_max_length=512):
        super(LSTMCharsEntityBertEmbedNodupModel, self).__init__()

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
        self.numeric_flags = numeric_flags
        self.rarity_flags = rarity_flags
        self.lstm = nn.LSTM(
            self.embedding_dim + 2 * self.char_encoder.embedding_dim + numeric_flags + rarity_flags,
            self.hidden_size, self.num_layers, bidirectional=True, dropout=0.2)

        self.output_dense = nn.Linear(self.hidden_size * 2, self.output_num)

    def forward(self, inputs):
        ''' Accepts PackedSequence, representing batch of token index sequences. '''

        sentence_batch, batch_char_idxs, numeric_flags, rarity_vals = inputs

        ## Deal with character embeddings
        ### self.char_encoder(char_idxs) -> (num_words, char_embedding_dim)
        char_emb = rnn.pad_sequence(
            [self.char_encoder(char_idxs) for char_idxs in batch_char_idxs],
            padding_value=0,
            batch_first=True)  # (batch_size, sequence_length, char_embedding_dim)

        ## Deal with word embeddings
        sentence_batch, lengths = rnn.pad_packed_sequence(sentence_batch, padding_value=0,
                                                     batch_first=True)  # (batch_size, sequence_length)
        cat_list = [sentence_batch, char_emb]
        if self.numeric_flags:
            numeric_flags, _ = rnn.pad_packed_sequence(numeric_flags, padding_value=0, batch_first=True)
            cat_list.append(numeric_flags.unsqueeze(-1))
        if self.rarity_flags:
            rarity_vals, _ = rnn.pad_packed_sequence(rarity_vals, padding_value=0, batch_first=True)
            cat_list.append(rarity_vals.unsqueeze(-1))
        # for i, c in enumerate(cat_list):
        #     print(i, c.shape)
        x = torch.cat(cat_list,
                      dim=2)  # (batch_size, sequence_length, 768+char_embedding_dim + 2)

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
        return SentenceEmbedCharBoostedPreloadDataset(self.tokenizer, self.tokenizer_max_length,
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


    def predict(self, anns, batch_size=1, inplace=True):
        if not inplace:
            anns = deepcopy(
                anns)  # [(p,i,Standoff(a.text,[],[],[],[])) for p,i,a in anns]
        entity_dataset = self.make_dataset([StandoffLabels(a) for p, i, a in anns])

        predictions = []
        for batch_x, _ in entity_dataset.dataloader(batch_size=batch_size,
                                                    shuffle=False):
            output = self(batch_x)
            outputs = [F.softmax(i, dim=1).cpu().detach().numpy() for i in
                       unpack_sequence(output)]
            predictions.extend(outputs)

        # Load predictions into Standoff objects
        for ann, index_map, pred in zip(entity_dataset.anns, entity_dataset.index_maps,
                                        predictions):
            for ent_type, start, end in token_labels_to_entities_map(
                    self.labels.inverse_transform(pred.argmax(1)), ann.token_idxs,
                    index_map):
                ann.standoff.entity(ent_type, start, end)

        return anns


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

    from annotations.models.training import perform_training, evaluate_entities_bert


    def parse_tuple(s):
        return tuple(int(i) for i in s.split('-'))

    parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
    parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
    parser.add_argument('bertmodel',action=FileAction, mustexist=True,help='Bert fine-tuned model.')
    parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
    parser.add_argument('-M','--modelbase',default='bert-base-cased',help='Base model to train from. Defaults to \'bert-base-cased\'.')
    parser.add_argument('-w','--class-weight',action='store_const',const='balanced',help='Flag to indicate that the loss function should be class-balanced for training.')
    parser.add_argument('--train-fractions',action='store_true',help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
    parser.add_argument('--eval',action='store_true',help='Flag to indicate that the model in modeldir should be loaded and evaluated, rather than a new model be trained.')
    parser.add_argument('--hidden',type=int,default=512,help='Number of neurons in hidden layers.')
    parser.add_argument('--layers',type=int,default=2,help='Number of layers of LSTM cells.')
    parser.add_argument('--char-emb',type=int,default=128,help='Length of character embeddings.')
    parser.add_argument('--seed', type=int, default=42, help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.8,0.1,0.1), help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
    parser.add_argument('--types',action=ListAction, help='Annotation types to consider.')
    parser.add_argument('-n', '--numeric',action='store_true',help='Flag to indicate that numeric flags should be included in the model.')
    parser.add_argument('-r', '--rarity',action='store_true',help='Flag to indicate that rarity flags should be included in the model.')
    parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size.')

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
    bert_model_name = args.modelbase
    config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
    bert_model = BertForMaskedLM.from_pretrained(bert_model_name, config=config)
    _, _, _, loaded_model_base = load_checkpoint(args.bertmodel, bert_model)


    stopwatch.tick('Loaded the bert model', report=True)
    print("Loaded the bert model", flush=True)

    # Make test/train split
    # Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics

    # Model parameters

    # Training parameters
    batch_size = args.batch
    epochs = 150
    patience = 25
    min_delta = 0.001


    # Generating functions
    def make_model():
        return (LSTMCharsEntityBertEmbedNodupModel(args.hidden, args.layers, args.char_emb,
                                                   output_labels, bert_model_name,
                                                   bert_model, numeric_flags=args.numeric,
                                                   rarity_flags=args.rarity
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
    class_metrics, overall_metrics = evaluate_entities_bert(model, ann_test,
                                                       batch_size=batch_size)
    class_metrics.to_csv(os.path.join(args.modeldir, 'class_metrics.csv'))
    overall_metrics.to_csv(os.path.join(args.modeldir, 'overall_metrics.csv'))

    stopwatch.tick('Finished evaluation', report=True)

    stopwatch.report()
