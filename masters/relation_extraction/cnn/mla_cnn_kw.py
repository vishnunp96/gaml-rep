import itertools

import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import numpy
from torch import optim

from annotations.datasets import AnnotationDataset
from annotations.models.base import BaseCNNRelationModule
# reusing what i can
from masters.relation_extraction.cnn.mla_cnn import (RelationEmbedding, MLA_LossFunction, PrimaryAttention, BasicPrimaryAttention, SecondaryAttention)


class RelationEmbedKWindowCNNDataset(AnnotationDataset):
    def __init__(self, ann_list, token_indexes, fallback_index, padding_index, pos_max_distance,
                 relation_focus, allowed_entity_pairs, relation_embedding: RelationEmbedding,
                 k_window,
                 cuda=False):
        super(RelationEmbedKWindowCNNDataset, self).__init__(ann_list, cuda)

        self.relation_focus = relation_focus
        self.allowed_entity_pairs = allowed_entity_pairs
        relation_fallback = relation_embedding.relation_fallback
        self.words_per_sentence = 1024
        # should be num_sentences * num_words_per_sentence
        self.k_window = k_window
        # will now contain num_sentences * num_words_per_sentence * k_window
        self.word_embeds = []
        # will now contain num_sentences * num_words_per_sentence * k_window
        self.pos_e1 = []
        self.pos_e2 = []
        self.pos_max_distance = pos_max_distance
        # should be num_sentences * num_tokens_per_entity
        self.e1 = []
        self.e2 = []
        self.tokens_per_entity = 64
        # should be num_sentences * embed_dim
        # (each relation class is an embedding)
        self.ys = []

        for standoff in ann_list:
            for start, end in itertools.permutations(standoff.entities, 2):
                if (start.type, end.type) in self.allowed_entity_pairs:
                    rel = next((r.type for r in standoff.relations if
                                r.arg1 == start and r.arg2 == end
                                and (relation_focus is None or r.type==relation_focus)), relation_fallback)
                    # relation = class name of label
                    # entire sentence = list of tokens
                    # initialise empty sentence
                    ss, se = standoff.get_token_idxs(start)
                    es, ee = standoff.get_token_idxs(end)
                    sen_token_idx = numpy.array([[padding_index for i in range(2*self.k_window)] for _ in range(self.words_per_sentence)])
                    sen_pos_e1 = numpy.array([[0 for i in range(2*self.k_window)] for _ in range(self.words_per_sentence)])
                    sen_pos_e2 = numpy.array([[0 for i in range(2*self.k_window)] for _ in range(self.words_per_sentence)])
                    sen_e1 = numpy.array([padding_index for _ in range(self.tokens_per_entity)])
                    sen_e2 = numpy.array([padding_index for _ in range(self.tokens_per_entity)])
                    for sen_idx, token in enumerate(standoff.tokens):
                        if sen_idx >= self.words_per_sentence:
                            print(f"Warning: sentence too long for standoff: {standoff.text[:100]}")
                            break
                        token_window = numpy.array([padding_index for i in range(2*self.k_window)])
                        pos_e1_window = numpy.array([0 for i in range(2*self.k_window)])
                        pos_e2_window = numpy.array([0 for i in range(2*self.k_window)])
                        for i in range(len(token_window)):
                            token_idx = sen_idx - self.k_window + i
                            if 0 <= token_idx < len(standoff.tokens):
                                token_window[i] = token_indexes.get(standoff.tokens[token_idx], fallback_index)
                                pos_e1_window[i] = self.get_pos_from(token_idx, ss, se)
                                pos_e2_window[i] = self.get_pos_from(token_idx, es, ee)
                        sen_token_idx[sen_idx] = token_window
                        sen_pos_e1[sen_idx] = pos_e1_window
                        sen_pos_e2[sen_idx] = pos_e2_window
                    for e1_idx, token in enumerate(standoff.tokens[ss:se]):
                        if e1_idx >= self.tokens_per_entity:
                            print(f"Warning: entity too long for entity: {start.text} \n in standoff: {standoff.text[:100]}")
                            break
                        sen_e1[e1_idx] = token_indexes.get(token, fallback_index)
                    for e2_idx, token in enumerate(standoff.tokens[es:ee]):
                        if e2_idx >= self.tokens_per_entity:
                            print(f"Warning: entity too long for entity: {end.text} \n in standoff: {standoff.text[:100]}")
                            break
                        sen_e2[e2_idx] = token_indexes.get(token, fallback_index)

                    self.word_embeds.append(sen_token_idx)
                    self.pos_e1.append(sen_pos_e1)
                    self.pos_e2.append(sen_pos_e2)
                    self.e1.append(sen_e1)
                    self.e2.append(sen_e2)

                    self.ys.append(numpy.array(relation_embedding.get(rel)))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        we = torch.from_numpy(self.word_embeds[idx]).long()
        pe1 = torch.from_numpy(self.pos_e1[idx]).long()
        pe2 = torch.from_numpy(self.pos_e2[idx]).long()
        e1 = torch.from_numpy(self.e1[idx]).long()
        e2 = torch.from_numpy(self.e2[idx]).long()

        y = torch.from_numpy(self.ys[idx]).float()
        return (we, pe1, pe2, e1, e2), y

    def collate_fn(self, sequences):
        xs, ys = zip(*sequences)
        we, pe1, pe2, e1, e2 = tuple(zip(*xs))
        we = rnn.pack_sequence(we, enforce_sorted=False)
        pe1 = rnn.pack_sequence(pe1, enforce_sorted=False)
        pe2 = rnn.pack_sequence(pe2, enforce_sorted=False)
        e1 = rnn.pack_sequence(e1, enforce_sorted=False)
        e2 = rnn.pack_sequence(e2, enforce_sorted=False)
        ys = rnn.pack_sequence(ys, enforce_sorted=False)
        if self.cuda:
            return (we.cuda(), pe1.cuda(), pe2.cuda(), e1.cuda(), e2.cuda()), ys.cuda()
        else:
            return (we, pe1, pe2, e1, e2), ys

    def get_pos_from(self, token_index, entity_start, entity_end):
        ans = self.pos_max_distance
        if token_index < entity_start:
            ans += min(entity_start - token_index, self.pos_max_distance-1)
        elif token_index > entity_end:
            ans -= min(token_index - entity_end, self.pos_max_distance-1)
        return ans



class MLACNNKWindowModel(BaseCNNRelationModule):
    def __init__(self, embedding_values, embedding_indexes, embedding_fallback, embedding_padidx,
                 relation_focus, allowed_entity_pairs, normalised_relation_embeds: RelationEmbedding,
                 k_window,
                 pos_max_distance=50,
                 pos_embedding_dim = 15,
                 kernel_dim = 30,
                 out_channels = 128,
                 cudaFlag=False):
        super(MLACNNKWindowModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_values.float(), freeze=True)
        self.embedding_indxs = embedding_indexes
        self.embedding_fallback = embedding_fallback
        self.embedding_padidx = embedding_padidx
        self.cudaFlag = cudaFlag
        self.k_window = k_window
        gaussian_tensor = torch.normal(0, 1.0, size=(pos_max_distance*2, pos_embedding_dim))
        self.pos_embedding = nn.Embedding.from_pretrained(gaussian_tensor.float(), freeze=True)
        self.pos_max_distance = pos_max_distance
        if relation_focus:
            print("Warning: relation focus is not implemented yet")
        self.relation_focus = relation_focus
        self.allowed_entity_pairs = allowed_entity_pairs
        self.relation_embedding = normalised_relation_embeds
        self.relation_fallback = normalised_relation_embeds.relation_fallback

        # self.primary_attention = PrimaryAttention(self.embedding.embedding_dim, cudaFlag)
        #                                           # + 2 * pos_embedding_dim)
        self.primary_attention = BasicPrimaryAttention(2*self.k_window*(self.embedding.embedding_dim + 2 * pos_embedding_dim)
                                                       + 2*self.embedding.embedding_dim, cudaFlag)
        self.secondary_attention = SecondaryAttention(out_channels,
                                                      len(self.relation_embedding.relations),
                                                      self.relation_embedding, cudaFlag)

        self.conv = nn.Conv2d(1, out_channels,
                              (kernel_dim, self.embedding.embedding_dim + 2 * pos_embedding_dim),
                              stride=(10, 10))
        self.linear_bias = None

    def average_entity_embeds(self, entity_idxs):
        # entity_idxs = (batch_size, tokens_per_entity)
        # dim = embedding_dim
        entity_embeds = []
        for entity in entity_idxs:
            embeds = self.embedding(entity[entity != self.embedding_padidx])
            entity_embeds.append(torch.mean(embeds, dim=0))
        return torch.stack(entity_embeds)


    def forward(self, inputs):
        we, pe1, pe2, e1, e2 = inputs
        we, _ = rnn.pad_packed_sequence(we, padding_value=self.embedding_padidx, batch_first=True)
        pe1, _ = rnn.pad_packed_sequence(pe1, padding_value=self.pos_max_distance, batch_first=True)
        pe2, _ = rnn.pad_packed_sequence(pe2, padding_value=self.pos_max_distance, batch_first=True)
        e1, _ = rnn.pad_packed_sequence(e1, padding_value=self.embedding_padidx, batch_first=True)
        e2, _ = rnn.pad_packed_sequence(e2, padding_value=self.embedding_padidx, batch_first=True)

        # we = (batch_size, seq_len, 2*k_window)
        batch_size, seq_len, _ = we.size()
        embedded_words = self.embedding(we) # batch_size x seq_len x 2k x embedding_dim
        pos1_embedding = self.pos_embedding(pe1) # batch_size x seq_len x 2k x pos_embedding_dim
        pos2_embedding = self.pos_embedding(pe2) # batch_size x seq_len x 2k x pos_embedding_dim
        e1_emb = self.average_entity_embeds(e1) # batch_size x embedding_dim
        e2_emb = self.average_entity_embeds(e2) # batch_size x embedding_dim


        x = torch.cat((embedded_words, pos1_embedding, pos2_embedding), dim=3) # batch_size x seq_len x 2k x (embedding_dim + 2*pos_embedding_dim)
        x = x.view(batch_size, seq_len, -1) # batch_size x seq_len x (2k * (embedding_dim + 2*pos_embedding_dim))
        e1_emb_repeat = e1_emb.unsqueeze(1).repeat(1, x.size(1), 1) # batch_size x seq_len x embedding_dim
        e2_emb_repeat = e2_emb.unsqueeze(1).repeat(1, x.size(1), 1) # batch_size x seq_len x embedding_dim
        entity_added_x = torch.cat((x, e1_emb_repeat, e2_emb_repeat), dim=2) # batch_size x seq_len x (2k(embedding_dim + 2*pos_embedding_dim) + 2*embedding_dim
        attention_weights = self.primary_attention(entity_added_x) # batch_size x seq_len
        x = x * attention_weights.unsqueeze(2) # batch_size x seq_len x (2k * (embedding_dim + 2*pos_embedding_dim))

        x = x.unsqueeze(1) # batch_size x 1 x seq_len x (2k * (embedding_dim + 2*pos_embedding_dim))
        if self.linear_bias is not None:
            x = F.tanh(self.conv(x) + self.linear_bias).squeeze(3) # batch_size x out_channels x out_size
        else:
            x = self.conv(x) # batch_size x out_channels x (out kernel)
            dim0, dim1, dim2, dim3 = x.shape
            x = F.tanh(x.view(dim0, dim1, dim2*dim3)) # batch_size x out_channels x out_size
        sec_attention_weights = self.secondary_attention(x) # (batch_size, out_size, embed_dim)
        '''
        R = conv output = out_channels * out_size        
        G = Rt * U * Wl
        Rt = out_size * out_channels
        Wl = num_classes * embed_dim
        => G = out_size * embed_dim, U = out_channels * num_classes
        R*G = out_channels * embed_dim
        w = argmax over cols in R => embed_dim        
        '''

        x = torch.bmm(x, sec_attention_weights) # batch_size x out_channels x embed_dim
        mx, _ = x.max(dim=1)  # batch_size x embed_dim

        return F.tanh(mx)

    def make_dataset(self, ann_list):
        return RelationEmbedKWindowCNNDataset(ann_list, self.embedding_indxs, self.embedding_fallback,
                                              self.embedding_padidx, self.pos_max_distance, self.relation_focus,
                                              self.allowed_entity_pairs, self.relation_embedding, k_window=self.k_window,
                                              cuda=self.cudaFlag)



def get_all_relation_to_entity(annotations):
    # Builds dictionary
    # Key: relation type
    # Value: set of tuples (Entity1, Entity2)
    relation_entity_set = dict()
    for annotation in annotations:
        for relation in annotation.relations:
            if relation.type not in relation_entity_set:
                relation_entity_set[relation.type] = set()
            relation_entity_set[relation.type].add((relation.arg1.type, relation.arg2.type))
    return relation_entity_set

if __name__ == '__main__':
    # from vishnu.paths import path

    from utilities import StopWatch
    stopwatch = StopWatch(memory=True)
    stopwatch.tick('Starting...', report=True)

    from utilities.argparseactions import ArgumentParser, IterFilesAction, \
        FileAction, DirectoryAction, ListAction
    from utilities.mlutils import split_data
    import os

    from annotations.models.training import mla_cnn_training, evaluate_cnn_relations


    def parse_tuple(s):
        return tuple(int(i) for i in s.split('-'))


    parser = ArgumentParser(
        description='Train MLA CNN to predict relations from entities in astrophysical text.')
    parser.add_argument('ann', action=IterFilesAction, recursive=True,
                        suffix='.ann',
                        help='Annotation file or directory containing files (searched recursively).')
    parser.add_argument('embeddings', action=FileAction, mustexist=True,
                        help='Word embeddings file.')
    parser.add_argument('modeldir', action=DirectoryAction, mustexist=False,
                        mkdirs=True,
                        help='Directory to use when saving outputs.')
    parser.add_argument('-w', '--class-weight', action='store_const',
                        const='balanced',
                        help='Flag to indicate that the loss function should be class-balanced for training.')
    parser.add_argument('--train-fractions', action='store_true',
                        help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
    parser.add_argument('--eval', action='store_true',
                        help='Flag to indicate that the model in modeldir should be loaded and evaluated, rather than a new model be trained.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.8, 0.1, 0.1),
                        help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
    parser.add_argument('--types', action=ListAction,
                        help='Annotation types to consider.')
    parser.add_argument('-b', '--batch', type=int, default=4, help='Batch size.')
    parser.add_argument('--kwindow', type=int, default=2,
                        help='Random number seed for this training run.')
    parser.add_argument('--kernel', type=int, default=50,
                        help='Random number seed for this training run.')
    parser.add_argument('--outchannels', type=int, default=128,
                        help='Random number seed for this training run.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Random number seed for this training run.')


    # k_window = args.kwindow, kernel_dim = args.kernel, out_channels = args.outchannels
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    modelname = os.path.basename(args.modeldir)

    from annotations.wordembeddings import WordEmbeddings
    from annotations.annmlutils import open_anns


    embeddings = WordEmbeddings.open(args.embeddings)
    # embeddings = WordEmbeddings.open(path['embeddings'])


    # Read in data
    if not args.types:
        args.types = ['MeasuredValue', 'Constraint', 'ParameterSymbol', 'ParameterName',
             'ConfidenceLimit', 'ObjectName', 'Confidence', 'Measurement',
             'Name', 'Property']
    anns = open_anns(args.ann, types=args.types, use_labelled=True)
    ann_train, ann_dev, ann_test = split_data(anns, args.split,
                                              random_state=args.seed)

    stopwatch.tick('Opened all annotations', report=True)
    # types = ['MeasuredValue', 'Constraint', 'ParameterSymbol', 'ParameterName',
    #          'ConfidenceLimit', 'ObjectName', 'Confidence', 'Measurement',
    #          'Name', 'Property']
    # anns = open_anns(path['organn'], types=types, use_labelled=True)
    # ann_train, ann_dev, ann_test = split_data(anns, (0.8, 0.1, 0.1),
    #                                           random_state=42)
    relation_entity_set = get_all_relation_to_entity(anns)
    relations = list(relation_entity_set.keys())
    allowed_entity_pairs = set()
    for relation in relations:
        allowed_entity_pairs.update(relation_entity_set[relation])
    relation_fallback = 'none'
    relations.append(relation_fallback)
    relation_embedding = RelationEmbedding(relations, relation_fallback, embeddings)

    stopwatch.tick('Opened all relations', report=True)

    def make_model():
        return MLACNNKWindowModel(torch.from_numpy(embeddings.values),
                           embeddings.indexes, embeddings.fallback_index,
                           embeddings.padding_index,
                           None, allowed_entity_pairs, relation_embedding,
                                  k_window=args.kwindow, kernel_dim=args.kernel, out_channels=args.outchannels,
                           cudaFlag=True).float().cuda()

    def make_opt(model):
        adam_lr = 0.01
        return optim.Adam(model.parameters(), lr=adam_lr)
        # sgd_lr = 0.01
        # return optim.SGD(model.parameters(), lr=sgd_lr)
    def make_loss_func(relation_embedding):
        return MLA_LossFunction(relation_embedding)
    def make_metrics(model):
        def get_classes(vectors):
            classes = []
            for vector in vectors:
                classes.append(relation_embedding.closest_class(vector))
            return classes

        def compute_f1_score(output, target):
            # output and target are both (batch_size, embed_dim)
            target_classes = get_classes(target)
            output_classes = get_classes(output)
            return sklearn.metrics.f1_score(target_classes, output_classes, average='macro',
                                            zero_division=0)

        f1_score = model.make_metric(compute_f1_score)

        def compute_precision(output, target):
            # output and target are both (batch_size, embed_dim)
            target_classes = get_classes(target)
            output_classes = get_classes(output)
            return sklearn.metrics.precision_score(target_classes, output_classes, average='macro',
                                            zero_division=0)
        precision_score = model.make_metric(compute_precision)

        def compute_recall(output, target):
            # output and target are both (batch_size, embed_dim)
            target_classes = get_classes(target)
            output_classes = get_classes(output)
            return sklearn.metrics.recall_score(target_classes, output_classes, average='macro',
                                            zero_division=0)
        recall_score = model.make_metric(compute_recall)

        return {'f1': f1_score, 'precision': precision_score,
                'recall': recall_score}


    # Training parameters
    batch_size = args.batch
    epochs = args.epochs
    patience = 10
    min_delta = 0.001

    stopwatch.tick('Starting training', report=True)
    model, _, history = mla_cnn_training(
        make_model, (ann_train, ann_dev, ann_test), args.modeldir,
        make_metrics, make_opt, make_loss_func,
        batch_size=batch_size, epochs=epochs,
        relation_embedding=relation_embedding,
        patience=patience, min_delta=min_delta,
        modelname=modelname,
        train_fractions=args.train_fractions,
        stopwatch=stopwatch)


    class_metrics, overall_metrics = evaluate_cnn_relations(model, ann_test,
                                                            relations,
                                                       batch_size=batch_size)
    class_metrics.to_csv(os.path.join(args.modeldir, 'class_metrics.csv'))
    overall_metrics.to_csv(os.path.join(args.modeldir, 'overall_metrics.csv'))

    stopwatch.tick('Finished evaluation', report=True)

    stopwatch.report()
