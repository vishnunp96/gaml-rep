import pandas
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

from annotations.models.base import BaseANNRelationModule
from masters.relation_extraction.datasets.SpecificRelationDataset import SpecificRelationDataset


'''
Still trying with W2V embeddings.
Using the same model as before.
Only change is that each class is separately trained for. 

'''

# Define model
class EntityAtt_Binary(
    BaseANNRelationModule):
    def __init__(self, window_pad, hidden, relation_focus, entity_labels,
                 allowed_entity_pairs, token_indexes, fallback_index,
                 embedding=None, embedding_dim=None, num_embeddings=None):
        super(EntityAtt_Binary,
              self).__init__()

        self.window_pad = window_pad
        self.hidden = tuple(hidden)
        self.token_indexes = token_indexes
        self.fallback_index = fallback_index

        self.relation_focus = relation_focus
        output_labels = [relation_focus, 'none']
        self.labels = LabelEncoder().fit(output_labels)
        self.output_num = len(self.labels.classes_)

        self.entity_labels = LabelBinarizer().fit(entity_labels)
        self.entity_classes_num = len(self.entity_labels.classes_)

        self.allowed_entity_pairs = allowed_entity_pairs

        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding,
                                                          freeze=True)
        else:
            assert embedding_dim is not None and num_embeddings is not None
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.projection = nn.Parameter(0.01 * torch.diag(
            torch.randn(self.embedding.embedding_dim)) + 0.001 * torch.randn(
            self.embedding.embedding_dim, self.embedding.embedding_dim))

        # idea is to pass in entity spans through a separate LSTM, get representation of each entity span
        # then concatenate this with the token embeddings and pass through another LSTM

        self.entity_pool = nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                   bidirectional=False, batch_first=True) # Output embedding_dim


        lstm_input_size = self.embedding.embedding_dim + self.entity_classes_num + 2

        self.pool = nn.LSTM(lstm_input_size, 64, bidirectional=True,
                            batch_first=True)  # Output 128 (64*2)

        input_features = 64 * 2 * 3 + self.entity_classes_num * 2 + 1
        self.dense = []
        for i, h in enumerate(hidden):
            if i == 0:
                self.dense.append(nn.Linear(input_features, h))
            else:
                self.dense.append(nn.Linear(hidden[i - 1], h))
            setattr(self, f'dense{i}', self.dense[-1])

        self.output_dense = nn.Linear(self.hidden[-1], self.output_num)


    def forward(self, x):

        def process_entity_span(spans):
            e, l = spans
            idxs, lengths = rnn.pad_packed_sequence(e, padding_value=0,
                                                    batch_first=True)  # batch_size * sequence_length
            e = self.embedding(
                idxs)  # batch_size * sequence_length * embedding_dim
            e = rnn.pack_padded_sequence(e, lengths, batch_first=True,
                                         enforce_sorted=False)
            e = self.entity_pool(e)[1][0].permute(1, 2, 0).reshape(-1, self.embedding.embedding_dim)
            e = F.relu(e)
            return e # batch_size * embedding_dim

        e_rep = torch.stack((process_entity_span(x[1]),
                             process_entity_span(x[3])), dim= 2) # batch_size * embedding_dim*2

        def process_span(spans, e_rep):
            s, l = spans
            idxs, lengths = rnn.pad_packed_sequence(s, padding_value=0,
                                                    batch_first=True)  # batch_size * sequence_length
            s = self.embedding(
                idxs)  # batch_size * sequence_length * embedding_dim
            s = torch.matmul(s,
                             self.projection)  # Perform projection # batch_size * sequence_length * embedding_dim
            e = torch.matmul(s, e_rep) # batch_size * sequence_length * 2
            l, _ = rnn.pad_packed_sequence(l, padding_value=0,
                                           batch_first=True)  # batch_size * sequence_length * label_encoding
            s = torch.cat((s, l, e),
                          2)  # Concatenate entnaity labels # batch_size, sequence_length, embedding_dim + label_encoding + e_att
            s = rnn.pack_padded_sequence(s, lengths, batch_first=True,
                                         enforce_sorted=False)
            s = self.pool(s)[1][0].permute(1, 2, 0).reshape(-1, 64 * 2)
            s = F.relu(s)
            return s


        x_list = []
        x_list.append(process_span(x[0], e_rep))
        x_list.append(process_span(x[2], e_rep))
        x_list.append(process_span(x[4], e_rep))
        x_list.extend(x[-2:])

        # x = tuple(process_span(i) for i in x[:-2]) + tuple(x[-2:])
        x = torch.cat(x_list, 1)

        for linear in self.dense:
            x = F.relu(linear(x))

        x = self.output_dense(x)
        return x

    def make_dataset(self, ann_list):
        return SpecificRelationDataset(ann_list, self.token_indexes,
                                       self.fallback_index, self.labels,
                                       self.entity_labels, self.relation_focus,
                                       self.allowed_entity_pairs,
                                       self.window_pad, cuda=next(
                self.output_dense.parameters()).is_cuda)

    def compute_class_weight(self, class_weight, dataset):
        return compute_class_weight(class_weight, classes=self.labels.classes_,
                                    y=dataset.y_labels)

    def make_loss_func(self, criterion, ignore_index=999):
        return criterion

    def make_metric(self, func):
        def metric(output, target):
            output = output.cpu().detach().numpy().argmax(1)
            target = target.cpu().detach().numpy()
            return func(output, target)

        return metric

    def predict(self, anns, batch_size=1, inplace=True):
        test_dataset = self.make_dataset(anns)
        relations = predict_from_dataloader(
            self,
            test_dataset.dataloader(batch_size=batch_size, shuffle=False),
            activation=lambda i: F.softmax(i, dim=1).cpu().detach()
        )
        predicted_relations = self.labels.inverse_transform(relations.argmax(1).numpy())
        actual_relations = self.labels.inverse_transform(test_dataset.y)

        return predicted_relations, actual_relations

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     _state_dict = super(IndexSplitSpansDirectedLabelledRelationModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
    #     _state_dict['embedding_dim'] = self.embedding.embedding_dim
    #     _state_dict['num_embeddings'] = self.embedding.num_embeddings
    #     _state_dict['window_pad'] = self.window_pad
    #     _state_dict['hidden'] = tuple(self.hidden)
    #     _state_dict['output_labels'] = list(self.labels.classes_)
    #     _state_dict['entity_labels'] = list(self.entity_labels.classes_)
    #     _state_dict['allowed_relations'] = self.allowed_relations
    #     _state_dict['token_indexes'] = self.token_indexes
    #     _state_dict['fallback_index'] = self.fallback_index
    #     return _state_dict

    # def load_from_state_dict(_state_dict):
    #     ''' Load model from state_dict with arbitrary shape. '''
    #     model = IndexSplitSpansDirectedLabelledRelationModel(
    #             _state_dict.pop('window_pad'),
    #             _state_dict.pop('hidden'),
    #             _state_dict.pop('output_labels'),
    #             _state_dict.pop('entity_labels'),
    #             _state_dict.pop('allowed_relations'),
    #             _state_dict.pop('token_indexes'),
    #             _state_dict.pop('fallback_index'),
    #             embedding_dim=_state_dict.pop('embedding_dim'),
    #             num_embeddings=_state_dict.pop('num_embeddings')
    #         )
    #     model.load_state_dict(_state_dict)
    #     return model


if __name__ == '__main__':

    from utilities import StopWatch

    stopwatch = StopWatch(memory=True)

    from utilities.torchutils import save_figs, \
    predict_from_dataloader  # predict_from_dataloader

    import matplotlib.pyplot as plt

    plt.switch_backend('agg')

    from utilities.argparseactions import ArgumentParser, IterFilesAction, \
        FileAction, DirectoryAction, ListAction
    from annotations.bratutils import StandoffConfigAction
    from utilities.mlutils import split_data
    import os

    from annotations.wordembeddings import WordEmbeddings
    from annotations.annmlutils import open_anns

    import sklearn.metrics

    from annotations.models.training import perform_training, evaluate_relations, \
    evaluate_relations_simple


    def parse_tuple(s):
        return tuple(int(i) for i in s.split('-'))


    parser = ArgumentParser(
        description='Train Keras ANN to predict relations in astrophysical text.')
    parser.add_argument('ann', action=IterFilesAction, recursive=True,
                        suffix='.ann',
                        help='Annotation file or directory containing files (searched recursively).')
    parser.add_argument('embeddings', action=FileAction, mustexist=True,
                        help='Word embeddings file.')
    parser.add_argument('modeldir', action=DirectoryAction, mustexist=False,
                        mkdirs=True,
                        help='Directory to use when saving outputs.')
    parser.add_argument('config', action=StandoffConfigAction,
                        help='Standoff config file for these annotations.')
    parser.add_argument('-w', '--class-weight', action='store_const',
                        const='balanced',
                        help='Flag to indicate that the loss function should be class-balanced for training. Should be used for this model.')
    parser.add_argument('--train-fractions', action='store_true',
                        help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
    parser.add_argument('--window-pad', type=int, default=5,
                        help='Size of window to consider when making relation predictions.')
    parser.add_argument('--hidden', type=parse_tuple, default=(32,),
                        help='Hidden layer widths as a hyphen-separated list, e.g. "1024-512-256".')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.6, 0.2, 0.2),
                        help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
    parser.add_argument('--types', action=ListAction,
                        help='Annotation types to consider.')
    parser.add_argument('--trial', action='store_const',
                        const='balanced',
                        help='Flag to indicate trial mode.')
    parser.add_argument('--relations', action=ListAction,
                        help='Relation types to consider.')
    parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    modelname = os.path.basename(args.modeldir)

    # Read in data
    if not args.types:
        args.types = ['MeasuredValue', 'Constraint', 'ParameterSymbol',
                      'ParameterName', 'ConfidenceLimit', 'ObjectName',
                      'Confidence', 'Measurement', 'Name', 'Property']
    anns = open_anns(args.ann, types=args.types, use_labelled=True)
    embeddings = WordEmbeddings.open(args.embeddings)

    stopwatch.tick('Opened all files', report=True)

    # Make test/train split
    # Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics
    ann_train, ann_dev, ann_test = split_data(anns, args.split,
                                              random_state=args.seed)

    from masters.scripts.relation_label_statistics import \
        get_all_relation_to_entity

    relation_entity_map = get_all_relation_to_entity(anns)
    relations = list(relation_entity_map.keys())
    if args.relations:
        relations = args.relations
    print(f'Going to evaluate for each of relations: {relations}')

    all_class_metrics = []
    for relation in relations:
        # Model parameters
        # window_pad = 5
        output_labels = [relation, 'none']
        entity_labels = list(set(e.type for a in anns for e in a.entities))
        relation_model_name = f'{modelname}_{relation}_entity_att'

        stopwatch.tick(f'Starting {relation} evaluation', report=True)
        print(f'Going to evaluate for relation: {relation}\n\twith entity '
              f'labels: {relation_entity_map[relation]}')
        # Training parameters
        batch_size = args.batch
        epochs = 100
        if args.trial:
            epochs = 2


        # Generating functions
        def make_model():
            return EntityAtt_Binary(
                args.window_pad, args.hidden, relation, entity_labels,
                relation_entity_map[relation], embeddings.indexes,
                embeddings.fallback_index,
                embedding=torch.from_numpy(embeddings.values)).float().cuda()


        def make_opt(model):
            return optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)


        def make_loss_func(model, train_dataset):
            class_weights = model.compute_class_weight(args.class_weight,
                                                       train_dataset)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(
                class_weights).float().cuda() if args.class_weight else None)
            return model.make_loss_func(criterion)


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
            patience=25, min_delta=0.0001,
            modelname=relation_model_name,
            train_fractions=args.train_fractions,
            stopwatch=stopwatch)

        ### TEST MODEL
        class_metrics, overall_metrics = evaluate_relations_simple(model, ann_test,
                                                            batch_size=batch_size)
        class_metrics.to_csv(
            os.path.join(args.modeldir, f'{relation}_class_metrics.csv'))
        all_class_metrics.append(class_metrics)
        # overall_metrics.to_csv(
        #     os.path.join(args.modeldir, f'{relation}_overall_metrics.csv'))

        stopwatch.tick(f'Finished {relation} evaluation', report=True)
    pandas.concat(all_class_metrics).to_csv(
        os.path.join(args.modeldir, 'all_class_metrics.csv'))


    stopwatch.tick(f'Finished all evaluations', report=True)
    stopwatch.report()
