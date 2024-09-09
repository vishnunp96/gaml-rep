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

Let us try
    - for each relation
    - for each window
    - what is the most relevant?

'''

# Define model
class IndexSplitSpansDirectedLabelledRelationModel_Binary(
    BaseANNRelationModule):
    def __init__(self, window_pad, hidden, relation_focus, entity_labels,
                 allowed_entity_pairs, token_indexes, fallback_index,
                 embedding, lstm_output=64, num_windows=5, skip_window=tuple()):
        super(IndexSplitSpansDirectedLabelledRelationModel_Binary,
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

        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)

        self.projection = nn.Parameter(0.01 * torch.diag(
            torch.randn(self.embedding.embedding_dim)) + 0.001 * torch.randn(
            self.embedding.embedding_dim, self.embedding.embedding_dim))

        lstm_input_size = self.embedding.embedding_dim + self.entity_classes_num

        self.pool = nn.LSTM(lstm_input_size, lstm_output, bidirectional=True,
                            batch_first=True)  # Output 128 (64*2)
        self.skip_window = skip_window
        if len(self.skip_window) >= 0:
            num_windows -= len(self.skip_window)
        input_features = (lstm_output * 2 * num_windows +
                          self.entity_classes_num * 2 + 1)
        self.dense = []
        for i, h in enumerate(hidden):
            if i == 0:
                self.dense.append(nn.Linear(input_features, h))
            else:
                self.dense.append(nn.Linear(hidden[i - 1], h))
            setattr(self, f'dense{i}', self.dense[-1])

        self.output_dense = nn.Linear(self.hidden[-1], self.output_num)

    def forward(self, x):

        def process_span(spans):
            s, l = spans
            idxs, lengths = rnn.pad_packed_sequence(s, padding_value=0,
                                                    batch_first=True)  # batch_size * sequence_length
            s = self.embedding(
                idxs)  # batch_size * sequence_length * embedding_dim
            s = torch.matmul(s,
                             self.projection)  # Perform projection # batch_size * sequence_length * embedding_dim
            l, _ = rnn.pad_packed_sequence(l, padding_value=0,
                                           batch_first=True)  # batch_size * sequence_length * label_encoding
            s = torch.cat((s, l),
                          2)  # Concatenate entity labels # batch_size, sequence_length, embedding_dim + label_encoding
            s = rnn.pack_padded_sequence(s, lengths, batch_first=True,
                                         enforce_sorted=False)
            s = self.pool(s)[1][0].permute(1, 2, 0).reshape(-1, 64 * 2)
            s = F.relu(s)
            return s

        x = tuple(process_span(i) for wind, i in enumerate(x[:-2])
                  if wind not in self.skip_window) + tuple(x[-2:])
        x = torch.cat(x, 1)

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
    parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--skip', type=parse_tuple,
                        help='windows to skip as hyphen-separated "0-1".')
    parser.add_argument('--relations', action=ListAction,
                        help='Relation types to consider.')
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
    skips = [(i,) for i in range(5)]
    skips.append(tuple())
    if args.skip:
        skips = [args.skip]
        print(f"\t\t Skipping window {skips}")
    for relation in relations:
        relation_metrics = []
        for skipWindow in skips:
            # Model parameters
            # window_pad = 5
            skipName = '-'.join([str(i) for i in skipWindow])
            relation_model_name = f'{modelname}_{relation}_{skipName}'
            output_labels = [relation, 'none']
            entity_labels = list(set(e.type for a in anns for e in a.entities))

            stopwatch.tick(f'Starting {relation} evaluation', report=True)
            print(f'Going to evaluate for relation: {relation}\n\twith entity '
                  f'labels: {relation_entity_map[relation]}')
            # Training parameters
            batch_size = args.batch
            epochs = 200
            if args.trial:
                epochs = 2


            # Generating functions
            def make_model():
                return IndexSplitSpansDirectedLabelledRelationModel_Binary(
                    args.window_pad, args.hidden, relation, entity_labels,
                    relation_entity_map[relation], embeddings.indexes,
                    embeddings.fallback_index, skip_window=skipWindow,
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
            class_metrics.insert(0, 'SkipWindow', [skipName, skipName])
            # class_metrics.to_csv(
            #     os.path.join(args.modeldir, f'{relation}_class_metrics.csv'))
            relation_metrics.append(class_metrics)
            all_class_metrics.append(class_metrics)
            overall_metrics.to_csv(
                os.path.join(args.modeldir, f'{relation}_overall_metrics.csv'))

            stopwatch.tick(f'Finished {relation} evaluation', report=True)
        pandas.concat(relation_metrics).to_csv(
            os.path.join(args.modeldir, f'{relation}_class_metrics.csv'))
    pandas.concat(all_class_metrics).to_csv(
        os.path.join(args.modeldir, 'all_class_metrics.csv'))


    stopwatch.tick(f'Finished all evaluations', report=True)
    stopwatch.report()
