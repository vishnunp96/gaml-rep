import pandas
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertConfig, BertForMaskedLM

from annotations.models.base import BaseANNRelationModule

from masters.berts.checkpointing import load_checkpoint
from masters.relation_extraction.datasets.SpecificRelationBertWindowedDataset import SpecificRelationBertWindowedDataset

'''
Still trying with W2V embeddings.
Using the same model as before.
Only change is that each class is separately trained for. 

'''

# Define model
class BertEmbedRelation_BinaryWindow(
    BaseANNRelationModule):
    def __init__(self, window_pad, hidden, tokenizer: BertTokenizer, bert_model, relation_focus, entity_labels,
                 allowed_entity_pairs, regularization=False):
        super(BertEmbedRelation_BinaryWindow,
              self).__init__()

        self.window_pad = window_pad
        self.hidden = tuple(hidden)
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.token_limit = 512
        self.sentence_overlap = 16
        self.regularization = regularization

        self.relation_focus = relation_focus
        output_labels = [relation_focus, 'none']
        self.labels = LabelEncoder().fit(output_labels)
        self.output_num = len(self.labels.classes_)

        self.entity_encoder = LabelBinarizer().fit(entity_labels)
        self.entity_classes_num = len(self.entity_encoder.classes_)
        self.allowed_entity_pairs = allowed_entity_pairs


        lstm_input_size = self.bert_model.config.hidden_size + self.entity_classes_num

        self.pool = nn.LSTM(lstm_input_size, 64, bidirectional=True,
                            batch_first=True)  # Output 128 (64*2)
        if self.regularization:
            self.dropout = nn.Dropout(p=0.5)

        input_features = 64 * 2 * 5 + self.entity_classes_num * 2 + 1
        self.dense = []
        for i, h in enumerate(hidden):
            if i == 0:
                self.dense.append(nn.Linear(input_features, h))
            else:
                self.dense.append(nn.Linear(hidden[i - 1], h))
            setattr(self, f'dense{i}', self.dense[-1])

        self.output_dense = nn.Linear(self.hidden[-1], self.output_num)

    def forward(self, x):

        def get_sentence_embeddings(input_ids, attention_mask):
            # batch_size x sequence_length
            total_length = input_ids.shape[1]
            sent_embed = []
            for i in range(0, total_length, self.token_limit - self.sentence_overlap):
                start = i
                end = min(i + self.token_limit, total_length)
                input_ids_slice = input_ids[:, start:end]
                attention_mask_slice = attention_mask[:, start:end]
                with torch.no_grad():
                    outputs = self.bert_model(input_ids=input_ids_slice,
                                            attention_mask=attention_mask_slice)
                    # batch_size x 512 x embedding_dim(768)
                    embed = outputs.hidden_states[-1].float()
                    if i != 0:
                        embed = embed[:, self.sentence_overlap:, ]
                    sent_embed.append(embed)
            # batch_size x sequence_length x embedding_dim
            return torch.cat(sent_embed, 1)

        def get_lstm_out(tot_embed, starts, ends):
            # tot_embed = batch_size x sequence_length x embedding_dim + entity_classes_num
            # starts, ends = batch_size
            embeds = []
            for bi, (start, end) in enumerate(zip(starts, ends)):
                start = start.item()
                end = end.item()
                if start >= end:
                    embeds.append(torch.zeros_like(tot_embed[bi, start:start+1, :]))
                else:
                    embeds.append(tot_embed[bi, start:end, :])
            lstm_in = rnn.pack_sequence(embeds, enforce_sorted=False)
            lstm_out = self.pool(lstm_in)[1][0].permute(1, 2, 0).reshape(-1, 64 * 2)
            return F.relu(lstm_out)

        input_ids, lengths = rnn.pad_packed_sequence(x[0], padding_value=self.tokenizer.pad_token_id, batch_first=True)
        attention_mask, _ = rnn.pad_packed_sequence(x[1], padding_value=0, batch_first=True)
        sentence_embedding = get_sentence_embeddings(input_ids, attention_mask)
        boundaries, _ = rnn.pad_packed_sequence(x[2], padding_value=0, batch_first=True) #batch_size x 4
        sentence_entity_labels, _ = rnn.pad_packed_sequence(x[3], padding_value=0, batch_first=True)
        # batch_size x sequence_length x embedding_dim + entity_classes_num
        total_embed = torch.cat((sentence_embedding, sentence_entity_labels), 2)
        # total_embed = rnn.pack_padded_sequence(total_embed, lengths, batch_first=True, enforce_sorted=False)

        cat_list = []
        cat_list.append(get_lstm_out(total_embed, torch.zeros_like(boundaries[:, 0]), boundaries[:, 0])) # pre window
        cat_list.append(get_lstm_out(total_embed, boundaries[:, 0], boundaries[:, 1])) # start window
        cat_list.append(get_lstm_out(total_embed, boundaries[:, 1], boundaries[:, 2])) # between window
        cat_list.append(get_lstm_out(total_embed, boundaries[:, 2], boundaries[:, 3])) # end window
        cat_list.append(get_lstm_out(total_embed, boundaries[:, 3], lengths)) # post window

        entity_labels = rnn.pad_packed_sequence(x[4], padding_value=0, batch_first=True)[0]
        cat_list.append(entity_labels)
        cat_list.append(x[5].unsqueeze(1))

        xe = torch.cat(cat_list, 1)

        for linear in self.dense:
            xe = F.relu(linear(xe))
            if self.regularization:
                xe = self.dropout(xe)

        xe = self.output_dense(xe)
        return xe

    def make_dataset(self, ann_list):
        return SpecificRelationBertWindowedDataset(ann_list, self.tokenizer, self.labels,
                                       self.entity_encoder, self.relation_focus,
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
        description='Train LSTM to predict relations in astrophysical text from BERT embeddings.')
    parser.add_argument('ann', action=IterFilesAction, recursive=True,
                        suffix='.ann',
                        help='Annotation file or directory containing files (searched recursively).')
    parser.add_argument('bertmodel',action=FileAction, mustexist=True,help='Bert fine-tuned model.')
    parser.add_argument('modeldir', action=DirectoryAction, mustexist=False,
                        mkdirs=True,
                        help='Directory to use when saving outputs.')
    parser.add_argument('-M','--modelbase',default='bert-base-cased',help='Base model to train from. Defaults to \'bert-base-cased\'.')
    parser.add_argument('-b', '--batch', type=int, default=512,
                        help='Batch size for training.')
    parser.add_argument('-w', '--class-weight', action='store_const',
                        const='balanced',
                        help='Flag to indicate that the loss function should be class-balanced for training. Should be used for this model.')
    parser.add_argument('--train-fractions', action='store_true',
                        help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
    parser.add_argument('--window-pad', type=int, default=5,
                        help='Size of window to consider when making relation predictions.')
    parser.add_argument('--hidden', type=parse_tuple, default=(128,128),
                        help='Hidden layer widths as a hyphen-separated list, e.g. "1024-512-256".')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.6, 0.2, 0.2),
                        help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
    parser.add_argument('--types', action=ListAction,
                        help='Annotation types to consider.')
    parser.add_argument('-r','--regularization',action='store_true',help='Flag to add regularization.')
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

    stopwatch.tick('Opened all files', report=True)

    # Make test/train split
    # Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics
    ann_train, ann_dev, ann_test = split_data(anns, args.split,
                                              random_state=args.seed)

    bert_model_name = args.modelbase
    stopwatch.tick(f'Loading bert model, using {bert_model_name}', report=True)
    config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
    bert_model = BertForMaskedLM.from_pretrained(bert_model_name, config=config)
    _, _, _, loaded_model_base = load_checkpoint(args.bertmodel, bert_model)
    bert_model = bert_model.to('cuda')
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    stopwatch.tick(f'Loaded bert model, from {args.bertmodel}', report=True)


    from vishnu.rel_extract.relation_label_statistics import \
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
        relation_model_name = f'{modelname}_{relation}'
        entity_labels = list(set(e.type for a in anns for e in a.entities))

        stopwatch.tick(f'Starting {relation} evaluation', report=True)
        print(f'Going to evaluate for relation: {relation}\n\twith entity '
              f'labels: {relation_entity_map[relation]}')
        # Training parameters
        batch_size = args.batch
        epochs = 100


        # Generating functions
        def make_model():
            return BertEmbedRelation_BinaryWindow(args.window_pad, args.hidden,
                                                  tokenizer=tokenizer,
                                                  bert_model=bert_model,
                                                  relation_focus=relation,
                                                  entity_labels=entity_labels,
                                                  allowed_entity_pairs=relation_entity_map[relation],
                                                  regularization=args.regularization).cuda()


        def make_opt(model):
            if args.regularization:
                return optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)
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
        overall_metrics.to_csv(
            os.path.join(args.modeldir, f'{relation}_overall_metrics.csv'))

        stopwatch.tick(f'Finished {relation} evaluation', report=True)
    pandas.concat(all_class_metrics).to_csv(
        os.path.join(args.modeldir, 'all_class_metrics.csv'))


    stopwatch.tick(f'Finished all evaluations', report=True)
    stopwatch.report()
