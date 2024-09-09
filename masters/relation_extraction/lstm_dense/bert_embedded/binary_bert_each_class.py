from typing import Optional

import pandas
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim
from utilities.torchutils import predict_from_dataloader

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertConfig, BertForMaskedLM

from annotations.models.base import BaseANNRelationModule
from masters.berts.checkpointing import load_checkpoint
from masters.relation_extraction.datasets.SpecificRelationBertWindowedDataset import SpecificRelationBertWindowedDataset


'''
Still trying with BERT embeddings.
Using the same model as before.
Only change is that each class is separately trained for. 

'''

# Define model
class BertEmbedRelation_BinarySkipWindow(
    BaseANNRelationModule):
    def __init__(self, window_pad, hidden,
                 bert_model_name,
                 tokenizer: Optional[BertTokenizer], bert_model,
                 relation_focus, entity_labels,
                 allowed_entity_pairs,
                 skip_window = tuple(),
                 regularization=False):
        super(BertEmbedRelation_BinarySkipWindow,
              self).__init__()

        self.window_pad = window_pad
        self.hidden = tuple(hidden)
        self.bert_model_name = bert_model_name
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


        # self.projection = nn.Parameter(0.01 * torch.diag(
        #     torch.randn(self.embedding.embedding_dim)) + 0.001 * torch.randn(
        #     self.embedding.embedding_dim, self.embedding.embedding_dim))

        lstm_input_size = self.bert_model.config.hidden_size + self.entity_classes_num

        self.pool = nn.LSTM(lstm_input_size, 64, bidirectional=True,
                            batch_first=True)  # Output 128 (64*2)
        if self.regularization:
            self.dropout = nn.Dropout(p=0.5)

        self.skip_window = skip_window
        num_windows = 5 - len(skip_window)
        input_features = 64 * 2 * num_windows + self.entity_classes_num * 2 + 1
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
        if 0 not in self.skip_window:
            cat_list.append(get_lstm_out(total_embed, torch.zeros_like(boundaries[:, 0]), boundaries[:, 0])) # pre window
        if 1 not in self.skip_window:
            cat_list.append(get_lstm_out(total_embed, boundaries[:, 0], boundaries[:, 1])) # start window
        if 2 not in self.skip_window:
            cat_list.append(get_lstm_out(total_embed, boundaries[:, 1], boundaries[:, 2])) # between window
        if 3 not in self.skip_window:
            cat_list.append(get_lstm_out(total_embed, boundaries[:, 2], boundaries[:, 3])) # end window
        if 4 not in self.skip_window:
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


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        _state_dict = super(BertEmbedRelation_BinarySkipWindow, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
        _state_dict['window_pad'] = self.window_pad
        _state_dict['hidden'] = tuple(self.hidden)
        _state_dict['relation_focus'] = self.relation_focus
        _state_dict['entity_labels'] = list(self.entity_encoder.classes_)
        _state_dict['allowed_entity_pairs'] = self.allowed_entity_pairs
        _state_dict['skip_window'] = self.skip_window
        _state_dict['regularization'] = self.regularization
        _state_dict['bert_model_name'] = self.bert_model_name
        return _state_dict

    def load_from_state_dict(_state_dict):
        ''' Load model from state_dict with arbitrary shape. '''
        model = BertEmbedRelation_BinarySkipWindow(
                _state_dict.pop('window_pad'),
                _state_dict.pop('hidden'),
                _state_dict.pop('bert_model_name'),
                None,
                None,
                _state_dict.pop('relation_focus'),
                _state_dict.pop('entity_labels'),
                _state_dict.pop('allowed_entity_pairs'),
                _state_dict.pop('skip_window'),
                _state_dict.pop('regularization'),
            )
        model.load_state_dict(_state_dict)
        return model



if __name__ == '__main__':

    from utilities import StopWatch

    stopwatch = StopWatch(memory=True)

    import matplotlib.pyplot as plt

    plt.switch_backend('agg')

    from utilities.argparseactions import ArgumentParser, IterFilesAction, \
        FileAction, DirectoryAction
    from utilities.mlutils import split_data
    import os

    from annotations.annmlutils import open_anns

    import sklearn.metrics

    from annotations.models.training import perform_training, evaluate_relations, \
    evaluate_relations_simple


    def parse_tuple(s):
        return tuple(int(i) for i in s.split('-'))


    parser = ArgumentParser(
        description='Train LSTM to predict relations in astrophysical text from BERT embeddings. Trains default embeddings')
    parser.add_argument('ann', action=IterFilesAction, recursive=True,
                        suffix='.ann',
                        help='Annotation file or directory containing files (searched recursively).')
    parser.add_argument('bertmodel',action=FileAction, mustexist=True,help='Bert fine-tuned model.')
    parser.add_argument('modeldir', action=DirectoryAction, mustexist=False,
                        mkdirs=True,
                        help='Directory to use when saving outputs.')
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument('-M','--modelbase',default='bert-base-cased',help='Base model to train from. Defaults to \'bert-base-cased\'.')
    parser.add_argument('-b', '--batch', type=int, default=512,
                        help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.7, 0.1, 0.2),
                        help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    base_model_name = os.path.basename(args.modeldir)

    types = ['MeasuredValue', 'Constraint', 'ParameterSymbol',
            'ParameterName', 'ConfidenceLimit', 'ObjectName',
            'Confidence', 'Measurement', 'Name', 'Property']
    anns = open_anns(args.ann, types=types, use_labelled=True)

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

    opt_configs = {}
    opt_configs['Name'] = {'class_weight': 'balanced', 'hidden': (512,),
                           'window_pad': 15, 'skip': [()]}
    opt_configs['Measurement'] = {'class_weight': None, 'hidden': (128, 64),
                           'window_pad': 9, 'skip': [()]}
    opt_configs['Confidence'] = {'class_weight': None, 'hidden': (128, ),
                           'window_pad': 9, 'skip': [(), (3,)]}
    opt_configs['Property'] = {'class_weight': 'balanced', 'hidden': (128, ),
                           'window_pad': 15, 'skip': [()]}



    from masters.scripts.relation_label_statistics import get_all_relation_to_entity

    relation_entity_map = get_all_relation_to_entity(anns)
    relations = list(relation_entity_map.keys())
    print(f'Going to evaluate for each of relations: {relations}')

    all_class_metrics = []
    for relation in opt_configs.keys():
        configs = opt_configs[relation]
        for skipWindow in configs['skip']:
            skipName = '-'.join([str(i) for i in skipWindow])

            output_labels = [relation, 'none']
            relation_model_name = f'{base_model_name}_{relation}_{skipName}'
            entity_labels = list(set(e.type for a in anns for e in a.entities))

            stopwatch.tick(f'Starting {relation} training', report=True)
            print(f'Going to train for relation: {relation}\n\twith entity '
                  f'labels: {relation_entity_map[relation]}')
            # Training parameters
            batch_size = args.batch
            epochs = 100


            # Generating functions
            def make_model():
                return BertEmbedRelation_BinarySkipWindow(configs['window_pad'], configs['hidden'],
                                                            bert_model_name=bert_model_name,
                                                          tokenizer=tokenizer,
                                                          bert_model=bert_model,
                                                          relation_focus=relation,
                                                          entity_labels=entity_labels,
                                                          skip_window=skipWindow,
                                                          allowed_entity_pairs=relation_entity_map[relation],
                                                          regularization=False).float().cuda()

            def make_opt(model):
                return optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

            def make_loss_func(model, train_dataset):
                class_weights = model.compute_class_weight(configs['class_weight'],
                                                           train_dataset)
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(
                    class_weights).float().cuda() if configs['class_weight'] else None)
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
                stopwatch=stopwatch)

            ### TEST MODEL
            class_metrics, overall_metrics = evaluate_relations_simple(model, ann_test,
                                                                batch_size=batch_size)
            class_metrics.insert(0, 'Classifier', [relation, relation])
            class_metrics.insert(0, 'SkipWindow', [skipName, skipName])
            class_metrics.to_csv(
                os.path.join(args.modeldir, f'{relation}_{skipName}_class_metrics.csv'))
            all_class_metrics.append(class_metrics)

            stopwatch.tick(f'Finished {relation} evaluation', report=True)
    pandas.concat(all_class_metrics).to_csv(
        os.path.join(args.modeldir, 'all_class_metrics.csv'))


    stopwatch.tick(f'Finished all evaluations', report=True)
    stopwatch.report()
