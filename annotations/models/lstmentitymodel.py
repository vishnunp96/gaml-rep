import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from annotations.datasets import EntityIndexesDataset

from annotations.models.base import BaseANNEntityModule

# Define model
class LSTMEntityModel(BaseANNEntityModule):
	def __init__(self, hidden_size, num_layers, output_labels, token_indexes, fallback_index, embedding=None, embedding_shape=None):
		super(LSTMEntityModel,self).__init__()

		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.token_indexes = token_indexes
		self.fallback_index = fallback_index

		self.labels = LabelEncoder().fit(output_labels)
		self.output_num = len(self.labels.classes_)

		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
		else:
			assert embedding_shape is not None
			num_embeddings,embedding_dim = embedding_shape
			self.embedding = nn.Embedding(num_embeddings, embedding_dim)

		self.projection = nn.Parameter(0.01*torch.diag(torch.randn(self.embedding.embedding_dim)) + 0.001*torch.randn(self.embedding.embedding_dim, self.embedding.embedding_dim))

		self.lstm = nn.LSTM(self.embedding.embedding_dim, self.hidden_size, self.num_layers, bidirectional=True, dropout=0.2)

		self.output_dense = nn.Linear(self.hidden_size*2, self.output_num)

	def forward(self, idxs):
		''' Accepts PackedSequence, representing batch of token index sequences. '''

		## Unpack embeddings and project document matrices
		idxs,lengths = rnn.pad_packed_sequence(idxs, padding_value=0, batch_first=True) # (batch_size, sequence_length)
		x = self.embedding(idxs) # (batch_size, sequence_length, embedding_dim)
		x = torch.matmul(x, self.projection) # Perform projection # (batch_size, sequence_length, embedding_dim)

		## LSTM layers
		x = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
		x,_ = self.lstm(x) # (batch_size, sequence_length, hidden_size*2)
		x,lengths = rnn.pad_packed_sequence(x, padding_value=0, batch_first=True)
		x = F.relu(x)
		x = self.output_dense(x) # (batch_size, sequence_length, output_num)
		x = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

		## CRF??

		return x

	def make_dataset(self, ann_list):
		return EntityIndexesDataset(ann_list, self.token_indexes, self.fallback_index, self.labels, cuda=next(self.output_dense.parameters()).is_cuda)

	def compute_class_weight(self, class_weight, dataset):
		return compute_class_weight(class_weight, classes=self.labels.classes_, y=[tag for a in dataset.anns for tag in a.labels])

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(LSTMEntityModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
		_state_dict['hidden_size'] = self.hidden_size
		_state_dict['num_layers'] = self.num_layers
		_state_dict['output_labels'] = list(self.labels.classes_)
		_state_dict['token_indexes'] = self.token_indexes
		_state_dict['fallback_index'] = self.fallback_index
		_state_dict['embedding_shape'] = (self.embedding.num_embeddings, self.embedding.embedding_dim)
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = LSTMEntityModel(
				_state_dict.pop('hidden_size'),
				_state_dict.pop('num_layers'),
				_state_dict.pop('output_labels'),
				_state_dict.pop('token_indexes'),
				_state_dict.pop('fallback_index'),
				embedding_shape=_state_dict.pop('embedding_shape')
			)
		model.load_state_dict(_state_dict)
		return model

if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	from utilities.torchutils import save_figs # predict_from_dataloader

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction,DirectoryAction,ListAction
	from utilities.mlutils import split_data
	import os

	from annotations.wordembeddings import WordEmbeddings
	from annotations.annmlutils import open_anns

	import sklearn.metrics

	from annotations.models.training import perform_training,evaluate_entities

	def parse_tuple(s):
		return tuple(int(i) for i in s.split('-'))

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
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

	torch.manual_seed(args.seed)

	modelname = os.path.basename(args.modeldir)

	# Read in data
	if not args.types:
		args.types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Definition']
	anns = open_anns(args.ann,types=args.types,use_labelled=True)
	ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)

	if args.eval:
		model = LSTMEntityModel.load_pretained(os.path.join(args.modeldir,modelname+'.pt'))
		stopwatch.tick('Opened all files',report=True)
	else:
		embeddings = WordEmbeddings.open(args.embeddings)

		stopwatch.tick('Opened all files',report=True)

		# Make test/train split
		# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics

		# Model parameters
		output_labels = list(set(l for a in anns for t,l in a))

		# Training parameters
		batch_size = 64
		epochs = 150
		patience = 25
		min_delta = 0.001

		# Generating functions
		def make_model():
			return LSTMEntityModel(args.hidden, args.layers, output_labels, embeddings.indexes, embeddings.fallback_index, embedding=torch.from_numpy(embeddings.values)).float().cuda()
		def make_opt(model):
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
				modelname=modelname,
				train_fractions=args.train_fractions,
				stopwatch=stopwatch)

		save_figs(history, modelname, args.modeldir)

	### TEST MODEL
	class_metrics,overall_metrics = evaluate_entities(model, ann_test, batch_size=batch_size)
	class_metrics.to_csv(os.path.join(args.modeldir,'class_metrics.csv'))
	overall_metrics.to_csv(os.path.join(args.modeldir,'overall_metrics.csv'))

	stopwatch.tick('Finished evaluation',report=True)

	stopwatch.report()
