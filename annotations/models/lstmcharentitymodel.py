import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from annotations.datasets import EntityIndexesCharsDataset

from annotations.models.base import BaseANNEntityModule

class CharacterEncoder(nn.Module):
	def __init__(self, embedding_dim, bidirectional=False):
		super(CharacterEncoder,self).__init__()

		self.char_indexes = {c:(i+1) for i,c in enumerate("$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~\'\"")}
		self.char_indexes[None] = 0 # Unknown character (fallback)
		## Replacements for some common non-ascii Unicode characters
		self.char_indexes['\u2018'] = self.char_indexes["'"] # Left single quote
		self.char_indexes['\u2019'] = self.char_indexes["'"] # Right single quote
		self.char_indexes['\u2032'] = self.char_indexes["'"] # Unicode prime mark (`)
		self.char_indexes['\u201c'] = self.char_indexes['"'] # Left double quote
		self.char_indexes['\u201d'] = self.char_indexes['"'] # Right double quote
		self.char_indexes['\u2013'] = self.char_indexes['-'] # En dash
		self.char_indexes['\u2014'] = self.char_indexes['-'] # Em dash

		self.embedding_dim = embedding_dim

		self.embedding = nn.Embedding(len(set(self.char_indexes.values())), embedding_dim)

		self.lstm = nn.LSTM(self.embedding.embedding_dim, embedding_dim, 1, bidirectional=bidirectional)

	def forward(self, idxs):
		### Data is provided as a PackedSequence of character tokens,
		### where each sequence in the batch represents a word.

		## Unpack embeddings
		idxs,lengths = rnn.pad_packed_sequence(idxs, padding_value=0, batch_first=True) # (batch_size, sequence_length)
		x = self.embedding(idxs) # (batch_size, sequence_length, embedding_dim)
		x = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

		## LSTM
		_,(h_n,_) = self.lstm(x) # h_n: (num_layers * num_directions, batch, hidden_size)

		if self.lstm.bidirectional:
			#h_n = h_n.view(1, 2, -1, self.embedding_dim) # (num_layers, num_directions, batch_size, hidden_size)
			## h_n[-1] = last layer, h_n[-1,0] = last layer forward, h_n[-1,1] = last layer backward
			#x = torch.cat((h_n[-1,0],h_n[-1,1]),dim=1) # (batch_size, hidden_size * 2)

			x = torch.cat((h_n[-2],h_n[-1]),dim=1) # (batch_size, hidden_size * 2)
		else:
			x = h_n[-1] # (batch_size, hidden_size)

		return x

# Define model
class LSTMCharsEntityModel(BaseANNEntityModule):
	def __init__(self, hidden_size, num_layers, char_embedding_dim, output_labels, token_indexes, fallback_index, word_embedding=None, word_embedding_shape=None):
		super(LSTMCharsEntityModel,self).__init__()

		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.token_indexes = token_indexes
		self.fallback_index = fallback_index

		self.labels = LabelEncoder().fit(output_labels)
		self.output_num = len(self.labels.classes_)

		if word_embedding is not None:
			self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=True)
		else:
			assert word_embedding_shape is not None
			num_word_embeddings,word_embedding_dim = word_embedding_shape
			self.word_embedding = nn.Embedding(num_word_embeddings, word_embedding_dim)

		self.char_encoder = CharacterEncoder(char_embedding_dim, bidirectional=True)

		self.word_projection = nn.Parameter(0.01*torch.diag(torch.randn(self.word_embedding.embedding_dim)) + 0.001*torch.randn(self.word_embedding.embedding_dim, self.word_embedding.embedding_dim))

		self.lstm = nn.LSTM(self.word_embedding.embedding_dim + 2*self.char_encoder.embedding_dim, self.hidden_size, self.num_layers, bidirectional=True, dropout=0.2)

		self.output_dense = nn.Linear(self.hidden_size*2, self.output_num)

	def forward(self, inputs):
		''' Accepts PackedSequence, representing batch of token index sequences. '''

		word_idxs,batch_char_idxs = inputs

		## Deal with character embeddings
		### self.char_encoder(char_idxs) -> (num_words, char_embedding_dim)
		char_emb = rnn.pad_sequence([self.char_encoder(char_idxs) for char_idxs in batch_char_idxs], padding_value=0, batch_first=True) # (batch_size, sequence_length, char_embedding_dim)

		## Deal with word embeddings
		word_idxs,lengths = rnn.pad_packed_sequence(word_idxs, padding_value=0, batch_first=True) # (batch_size, sequence_length)
		word_emb = self.word_embedding(word_idxs) # (batch_size, sequence_length, word_embedding_dim)
		word_proj = torch.matmul(word_emb, self.word_projection) # Perform projection # (batch_size, sequence_length, word_embedding_dim)

		x = torch.cat((word_proj,char_emb), dim=2) # (batch_size, sequence_length, word_embedding_dim+char_embedding_dim)

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
		return EntityIndexesCharsDataset(ann_list, self.token_indexes, self.char_encoder.char_indexes, self.fallback_index, self.labels, cuda=next(self.output_dense.parameters()).is_cuda)

	def compute_class_weight(self, class_weight, dataset):
		return compute_class_weight(class_weight, classes=self.labels.classes_, y=[tag for a in dataset.anns for tag in a.labels])

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(LSTMCharsEntityModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
		_state_dict['hidden_size'] = self.hidden_size
		_state_dict['num_layers'] = self.num_layers
		_state_dict['char_embedding_dim'] = self.char_encoder.embedding_dim
		_state_dict['output_labels'] = list(self.labels.classes_)
		_state_dict['token_indexes'] = self.token_indexes
		_state_dict['fallback_index'] = self.fallback_index
		_state_dict['word_embedding_shape'] = (self.word_embedding.num_embeddings, self.word_embedding.embedding_dim)
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = LSTMCharsEntityModel(
				_state_dict.pop('hidden_size'),
				_state_dict.pop('num_layers'),
				_state_dict.pop('char_embedding_dim'),
				_state_dict.pop('output_labels'),
				_state_dict.pop('token_indexes'),
				_state_dict.pop('fallback_index'),
				word_embedding_shape=_state_dict.pop('word_embedding_shape')
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
	parser.add_argument('--hidden',type=int,default=512,help='Number of neurons in hidden layers.')
	parser.add_argument('--layers',type=int,default=2,help='Number of layers of LSTM cells.')
	parser.add_argument('--char-emb',type=int,default=128,help='Length of character embeddings.')
	parser.add_argument('--seed', type=int, default=42, help='Random number seed for this training run.')
	parser.add_argument('--split', type=parse_tuple, default=(0.8,0.1,0.1), help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
	parser.add_argument('--types',action=ListAction, help='Annotation types to consider.')
	parser.add_argument('-b', '--batch', type=int, default=64,
						help='Batch size.')

	args = parser.parse_args()

	torch.manual_seed(args.seed)

	modelname = os.path.basename(args.modeldir)

	# Read in data
	if not args.types:
		args.types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Definition']
	anns = open_anns(args.ann,types=args.types,use_labelled=True)
	ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)

	if args.eval:
		model = LSTMCharsEntityModel.load_pretained(os.path.join(args.modeldir,modelname+'.pt'))
		stopwatch.tick('Opened all files',report=True)
	else:
		embeddings = WordEmbeddings.open(args.embeddings)

		stopwatch.tick('Opened all files',report=True)

		# Make test/train split
		# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics

		# Model parameters
		output_labels = list(set(l for a in anns for t,l in a))

		# Training parameters
		batch_size = args.batch
		epochs = 150
		patience = 25
		min_delta = 0.001

		# Generating functions
		def make_model():
			return LSTMCharsEntityModel(args.hidden, args.layers, args.char_emb, output_labels, embeddings.indexes, embeddings.fallback_index, word_embedding=torch.from_numpy(embeddings.values)).float().cuda()
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
			f1_score = model.make_metric(lambda o,t: sklearn.metrics.f1_score(t, o, average=average, zero_division=0))
			precision_score = model.make_metric(lambda o,t: sklearn.metrics.precision_score(t, o, average=average, zero_division=0))
			recall_score = model.make_metric(lambda o,t: sklearn.metrics.recall_score(t, o, average=average, zero_division=0))
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
