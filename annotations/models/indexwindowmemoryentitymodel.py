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
class IndexWindowMemoryEntityModel(BaseANNEntityModule):
	def __init__(self, window_pad, hidden, output_labels, token_indexes, fallback_index, embedding=None, embedding_dim=None, num_embeddings=None):
		super(IndexWindowMemoryEntityModel,self).__init__()

		self.window_pad = window_pad
		self.hidden = hidden
		#self.output_labels = list(output_labels)
		self.token_indexes = token_indexes
		self.fallback_index = fallback_index

		self.labels = LabelEncoder().fit(output_labels)
		self.output_num = len(self.labels.classes_)

		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
		else:
			assert embedding_dim is not None and num_embeddings is not None
			self.embedding = nn.Embedding(num_embeddings, embedding_dim)

		self.initial_state = nn.Parameter(torch.zeros(self.output_num * self.window_pad))
		self.projection = nn.Parameter(0.01*torch.diag(torch.randn(self.embedding.embedding_dim)) + 0.001*torch.randn(self.embedding.embedding_dim, self.embedding.embedding_dim))

		input_features = (self.embedding.embedding_dim * (self.window_pad*2 + 1)) + self.output_num * self.window_pad
		self.dense1 = nn.Linear(input_features, self.hidden)

		self.output_dense = nn.Linear(self.hidden, self.output_num)

	def forward(self, idxs):
		''' Accepts PackedSequence, representing batch of token index sequences. '''

		idxs,lengths = rnn.pad_packed_sequence(idxs, padding_value=0, batch_first=True) # batch_size * sequence_length
		idxs_pad = F.pad(idxs,(self.window_pad,self.window_pad),value=0)
		x = self.embedding(idxs_pad) # batch_size * sequence_length * embedding_dim
		x = torch.matmul(x, self.projection) # Perform projection # batch_size * sequence_length * embedding_dim

		y = torch.empty(idxs.shape[0], idxs.shape[1], self.output_num, device=x.device) # batch_size * sequence_length * output_num
		previous_output = self.initial_state.repeat(idxs.shape[0], 1).to(device=x.device) # batch_size * (output_num * window_pad)
		for i in range(self.window_pad, x.shape[1] - self.window_pad): # Over sequence_length dimension (ignoring padding)
			e_i = x[:,(i-self.window_pad):(i+self.window_pad+1),:] # batch_size * (window_pad*2+1) * embedding_dim
			e_i = e_i.reshape(idxs.shape[0], -1) # batch_size * ((window_pad*2+1)*embedding_dim)
			x_i = torch.cat((e_i, previous_output), dim=1)
			y_i = F.relu(self.dense1(x_i))
			y_i = self.output_dense(y_i)
			previous_output = torch.cat((previous_output[:,self.output_num:], F.softmax(y_i.detach(),dim=1)), dim=1)
			y[:,i-self.window_pad,:] = y_i

		## Some smoothing, maybe?

		x = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
		return x

	def make_dataset(self, ann_list):
		return EntityIndexesDataset(ann_list, self.token_indexes, self.fallback_index, self.labels, cuda=next(self.output_dense.parameters()).is_cuda)

	def compute_class_weight(self, class_weight, dataset):
		return compute_class_weight(class_weight, self.labels.classes_, [tag for a in dataset.anns for tag in a.labels])

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(IndexWindowMemoryEntityModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
		_state_dict['embedding_dim'] = self.embedding.embedding_dim
		_state_dict['num_embeddings'] = self.embedding.num_embeddings
		_state_dict['window_pad'] = self.window_pad
		_state_dict['hidden'] = self.hidden
		_state_dict['output_labels'] = list(self.labels.classes_)
		_state_dict['token_indexes'] = self.token_indexes
		_state_dict['fallback_index'] = self.fallback_index
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = IndexWindowMemoryEntityModel(
				_state_dict.pop('window_pad'),
				_state_dict.pop('hidden'),
				_state_dict.pop('output_labels'),
				_state_dict.pop('token_indexes'),
				_state_dict.pop('fallback_index'),
				embedding_dim=_state_dict.pop('embedding_dim'),
				num_embeddings=_state_dict.pop('num_embeddings')
			)
		model.load_state_dict(_state_dict)
		return model

if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	from utilities.torchutils import save_figs # predict_from_dataloader

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction,DirectoryAction
	from utilities.mlutils import split_data
	import os

	from annotations.wordembeddings import WordEmbeddings
	from annotations.annmlutils import open_anns

	import sklearn.metrics

	from annotations.models.training import perform_training,evaluate_entities

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
	parser.add_argument('-w','--class-weight',action='store_const',const='balanced',help='Flag to indicate that the loss function should be class-balanced for training.')
	parser.add_argument('--train-fractions',action='store_true',help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
	parser.add_argument('--eval',action='store_true',help='Flag to indicate that the model in modeldir should be loaded and evaluated, rather than a new model be trained.')
	parser.add_argument('--window-pad',type=int,default=5,help='Size of window to consider when making token prediction.')
	parser.add_argument('--hidden',type=int,default=1024,help='Number of neurons in hidden layer.')
	args = parser.parse_args()

	torch.manual_seed(42)

	modelname = os.path.basename(args.modeldir)

	# Read in data
	types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Definition']
	#types = None
	anns = open_anns(args.ann,types=types,use_labelled=True)
	ann_train,ann_dev,ann_test = split_data(anns, (0.8,0.1,0.1), random_state=42)

	if args.eval:
		model = IndexWindowMemoryEntityModel.load_pretained(os.path.join(args.modeldir,modelname+'.pt'))
		stopwatch.tick('Opened all files',report=True)
	else:
		embeddings = WordEmbeddings.open(args.embeddings)

		stopwatch.tick('Opened all files',report=True)

		# Make test/train split
		# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics

		# Model parameters
		output_labels = list(set(l for a in anns for t,l in a))
		#window_pad = 5

		# Training parameters
		batch_size = 256
		epochs = 150
		patience = 25
		min_delta = 0.001

		# Generating functions
		def make_model():
			return IndexWindowMemoryEntityModel(args.window_pad, args.hidden, output_labels, embeddings.indexes, embeddings.fallback_index, embedding=torch.from_numpy(embeddings.values)).float().cuda()
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
	metrics_dataframe = evaluate_entities(model, ann_test, batch_size=batch_size)
	metrics_dataframe.to_csv(os.path.join(args.modeldir,'test_metrics.csv'))

	stopwatch.tick('Finished evaluation',report=True)

	stopwatch.report()
