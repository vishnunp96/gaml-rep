import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from gaml.utilities.torchutils import unpack_sequence
from gaml.annotations.datasets import EntityIndexesDataset

from gaml.annotations.models.base import BaseANNEntityModule

# Define model
class IndexSmoothedWindowMemoryEntityModel(BaseANNEntityModule):
	def __init__(self, window_pad, hidden, output_labels, token_indexes, fallback_index, embedding=None, embedding_dim=None, num_embeddings=None):
		super(IndexSmoothedWindowMemoryEntityModel,self).__init__()

		self.window_pad = window_pad
		self.hidden = tuple(hidden)
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
		self.dense = []
		for i,h in enumerate(hidden):
			if i==0:
				self.dense.append(nn.Linear(input_features,h))
			else:
				self.dense.append(nn.Linear(hidden[i-1],h))
			setattr(self, f'dense{i}', self.dense[-1])

		self.output_dense = nn.Linear(self.hidden[-1], self.output_num)

		##self.kernel = nn.Parameter(torch.tensor([0.25,0.5,0.25]).repeat(self.output_num,1,1))
		#self.kernel = nn.Parameter(torch.tensor([0.0,1.0,0.0]).repeat(self.output_num,1,1))
		#self.padding = 2*(int((self.kernel.shape[-1]-1)/2),)
		##self.kernel.requires_grad = False

		self.conv = nn.Conv1d(self.output_num,self.output_num,1+self.window_pad*2,padding=self.window_pad,padding_mode='replicate')

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
			for linear in self.dense:
				x_i = F.relu(linear(x_i))
			y_i = self.output_dense(x_i)
			previous_output = torch.cat((previous_output[:,self.output_num:], F.softmax(y_i.detach(),dim=1)), dim=1)
			y[:,i-self.window_pad,:] = y_i

		#y = F.pad(y.permute(0,2,1), self.padding, mode='replicate') # batch_size * output_num * sequence_length+2
		#y = F.conv1d(y, self.kernel, groups=self.output_num) # batch_size * output_num * sequence_length
		#y = y.permute(0,2,1) # batch_size * sequence_length * output_num

		y = self.conv(y.permute(0,2,1)).permute(0,2,1)

		y = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
		return y

	def make_dataset(self, ann_list):
		return EntityIndexesDataset(ann_list, self.token_indexes, self.fallback_index, self.labels, cuda=next(self.output_dense.parameters()).is_cuda)

	def compute_class_weight(self, class_weight, dataset):
		return compute_class_weight(class_weight, classes=self.labels.classes_, y=[tag for a in dataset.anns for tag in a.labels])

	def make_loss_func(self, criterion, ignore_index=999):
		def loss_func(output, target):
			output,_ = rnn.pad_packed_sequence(output)
			output = output.view(-1, output.shape[-1])
			target,_ = rnn.pad_packed_sequence(target, padding_value=ignore_index)
			target = target.view(-1)
			return criterion(output, target)
		return loss_func

	def make_metric(self, func):
		def metric(output, target):
			output = numpy.vstack([F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(output)]).argmax(1)
			target = numpy.hstack([i.cpu().detach().numpy() for i in unpack_sequence(target)])
			return func(output, target)
		return metric

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(IndexSmoothedWindowMemoryEntityModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
		_state_dict['embedding_dim'] = self.embedding.embedding_dim
		_state_dict['num_embeddings'] = self.embedding.num_embeddings
		_state_dict['window_pad'] = self.window_pad
		_state_dict['hidden'] = tuple(self.hidden)
		_state_dict['output_labels'] = list(self.labels.classes_)
		_state_dict['token_indexes'] = self.token_indexes
		_state_dict['fallback_index'] = self.fallback_index
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = IndexSmoothedWindowMemoryEntityModel(
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

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	from gaml.utilities.torchutils import save_figs # predict_from_dataloader

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction,DirectoryAction,ListAction
	from gaml.utilities.mlutils import split_data
	import os

	from gaml.annotations.wordembeddings import WordEmbeddings
	from gaml.annotations.annmlutils import open_anns

	import sklearn.metrics

	from gaml.annotations.models.training import perform_training,evaluate_entities

	def parse_tuple(s):
		return tuple(int(i) for i in s.split('-'))

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
	parser.add_argument('-w','--class-weight',action='store_const',const='balanced',help='Flag to indicate that the loss function should be class-balanced for training.')
	parser.add_argument('--train-fractions',action='store_true',help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
	parser.add_argument('--window-pad',type=int,default=5,help='Size of window to consider when making token predictions.')
	parser.add_argument('--hidden',type=parse_tuple, default=(1024,512,256),help='Hidden layer widths as a hyphen-separated list, e.g. "1024-512-256".')
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
	embeddings = WordEmbeddings.open(args.embeddings)

	stopwatch.tick('Opened all files',report=True)

	# Make test/train split
	# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics
	ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)

	# Model parameters
	output_labels = list(set(l for a in anns for t,l in a))
	#window_pad = 5

	# Training parameters
	batch_size = 256
	epochs = 300
	patience = 25
	min_delta = 0.001

	# Generating functions
	def make_model():
		return IndexSmoothedWindowMemoryEntityModel(args.window_pad, args.hidden, output_labels, embeddings.indexes, embeddings.fallback_index, embedding=torch.from_numpy(embeddings.values)).float().cuda()
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

	print(model.conv.weight)

	### TEST MODEL
	class_metrics,overall_metrics = evaluate_entities(model, ann_test, batch_size=batch_size)
	class_metrics.to_csv(os.path.join(args.modeldir,'class_metrics.csv'))
	overall_metrics.to_csv(os.path.join(args.modeldir,'overall_metrics.csv'))

	stopwatch.tick('Finished evaluation',report=True)

	save_figs(history, modelname, args.modeldir)

	stopwatch.report()
