import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy

from gaml.annotations.brattowindow import StandoffLabels
from gaml.annotations.datasets import AttributeSplitSpansIndexesEncodingsDataset
from gaml.utilities.torchutils import predict_from_dataloader

from gaml.annotations.models.base import BaseANNModule

# Define model
class WindowAttributeModel(BaseANNModule):
	def __init__(self, window_pad, hidden, output_labels, entity_labels, allowed_subject_entities, token_indexes, fallback_index, embedding=None, embedding_dim=None, num_embeddings=None):
		super(WindowAttributeModel,self).__init__()

		self.window_pad = window_pad
		self.hidden = tuple(hidden)
		self.token_indexes = token_indexes
		self.fallback_index = fallback_index

		self.labels = LabelEncoder().fit(output_labels)
		self.output_num = len(self.labels.classes_)

		self.entity_labels = LabelBinarizer().fit(entity_labels)
		self.entity_classes_num = len(self.entity_labels.classes_)

		self.allowed_subject_entities = allowed_subject_entities

		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
		else:
			assert embedding_dim is not None and num_embeddings is not None
			self.embedding = nn.Embedding(num_embeddings, embedding_dim)

		self.projection = nn.Parameter(0.01*torch.diag(torch.randn(self.embedding.embedding_dim)) + 0.001*torch.randn(self.embedding.embedding_dim, self.embedding.embedding_dim))

		self.pool = nn.LSTM(self.embedding.embedding_dim + self.entity_classes_num, 64, bidirectional=True, batch_first=True)

		input_features = 64*2*3
		#if self.entity_classes_num > 1:
		#	input_features += self.entity_classes_num
		self.dense = []
		for i,h in enumerate(hidden):
			if i==0:
				self.dense.append(nn.Linear(input_features,h))
			else:
				self.dense.append(nn.Linear(hidden[i-1],h))
			setattr(self, f'dense{i}', self.dense[-1])

		self.output_dense = nn.Linear(self.hidden[-1], self.output_num)


	def forward(self, x):

		ts,ls = x

		def process_span(tokens,labels):
			idxs,lengths = rnn.pad_packed_sequence(tokens, padding_value=0, batch_first=True) # (batch_size, sequence_length)
			labels,_ = rnn.pad_packed_sequence(labels, padding_value=0, batch_first=True) # (batch_size, sequence_length, label_encoding)
			emb = self.embedding(idxs) # (batch_size, sequence_length, embedding_dim)
			emb = torch.matmul(emb, self.projection) # Perform projection # (batch_size, sequence_length, embedding_dim)
			span = torch.cat((emb,labels), 2) # Concatenate entity labels # (batch_size, sequence_length, embeddding_dim + label_encoding)
			span = rnn.pack_padded_sequence(span, lengths, batch_first=True, enforce_sorted=False)
			span = F.relu(self.pool(span)[1][0].permute(1,2,0).reshape(-1, 64*2)) # (batch_size, 64*2)
			return span

		x = tuple(process_span(t,l) for t,l in zip(ts,ls)) # 3 * (batch_size, 64*2 + label_encoding)

		x = torch.cat(x,1) # (batch_size, 3*(64*2 + label_encoding))

		for linear in self.dense:
			x = F.relu(linear(x))

		x = self.output_dense(x) # (batch_size, output_num)
		return x

	def make_dataset(self, ann_list):
		return AttributeSplitSpansIndexesEncodingsDataset(ann_list, self.token_indexes, self.fallback_index, self.labels, self.entity_labels, self.allowed_subject_entities, self.window_pad, cuda=next(self.output_dense.parameters()).is_cuda)

	def compute_class_weight(self, class_weight, dataset):
		return compute_class_weight(class_weight, classes=self.labels.classes_, y=dataset.y_labels)

	def make_loss_func(self, criterion, ignore_index=999):
		return criterion

	def make_metric(self, func):
		def metric(output, target):
			output = output.cpu().detach().numpy().argmax(1)
			target = target.cpu().detach().numpy()
			return func(output, target)
		return metric

	def predict(self, anns, batch_size=1, inplace=True):
		if not inplace:
			anns = deepcopy(anns)

		attribute_dataset = self.make_dataset([StandoffLabels(a) for p,i,a in anns])

		if len(attribute_dataset)>0:
			attributes = predict_from_dataloader(
					self,
					attribute_dataset.dataloader(batch_size=batch_size, shuffle=False),
					activation=lambda i: F.softmax(i,dim=1)
				)
			attributes = self.labels.inverse_transform(attributes.cpu().detach().numpy().argmax(1))

			for pred_label,(ann,subject) in zip(attributes, attribute_dataset.origins):
				if pred_label != 'none':
					ann.standoff.attribute(pred_label, subject)

		return anns

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(WindowAttributeModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
		_state_dict['embedding_dim'] = self.embedding.embedding_dim
		_state_dict['num_embeddings'] = self.embedding.num_embeddings
		_state_dict['window_pad'] = self.window_pad
		_state_dict['hidden'] = tuple(self.hidden)
		_state_dict['output_labels'] = list(self.labels.classes_)
		_state_dict['entity_labels'] = list(self.entity_labels.classes_)
		_state_dict['allowed_subject_entities'] = self.allowed_subject_entities
		_state_dict['token_indexes'] = self.token_indexes
		_state_dict['fallback_index'] = self.fallback_index
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = WindowAttributeModel(
				_state_dict.pop('window_pad'),
				_state_dict.pop('hidden'),
				_state_dict.pop('output_labels'),
				_state_dict.pop('entity_labels'),
				_state_dict.pop('allowed_subject_entities'),
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

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction,DirectoryAction,ListAction
	from gaml.utilities.mlutils import split_data
	import os

	from gaml.annotations.wordembeddings import WordEmbeddings
	from gaml.annotations.annmlutils import open_anns

	import sklearn.metrics

	from gaml.annotations.models.training import perform_training,evaluate_attributes

	def parse_tuple(s):
		return tuple(int(i) for i in s.split('-'))

	parser = ArgumentParser(description='Train Keras ANN to predict relations in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
	parser.add_argument('-w','--class-weight',action='store_const',const='balanced',help='Flag to indicate that the loss function should be class-balanced for training. Should be used for this model.')
	parser.add_argument('--train-fractions',action='store_true',help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
	parser.add_argument('--window-pad',type=int,default=5,help='Size of window to consider when making attribute predictions.')
	parser.add_argument('--hidden',type=parse_tuple,default=(1024,512,256),help='Hidden layer widths as a hyphen-separated list, e.g. "1024-512-256".')
	parser.add_argument('--seed', type=int, default=42, help='Random number seed for this training run.')
	parser.add_argument('--split', type=parse_tuple, default=(0.6,0.2,0.2), help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
	parser.add_argument('--types',action=ListAction, help='Annotation types to consider.')
	args = parser.parse_args()

	torch.manual_seed(args.seed)

	modelname = os.path.basename(args.modeldir)

	# Read in data
	if not args.types:
		args.types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Confidence','Measurement','Name','Property','UpperBound','LowerBound']
	anns = open_anns(args.ann,types=args.types,use_labelled=True)
	embeddings = WordEmbeddings.open(args.embeddings)

	stopwatch.tick('Opened all files',report=True)

	# Make test/train split
	# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics
	ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)

	# Model parameters
	#window_pad = 5
	output_labels = ['UpperBound','LowerBound','none']
	subject_labels = ['MeasuredValue']
	entity_labels = StandoffLabels.get_labels(['MeasuredValue','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName'])

	# Training parameters
	batch_size = 256
	epochs = 200

	# Generating functions
	def make_model():
		return WindowAttributeModel(args.window_pad, args.hidden, output_labels, entity_labels, subject_labels, embeddings.indexes, embeddings.fallback_index, embedding=torch.from_numpy(embeddings.values)).float().cuda()
	def make_opt(model):
		return optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
	def make_loss_func(model, train_dataset):
		class_weights = model.compute_class_weight(args.class_weight, train_dataset)
		criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().cuda() if args.class_weight else None)
		return model.make_loss_func(criterion)
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
			patience=25, min_delta=0.0001,
			modelname=modelname,
			train_fractions=args.train_fractions,
			stopwatch=stopwatch)

	### TEST MODEL
	class_metrics,overall_metrics = evaluate_attributes(model, ann_test, batch_size=batch_size)
	class_metrics.to_csv(os.path.join(args.modeldir,'class_metrics.csv'))
	overall_metrics.to_csv(os.path.join(args.modeldir,'overall_metrics.csv'))

	stopwatch.tick('Finished evaluation',report=True)

	save_figs(history, modelname, args.modeldir)

	stopwatch.report()
