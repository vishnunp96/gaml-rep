import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

#from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from gaml.annotations.datasets import CompositeDataset

from gaml.annotations.models.base import BaseANNEntityModule
from gaml.annotations.models import _class_dict,load_ann_model

# Define model
class StackedEntityModel(BaseANNEntityModule):
	def __init__(self, models):
		super(StackedEntityModel,self).__init__()

		self.models = models

		if len(self.models) < 1:
			raise ValueError('Must provide at least one base model to StackedEntityModel.')
		assert len(set([frozenset(m.labels.classes_) for m in self.models])) == 1

		self.labels = self.models[0].labels
		self.output_num = len(self.labels.classes_)

		# Freeze all stacked models
		for i,m in enumerate(self.models):
			for p in m.parameters():
				p.requires_grad = False
			setattr(self, f'model{i}', m)

		self.input_features = len(models) * self.output_num

		self.output_dense = nn.Linear(self.input_features, self.output_num)

	def forward(self, xs):

		results = [rnn.pad_packed_sequence(m.forward(xm), padding_value=0, batch_first=True) for m,xm in zip(self.models,xs)] # List of m x (batch_size, sequence_length, output_num) and lengths
		lengths = results[0][1] # All lengths should be equal, so just take first
		results = [F.softmax(r[0].detach(),dim=1) for r in results] # Extract only the values, discarding lengths

		x = torch.cat(results, dim=2) # (batch_size, sequence_length, m * output_num)
		x = self.output_dense(x) # (batch_size, sequence_length, output_num)

		x = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
		return x

	def make_dataset(self, ann_list):
		return CompositeDataset(ann_list, (m.make_dataset(ann_list) for m in self.models), cuda=next(self.output_dense.parameters()).is_cuda)

	def compute_class_weight(self, class_weight, dataset):
		return compute_class_weight(class_weight, self.labels.classes_, [tag for a in dataset.anns for tag in a.labels])

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(StackedEntityModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
		_state_dict['output_labels'] = list(self.labels.classes_)
		_state_dict['model_classes'] = [m.__class__.__name__ for m in self.models]
		_state_dict['models'] = [m.state_dict() for m in self.models]
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from stored state_dict with arbitrary shape. '''
		model_classes = _state_dict.pop('model_classes')
		model_states = _state_dict.pop('models')
		model_instances = [_class_dict[c].load_from_state_dict(s) for c,s in zip(model_classes,model_states)]
		model = StackedEntityModel(
				_state_dict.pop('output_labels'),
				model_instances
			)
		model.load_state_dict(_state_dict)
		return model

if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import itertools
	from gaml.utilities.torchutils import save_figs # predict_from_dataloader

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction,DirectoryAction
	from gaml.utilities.mlutils import split_data
	import os

	from gaml.annotations.annmlutils import open_anns
	from gaml.annotations.brattowindow import StandoffLabels

	import sklearn.metrics

	from gaml.annotations.models.training import perform_training,evaluate_entities

	def parse_tuple(s):
		return tuple(int(i) for i in s.split('-'))

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
	parser.add_argument('models',action=FileAction, mustexist=True,help='A file containing the paths of model files to use for this stack.')
	parser.add_argument('-w','--class-weight',action='store_const',const='balanced',help='Flag to indicate that the loss function should be class-balanced for training.')
	parser.add_argument('--train-fractions',action='store_true',help='Flag to indicate that training should be conducted with different fractions of the training dataset.')
	parser.add_argument('--eval',action='store_true',help='Flag to indicate that the model in modeldir should be loaded and evaluated, rather than a new model be trained.')
	parser.add_argument('--seed',type=int,default=42,help='Random number seed for this training run.')
	parser.add_argument('--split',type=parse_tuple,default=(0.8,0.1,0.1),help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
	args = parser.parse_args()

	torch.manual_seed(args.seed)

	modelname = os.path.basename(args.modeldir)

	if args.eval:
		model = StackedEntityModel.load_pretained(os.path.join(args.modeldir,modelname+'.pt'))

		types = StandoffLabels.retrieve_types(model.labels.classes_)
		anns = open_anns(args.ann,types=types,use_labelled=True)
		ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)

		stopwatch.tick('Opened all files',report=True)
	else:

		#model_states = [torch.load(modelpath) for modelpath in args.models]
		#models = [class_dict[s['__model_class']].load_from_state_dict(s) for s in model_states]

		with open(args.models, 'r') as f:
			modelpaths = [i.strip() for i in f.readlines()]

		models = [load_ann_model(path) for path in modelpaths]

		types = StandoffLabels.retrieve_types(list(itertools.chain.from_iterable(m.labels.classes_ for m in models)))
		anns = open_anns(args.ann,types=types,use_labelled=True)
		ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)

		stopwatch.tick('Opened all files',report=True)

		# Make test/train split
		# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics

		# Training parameters
		batch_size = 32
		epochs = 150
		patience = 25
		min_delta = 0.01

		# Generating functions
		def make_model():
			return StackedEntityModel(models).float().cuda()
		def make_opt(model):
			adam_lr = 0.1 # 0.0001
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
