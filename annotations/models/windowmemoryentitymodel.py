import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder

from gaml.annotations.datasets import EntityIndexesDataset

from gaml.annotations.models.base import BaseANNEntityModule

# Define model
class WindowMemoryEntityModel(BaseANNEntityModule):
	def __init__(self, embedding_dim, window_pad, hidden, output_labels):
		super(WindowMemoryEntityModel,self).__init__()

		self.embedding_dim = embedding_dim
		self.window_pad = window_pad
		self.hidden = hidden
		self.output_labels = list(output_labels)

		self.labels = LabelEncoder().fit(self.output_labels)
		self.output_num = len(self.labels.classes_)

		self.initial_state = nn.Parameter(torch.zeros(self.output_num * self.window_pad))

		input_features = (self.embedding_dim + (self.window_pad*2 + 1)) + self.output_num * self.window_pad
		self.dense1 = nn.Linear(input_features, self.hidden)

		self.output_dense = nn.Linear(self.hidden, self.output_num)

	def forward(self, x):
		''' Accepts PackedSequence, representing batch of data points. '''

		x,lengths = rnn.pad_packed_sequence(x, batch_first=True) # batch_size * sequence_length * num_features

		## Projection layer?

		y = torch.empty(x.shape[0], x.shape[1], self.output_num, device=x.device) # batch_size * sequence_length * output_num
		previous_output = self.initial_state.repeat(x.shape[0], 1).to(device=x.device) # batch_size * (output_num * window_pad)
		for i in range(x.shape[1]): # Over sequence_length dimension
			x_i = torch.cat((x[:,i,:], previous_output), dim=1)
			y_i = F.relu(self.dense1(x_i))
			y_i = self.output_dense(y_i)
			previous_output = torch.cat((previous_output[:,self.output_num:], F.softmax(y_i.detach(),dim=1)), dim=1)
			y[:,i,:] = y_i

		x = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
		return x

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(WindowMemoryEntityModel, self).state_dict(destination=destination,prefix=prefix,keep_vars=keep_vars)
		_state_dict['embedding_dim'] = self.embedding_dim
		_state_dict['window_pad'] = self.window_pad
		_state_dict['hidden'] = self.hidden
		_state_dict['output_labels'] = self.output_labels
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = WindowMemoryEntityModel(
				_state_dict.pop('embedding_dim'),
				_state_dict.pop('window_pad'),
				_state_dict.pop('hidden'),
				_state_dict.pop('output_labels')
			)
		model.load_state_dict(_state_dict)
		return model

class IndexWindowMemoryEntityModel(nn.Module):
	def __init__(self, window_pad, hidden, output_labels, token_indexes, embedding=None, embedding_dim=None, num_embeddings=None):
		super(IndexWindowMemoryEntityModel,self).__init__()

		self.window_pad = window_pad
		self.hidden = hidden
		self.output_labels = list(output_labels)
		self.token_indexes = token_indexes

		self.labels = LabelEncoder().fit(self.output_labels)
		self.output_num = len(self.labels.classes_)

		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
		else:
			assert embedding_dim is not None and num_embeddings is not None
			self.embedding = nn.Embedding(num_embeddings, embedding_dim)

		self.initial_state = nn.Parameter(torch.zeros(self.output_num * self.window_pad))

		input_features = (self.embedding_dim + (self.window_pad*2 + 1)) + self.output_num * self.window_pad
		self.dense1 = nn.Linear(input_features, self.hidden)

		self.output_dense = nn.Linear(self.hidden, self.output_num)

	def forward(self, idxs):
		''' Accepts PackedSequence, representing batch of token index sequences. '''

		idxs,lengths = rnn.pad_packed_sequence(idxs, padding_value=0, batch_first=True) # batch_size * sequence_length
		idxs_pad = F.pad(idxs,(self.window_pad,self.window_pad),value=0)
		x = self.embedding(idxs_pad) # batch_size * sequence_length * embedding_dim

		y = torch.empty(idxs.shape[0], idxs.shape[1], self.output_num, device=x.device) # batch_size * sequence_length * output_num
		previous_output = self.initial_state.repeat(idxs.shape[0], 1).to(device=x.device) # batch_size * (output_num * window_pad)
		for i in range(self.window_pad, x.shape[1] - self.window_pad): # Over sequence_length dimension (ignoring padding)
			e_i = x[:,(i-self.window_pad):(i+self.window_pad+1),:] # batch_size * (window_pad*2+1) * embedding_dim
			e_i = e_i.reshape(idxs.shape[0], -1) # batch_size * ((window_pad*2+1)*embedding_dim)
			x_i = torch.cat((e_i, previous_output), dim=1)
			y_i = F.relu(self.dense1(x_i))
			y_i = self.output_dense(y_i)
			previous_output = torch.cat((previous_output[:,self.output_num:], F.softmax(y_i.detach(),dim=1)), dim=1)
			y[:,i,:] = y_i

		x = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
		return x

	def make_dataset(self, ann_list):
		return EntityIndexesDataset(ann_list, self.token_indexes, self.labels)

	def state_dict(self):
		_state_dict = super(IndexWindowMemoryEntityModel, self).state_dict()
		_state_dict['embedding_dim'] = self.embedding.embedding_dim
		_state_dict['num_embeddings'] = self.embedding.num_embeddings
		_state_dict['window_pad'] = self.window_pad
		_state_dict['hidden'] = self.hidden
		_state_dict['output_labels'] = self.output_labels
		_state_dict['token_indexes'] = self.token_indexes
		return _state_dict

	def save(self, filename):
		''' Save model to specified location. '''
		torch.save(self.state_dict(), filename)

	def load_pretrained(filename):
		''' Load model from stored state_dict with arbitrary shape. '''
		_state_dict = torch.load(filename)
		model = WindowMemoryEntityModel(
				_state_dict.pop('window_pad'),
				_state_dict.pop('hidden'),
				_state_dict.pop('output_labels'),
				_state_dict.pop('token_indexes'),
				embedding_dim=_state_dict.pop('embedding_dim'),
				num_embeddings=_state_dict.pop('num_embeddings')
			)
		model.load_state_dict(_state_dict)
		return model

if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	from gaml.utilities.torchutils import unpack_sequence,train,save_figs # predict_from_dataloader

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas
	from sklearn.model_selection import train_test_split

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction,DirectoryAction
	import os

	from gaml.annotations.wordembeddings import WordEmbeddings
	from gaml.annotations.annmlutils import open_anns
	from gaml.annotations.datasets import EntityEmbeddingsDataset

	import sklearn.metrics
	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
	#from sklearn.utils.class_weight import compute_class_weight

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
	args = parser.parse_args()

	torch.manual_seed(42)

	modelname = os.path.basename(args.modeldir)

	# Read in data
	types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName']
	#types = None
	anns = open_anns(args.ann,types=types,use_labelled=True)
	embeddings = WordEmbeddings.open(args.embeddings)

	stopwatch.tick('Opened all files',report=True)

	# Make test/train split
	# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics
	ann_train,ann_other = train_test_split(anns, test_size=0.2, random_state=42)
	ann_dev,ann_test = train_test_split(ann_other, test_size=0.5, random_state=42)

	# Model parameters
	output_labels = list(set(l for a in anns for t,l in a))
	window_pad = 5

	# Create model
	model = WindowMemoryEntityModel(embeddings.dim, window_pad, 1024, output_labels).float().cuda()
	print(model)

	train_dataset = EntityEmbeddingsDataset(ann_train, embeddings, model.labels, window_pad)
	dev_dataset = EntityEmbeddingsDataset(ann_dev, embeddings, model.labels, window_pad)
	test_dataset = EntityEmbeddingsDataset(ann_test, embeddings, model.labels, window_pad)

	stopwatch.tick(f'Created model and constructed datasets: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} testing samples',report=True)

	opt = optim.Adam(model.parameters(), lr=0.001)
	ignore_index = 999
	criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
	#class_weights = compute_class_weight('balanced', labels.classes_, [tag for a in ann_train for tag in a.labels])
	#criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=torch.tensor(class_weights).float().cuda())

	def loss_func(output, target):
		output,_ = rnn.pad_packed_sequence(output)
		output = output.view(-1, output.shape[-1])
		target,_ = rnn.pad_packed_sequence(target, padding_value=ignore_index)
		target = target.view(-1)
		return criterion(output, target)

	def make_metric(func):
		def metric(output, target):
			output = numpy.vstack([F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(output)]).argmax(1)
			target = numpy.hstack([i.cpu().detach().numpy() for i in unpack_sequence(target)])
			return func(output, target)
		return metric
	average = 'macro'
	f1_score = make_metric(lambda o,t: sklearn.metrics.f1_score(t,o,average=average,zero_division=0))
	precision_score = make_metric(lambda o,t: sklearn.metrics.precision_score(t,o,average=average,zero_division=0))
	recall_score = make_metric(lambda o,t: sklearn.metrics.recall_score(t,o,average=average,zero_division=0))

	# Training parameters
	batch_size = 128
	epochs = 100

	# Setup training
	dataloader = train_dataset.dataloader(batch_size=batch_size,shuffle=True)
	dev_dataloader = dev_dataset.dataloader(batch_size=batch_size)

	stopwatch.tick('Setup training',report=True)

	history = train(model, dataloader, epochs, opt, loss_func, dev_dataloader, metrics={'f1':f1_score,'precision':precision_score,'recall':recall_score}, patience=25, min_delta=0.001, verbose=1)

	model.save(os.path.join(args.modeldir,modelname+'.pt'))
	pandas.DataFrame(history).set_index('epoch').to_csv(os.path.join(args.modeldir,'logs.csv'))

	stopwatch.tick('Completed training',report=True)

	### TEST MODEL
	model.eval()

	test_subjects = ann_test[10:12]
	subjects_dataset = EntityEmbeddingsDataset(test_subjects, embeddings, model.labels, window_pad)

	subject_x, subject_y = subjects_dataset.get_data()
	predict_subjects = model(subject_x.cuda())
	subject_preds = [F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_subjects)]

	print_len = max(len(c) for c in model.labels.classes_) + 2
	for subject,preds in zip(test_subjects,subject_preds):
		print(f'Lengths: {subject.labels.shape}, {preds.shape}')
		sub_labels = model.labels.inverse_transform(preds.argmax(1))
		for (token,label),pred,pred_lab in zip(subject,preds,sub_labels):
			vec_str = ', '.join(f'{p:.2f}' for p in pred)
			print(('{0:20} {1:'+str(print_len)+'} {2:'+str(print_len)+'} [{3}]').format(token,label,pred_lab,vec_str))
	print(f'[{", ".join(str(i) for i in model.labels.classes_)}]')

	test_x, test_y = test_dataset.get_data()
	predict_test = model(test_x.cuda())
	test_preds = numpy.vstack([F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_test)])
	print(test_preds.shape)
	predicted_test_labels = model.labels.inverse_transform(test_preds.argmax(1))

	val_target = numpy.hstack([i.cpu().detach().numpy() for i in unpack_sequence(test_y)])
	print(val_target.shape)
	true_test_labels = model.labels.inverse_transform(val_target)

	confusion = confusion_matrix(true_test_labels, predicted_test_labels, labels=model.labels.classes_)
	print(confusion)
	print(confusion.shape)
	p,r,f,s = precision_recall_fscore_support(true_test_labels, predicted_test_labels, labels=model.labels.classes_)

	print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
	for i,label in enumerate(model.labels.classes_):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

	pandas.DataFrame([p,r,f,s],columns=model.labels.classes_,index=['Precision','Recall','F1','Support']).T.to_csv(os.path.join(args.modeldir,'test_metrics.csv'))

	stopwatch.tick('Finished evaluation',report=True)

	save_figs(history, args.modeldir)

	stopwatch.report()
