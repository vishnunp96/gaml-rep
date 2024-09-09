import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

import numpy

from contextlib import contextmanager
from copy import deepcopy

from gaml.utilities.torchutils import unpack_sequence
from gaml.annotations.brattowindow import StandoffLabels
from gaml.annotations.models.predictionutils import token_labels_to_entities
from gaml.utilities.torchutils import predict_from_dataloader

# Define model base class
class BaseANNModule(nn.Module):

	def make_dataset(self, ann_list):
		raise NotImplementedError(f'{self.__class__.__name__} has no \'make_dataset\' method implementation.')

	def compute_class_weight(self, class_weight, dataset):
		raise NotImplementedError(f'{self.__class__.__name__} has no \'compute_class_weight\' method implementation.')

	def make_loss_func(self, criterion, ignore_index=999):
		''' The most basic make_loss_func method (just returns the unmodified criterion). '''
		return criterion

	def make_metric(self, func):
		''' A basic version of a make_metric method (simply converts output and target to numpy arrays). '''
		def metric(output, target):
			output = output.cpu().detach().numpy()
			target = target.cpu().detach().numpy()
			return func(output, target)
		return metric

	def predict(self, anns, batch_size=1, inplace=True):
		raise NotImplementedError(f'{self.__class__.__name__} has no \'predict\' method implementation.')

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(BaseANNModule, self).state_dict(destination=destination,prefix=prefix,keep_vars=False)
		_state_dict['__model_class'] = self.__class__.__name__
		return _state_dict

	def load_state_dict(self, state_dict, strict=True):
		state_dict.pop('__model_class', None) # None prevents KeyError being raised if __model_class is missing.
		return super(BaseANNModule, self).load_state_dict(state_dict, strict=strict)

	def save(self, filename):
		''' Save model to specified location. '''
		torch.save(self.state_dict(), filename)

	@classmethod
	def load_pretrained(cls, filename, **load_args):
		''' Load a pretrained model from a file. '''
		_state_dict = torch.load(filename, **load_args)
		return cls.load_from_state_dict(_state_dict)

	def load_from_state_dict(_state_dict):
		''' Create an instance of the model based on a provided state_dict of arbitrary shape. '''
		raise NotImplementedError

	@contextmanager
	def evaluation(self):
		with torch.no_grad():
			if self.training:
				self.eval()
				set_training = True
			else:
				set_training = False
			yield self
			if set_training:
				self.train()

# Base class for Entity prediction models
class BaseANNEntityModule(BaseANNModule):
	''' A Module which produces a PackedSequence of entity predictions when called (shape: (batch_size,[sequence_length],output_um)). '''

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

	def predict(self, anns, batch_size=1, inplace=True):
		if not inplace:
			anns = deepcopy(anns) # [(p,i,Standoff(a.text,[],[],[],[])) for p,i,a in anns]
		entity_dataset = self.make_dataset([StandoffLabels(a) for p,i,a in anns])

		predictions = []
		for batch_x,_ in entity_dataset.dataloader(batch_size=batch_size, shuffle=False):
			output = self(batch_x)
			outputs = [F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(output)]
			predictions.extend(outputs)

		# Load predictions into Standoff objects
		for ann,pred in zip(entity_dataset.anns, predictions):
			for ent_type,start,end in token_labels_to_entities(self.labels.inverse_transform(pred.argmax(1)), ann.token_idxs):
				ann.standoff.entity(ent_type,start,end)

		return anns

# Base class for Relation prediction models
class BaseANNRelationModule(BaseANNModule):
	''' A Module which produces a tensor of relation predictions when called (shape: (relation_count,output_num)). '''

	def make_metric(self, func):
		def metric(output, target):
			output = output.cpu().detach().numpy().argmax(1)
			target = target.cpu().detach().numpy()
			return func(output, target)
		return metric

	def predict(self, anns, batch_size=1, inplace=True):
		if not inplace:
			anns = deepcopy(anns)
		relation_dataset = self.make_dataset([StandoffLabels(a) for p,i,a in anns])
		relations = predict_from_dataloader(
				self,
				relation_dataset.dataloader(batch_size=batch_size, shuffle=False),
				activation=lambda i: F.softmax(i,dim=1).cpu().detach()
			)
		relations = self.labels.inverse_transform(relations.argmax(1).numpy())

		for pred_label,(ann,start,end) in zip(relations, relation_dataset.origins):
			if pred_label != 'none':
				ann.standoff.relation(pred_label, start, end)

		return anns

