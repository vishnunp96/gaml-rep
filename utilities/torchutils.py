import os

from collections import defaultdict
import itertools

import numpy
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import matplotlib.pyplot as plt

def unpack_sequence(packed_sequence):
	''' Function to unpack a packed sequence back to a list of tensors. '''
	padded,lengths = rnn.pad_packed_sequence(packed_sequence, batch_first=False)
	unpacked = []
	for i in range(padded.shape[1]):
		x = padded[:,i]
		unpacked.append(x[:lengths[i]])
	return unpacked

def cat_packed_sequences(packed_sequences,enforce_sorted=False):
	return rnn.pack_sequence(list(itertools.chain.from_iterable(unpack_sequence(i) for i in packed_sequences)), enforce_sorted=enforce_sorted)

def make_temporal_mask(padded,lens):
	mask = torch.arange(padded.shape[1]).expand(len(lens), padded.shape[1]) < lens.unsqueeze(1)
	mask = mask.unsqueeze(2).repeat(1,1,padded.shape[2])
	return mask

class TemporalMaxPooling(nn.Module):
	def __init__(self):
		super(TemporalMaxPooling,self).__init__()
	def forward(self,x):
		''' Accepts PackedSequence objects, for rank 3 tensor (batch_size * num_timesteps * num_features) '''
		padded,lens = rnn.pad_packed_sequence(x,batch_first=True)
		mask = make_temporal_mask(padded,lens)
		return torch.max(torch.where(mask, padded, torch.ones_like(padded) * -numpy.inf),1)[0]
class TemporalMinPooling(nn.Module):
	def __init__(self):
		super(TemporalMinPooling,self).__init__()
	def forward(self,x):
		''' Accepts PackedSequence objects, for rank 3 tensor (batch_size * num_timesteps * num_features) '''
		padded,lens = rnn.pad_packed_sequence(x,batch_first=True)
		mask = make_temporal_mask(padded,lens)
		return torch.min(torch.where(mask, padded, torch.ones_like(padded) * numpy.inf),1)[0]
class TemporalMeanPooling(nn.Module):
	def __init__(self):
		super(TemporalMeanPooling,self).__init__()
	def forward(self,x):
		''' Accepts PackedSequence objects, for rank 3 tensor (batch_size * num_timesteps * num_features) '''
		padded,lens = rnn.pad_packed_sequence(x,batch_first=True,padding_value=0)
		mask = make_temporal_mask(padded,lens)
		return torch.sum(x, axis=1) / torch.sum(mask.float(), axis=1)
class TemporalMinMaxMeanPooling(nn.Module):
	def __init__(self):
		super(TemporalMinMaxMeanPooling,self).__init__()
	def forward(self,x):
		'''
			Accepts PackedSequence objects, for rank 3 tensor (batch_size, num_timesteps, num_features)
			Returns rank 2 Tensor (batch_size, 3 * num_features)
		'''
		padded,lens = rnn.pad_packed_sequence(x,batch_first=True,padding_value=0)
		mask = make_temporal_mask(padded,lens).to(x.data.device) # Is this the right place to do .to(device)?
		minx = torch.min(torch.where(mask, padded, torch.ones_like(padded) * numpy.inf),1)[0]
		maxx = torch.max(torch.where(mask, padded, torch.ones_like(padded) * -numpy.inf),1)[0]
		meanx = torch.sum(padded, axis=1) / torch.sum(mask.float(), axis=1)
		return torch.cat((minx,maxx,meanx),1)


class EarlyStopping:

	def __init__(self, patience=0, min_delta=0.0, verbose=False):
		self.patience = patience
		self.min_delta = abs(min_delta)
		self.verbose = verbose

		self.counter = 0
		self.early_stop = False
		self.loss_min = None

		self.checkpoint = None

	def __call__(self, loss, model):

		if self.loss_min is None or (self.loss_min - loss) > self.min_delta:
			if self.verbose > 1 and self.loss_min is not None:
				print(f'Better weights found: {loss:.5f} < {self.loss_min:.5f}')
			self.counter = 0
			self.loss_min = loss
			self.checkpoint = model.state_dict()
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
				if self.verbose > 0: print(f'Patience exceeded, best loss: {self.loss_min:.5f}')

		return self.early_stop

	def reload_model(self,model):
		model.load_state_dict(self.checkpoint, strict=False) ## Is this robust?

def train(model, dataloader, epochs, opt, loss_func, val_dataloader=None, metrics={}, patience=None,min_delta=0.0, verbose=True):
	''' loss_func is callable with signature f(output,target) and returns a tensor. Assumed batch_first on all tensors.  '''

	history = defaultdict(list)

	if patience is not None:
		earlystopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=verbose-1)

	def get_batch_samples(output):
		if isinstance(output, rnn.PackedSequence):
			batch_num = output[0].shape[0]
		else:
			batch_num = output.shape[0]
		return batch_num

	def combine(ts):
		''' Combine along batch dimension (0). '''
		if isinstance(ts[0], rnn.PackedSequence):
			ts = list(itertools.chain.from_iterable(unpack_sequence(i) for i in ts))
			return rnn.pack_sequence([i.detach() for i in ts], enforce_sorted=False)
		else:
			return torch.cat(tuple(i.detach() for i in ts), 0)

	# Calculate loss for 0th epoch (i.e. before training)
	with torch.no_grad():
		history['epoch'].append(0)

		## Iterate over data batches
		model.eval()
		epoch_loss = 0.0
		epoch_samples = 0
		outputs = []
		targets = []
		for batch_i,(batch_x,batch_y) in enumerate(dataloader):
			output = model(batch_x)
			loss = loss_func(output, batch_y)
			batch_num = get_batch_samples(output)
			epoch_loss += batch_num * loss.item()
			epoch_samples += batch_num
			outputs.append(output.cpu())
			targets.append(batch_y.cpu())
		epoch_loss = epoch_loss/epoch_samples
		history['loss'].append(epoch_loss)
		output = combine(outputs)
		target = combine(targets)
		for name,metric in metrics.items():
			history[name].append(metric(output, target))

		if val_dataloader is not None:
			val_loss = 0.0
			val_samples = 0
			outputs = []
			targets = []
			for val_x,val_y in val_dataloader:
				val_output = model(val_x)
				batch_num = get_batch_samples(output)
				val_loss += batch_num * loss_func(val_output, val_y).item()
				val_samples += batch_num
				outputs.append(val_output.cpu())
				targets.append(val_y.cpu())
			val_loss = val_loss/val_samples
			history['val_loss'].append(val_loss)
			output = combine(outputs)
			target = combine(targets)
			for name,metric in metrics.items():
				history['val_'+name].append(metric(output, target))
		model.train()

		if verbose > 0:
			print(f'Epoch 0/{epochs}')
			print('\t'+ ', '.join(f'{name}: {vals[-1]:.4f}' for name,vals in history.items() if name!='epoch'))

	# Training loop
	for epoch in range(epochs):
		history['epoch'].append(epoch+1)
		if verbose > 0: print(f'Epoch {epoch+1}/{epochs}', flush=True)

		## Iterate over data batches
		epoch_loss = 0.0
		epoch_samples = 0
		outputs = []
		targets = []
		for batch_i,(batch_x,batch_y) in enumerate(dataloader):
			opt.zero_grad()
			output = model(batch_x)
			loss = loss_func(output, batch_y)
			loss.backward()
			opt.step()
			batch_num = get_batch_samples(output)
			epoch_loss += batch_num * loss.item()
			epoch_samples += batch_num
			outputs.append(output.cpu())
			targets.append(batch_y.cpu())
		epoch_loss = epoch_loss/epoch_samples
		history['loss'].append(epoch_loss)
		output = combine(outputs)
		target = combine(targets)
		for name,metric in metrics.items():
			history[name].append(metric(output, target))

		if val_dataloader is not None:
			with torch.no_grad():
				model.eval()
				val_loss = 0.0
				val_samples = 0
				outputs = []
				targets = []
				for val_x,val_y in val_dataloader:
					val_output = model(val_x)
					batch_num = get_batch_samples(output)
					val_loss += batch_num * loss_func(val_output, val_y).item()
					val_samples += batch_num
					outputs.append(val_output.cpu())
					targets.append(val_y.cpu())
				val_loss = val_loss/val_samples
				model.train()
			history['val_loss'].append(val_loss)
			output = combine(outputs)
			target = combine(targets)
			for name,metric in metrics.items():
				history['val_'+name].append(metric(output, target))

		#if verbose > 0: print(f'\tLoss: {epoch_loss:.4f}, Validation loss: {val_loss:.4f}')
		if verbose > 0: print('\t'+ ', '.join(f'{name}: {vals[-1]:.4f}' for name,vals in history.items() if name!='epoch'))

		if patience is not None:
			if earlystopping(val_loss, model):
				break

	if patience is not None:
		earlystopping.reload_model(model)

	return dict(history)

def predict_from_dataloader(model, dataloader, activation=None):
	with torch.no_grad(): # There should be no need to call .detach() for efficiency reasons after this
		if model.training:
			model.eval()
			set_training = True
		else:
			set_training = False

		packed_sequence = False

		predictions = []
		for batch_x,batch_y in dataloader:
			output = model(batch_x)
			if isinstance(output,rnn.PackedSequence):
				output = [i if activation is None else activation(i) for i in unpack_sequence(output)]
				predictions.extend(output)
				packed_sequence = True
			else: # isinstance(output,torch.Tensor) == True
				if activation is not None:
					output = activation(output)
				predictions.append(output)

		if set_training:
			model.train()

		if packed_sequence:
			return rnn.pack_sequence(predictions, enforce_sorted=False)
		else:
			return torch.cat(predictions, dim=0)

def save_figs(history, modelname, directory):
	handles = set((k if not k.startswith('val_') else k[4:]) for k in history.keys() if k != 'epoch')
	for h in handles:
		plt.figure()
		if h in history: plt.plot(history[h])
		if ('val_'+h) in history: plt.plot(history['val_'+h])
		plt.title(f'{modelname} {h}')
		plt.ylabel(h)
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig(os.path.join(directory,h+'_curve.png'))
