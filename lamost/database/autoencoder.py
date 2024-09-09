# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:08:17 2020

@author: Tom
"""

import time
from datetime import datetime
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
#import matplotlib.ticker as ticker
import numpy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import sqlite3

class MLP(nn.Module):
	def __init__(self, in_features, hidden, bias=True, device=None, dtype=None):
		super(MLP, self).__init__()

		self.hidden = tuple(hidden) # Should be a n-tuple of values indicating the widths of the n layers

		self.dense = []
		for i,h in enumerate(hidden):
			if i==0:
				self.dense.append(nn.Linear(in_features, h))
			else:
				self.dense.append(nn.Linear(hidden[i-1], h))
			setattr(self, f'dense{i}', self.dense[-1])

	def forward(self, x):
		for linear in self.dense:
			x = F.relu(linear(x))
		return x

class EncoderCNN(nn.Module):
	def __init__(self, num_wavelengths, encoding_size, kernel_size):
		super(EncoderCNN, self).__init__()
		self.num_wavelengths = num_wavelengths
		self.encoding_size = encoding_size
		self.kernel_size = kernel_size

		if self.kernel_size % 2 != 1:
			raise ValueError('Kernel size must be odd number.')

		self.padding = int((self.kernel_size - 1) / 2)

		self.conv1 = nn.Conv1d(1, 4, self.kernel_size, stride=1, padding=self.padding, groups=1)
		self.conv2 = nn.Conv1d(4, 16, self.kernel_size, stride=1, padding=self.padding, groups=1) # groups=4

		self.pool_kernel = 4
		self.pool_pad = 0
		self.pooling = torch.nn.MaxPool1d(self.pool_kernel, padding=self.pool_pad)
		self.pooled_size = int(numpy.floor((self.num_wavelengths + 2*self.pool_pad - self.pool_kernel) / self.pool_kernel) + 1)

		self.dense1 = nn.Linear(16*self.pooled_size, 256)
		self.dense2 = nn.Linear(256, 128)

		self.mean = nn.Linear(128, self.encoding_size)
		self.logvar = nn.Linear(128, self.encoding_size)

	def forward(self, x):

		# x: (batch_size, num_wavelengths)
		x = x.view(-1, 1, self.num_wavelengths) # (batch_size, 1, num_wavelengths) = (batch_size, channels, num_wavelengths)

		x = F.relu(self.conv1(x)) # (batch_size, 4, num_wavelengths)
		x = F.relu(self.conv2(x)) # (batch_size, 16, num_wavelengths)

		x = self.pooling(x) # (batch_size, 16, pooled_size)

		x = x.view(-1, 16*self.pooled_size) # (batch_size, 16*pooled_size)

		x = F.relu(self.dense1(x)) # (batch_size, 256)
		x = F.relu(self.dense2(x)) # (batch_size, 128)

		mean = torch.tanh(self.mean(x)) # (batch_size, encoding_size)
		logvar = torch.tanh(self.logvar(x)) # (batch_size, encoding_size)

		return mean, logvar

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(EncoderCNN, self).state_dict(destination=destination, prefix=prefix, keep_vars=False)
		_state_dict['num_wavelengths'] = self.num_wavelengths
		_state_dict['encoding_size'] = self.encoding_size
		_state_dict['kernel_size'] = self.kernel_size
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = EncoderCNN(
				_state_dict.pop('num_wavelengths'),
				_state_dict.pop('encoding_size'),
				_state_dict.pop('kernel_size')
			)
		model.load_state_dict(_state_dict)
		return model

	def save(self, filename):
		''' Save model to specified location. '''
		torch.save(self.state_dict(), filename)

	def load(filename):
		_state_dict = torch.load(filename)
		return EncoderCNN.load_from_state_dict(_state_dict)

class DecoderCNN(nn.Module):
	def __init__(self, num_wavelengths, encoding_size, kernel_size):
		super(DecoderCNN, self).__init__()
		self.num_wavelengths = num_wavelengths
		self.encoding_size = encoding_size
		self.kernel_size = kernel_size

		if self.kernel_size % 2 != 1:
			raise ValueError('Kernel size must be odd number.')

		self.padding = int((self.kernel_size - 1) / 2)

		self.pool_pad = 0
		self.pool_kernel = 4
		self.pooled_size = int(numpy.floor(self.num_wavelengths / self.pool_kernel))
		unpool_size_no_outpad = self.pool_kernel*(self.pooled_size-1) - 2*self.pool_pad + self.pool_kernel
		self.pool_padout = int(abs(self.num_wavelengths - unpool_size_no_outpad))

		self.dense0 = nn.Linear(self.encoding_size, 128)
		self.dense1 = nn.Linear(128, 256)
		self.dense2 = nn.Linear(256, 16*self.pooled_size)

		self.unpooling = nn.ConvTranspose1d(16, 16, self.pool_kernel, stride=self.pool_kernel, padding=self.pool_pad, output_padding=self.pool_padout, groups=1) # groups=16

		self.convT1 = nn.ConvTranspose1d(16, 4, self.kernel_size, stride=1, padding=self.padding, groups=1) # groups=4
		self.convT2 = nn.ConvTranspose1d(4, 1, self.kernel_size, stride=1, padding=self.padding, groups=1)

		self.output = nn.Linear(self.num_wavelengths, self.num_wavelengths)

	def forward(self, x):

		# x: (batch_size, encoding_size)

		x = F.relu(self.dense0(x)) # (batch_size, 128)

		x = F.relu(self.dense1(x)) # (batch_size, 256)
		x = F.relu(self.dense2(x)) # (batch_size, 16*pooled_size)

		x = x.view(-1, 16, self.pooled_size) # (batch_size, 16, pooled_size)

		x = F.relu(self.unpooling(x)) # (batch_size, 16, num_wavelengths)

		x = F.relu(self.convT1(x)) # (batch_size, 4, num_wavelengths)
		x = F.relu(self.convT2(x)) # (batch_size, 1, num_wavelengths)

		x = x.view(-1, self.num_wavelengths) # (batch_size, num_wavelengths)

		x = torch.tanh(self.output(x)) # (batch_size, num_wavelengths)

		return x

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = super(DecoderCNN, self).state_dict(destination=destination, prefix=prefix, keep_vars=False)
		_state_dict['num_wavelengths'] = self.num_wavelengths
		_state_dict['encoding_size'] = self.encoding_size
		_state_dict['kernel_size'] = self.kernel_size
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Load model from state_dict with arbitrary shape. '''
		model = DecoderCNN(
				_state_dict.pop('num_wavelengths'),
				_state_dict.pop('encoding_size'),
				_state_dict.pop('kernel_size')
			)
		model.load_state_dict(_state_dict)
		return model

	def save(self, filename):
		''' Save model to specified location. '''
		torch.save(self.state_dict(), filename)

	def load(filename):
		_state_dict = torch.load(filename)
		return DecoderCNN.load_from_state_dict(_state_dict)


def asMinutes(secs):
	mins = math.floor(secs / 60)
	secs -= mins * 60
	return f'{mins:02.0f}m {secs:02.0f}s'
def timeSince(start, fraction_complete):
	now = time.time()
	elapsed = now - start
	estimated_total = elapsed / fraction_complete
	estimated_remaining = estimated_total - elapsed
	return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {asMinutes(elapsed)} (r. {asMinutes(estimated_remaining)})'

class SQLiteDataset(torch.utils.data.IterableDataset):
	def __init__(self, databasepath, rowcount):
		super(SQLiteDataset).__init__()
		self.databasepath = databasepath
		self.rowcount = rowcount

	def __len__(self):
		return self.rowcount

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()

		if worker_info is None:  # Single-process data loading, return the full iterator
			offset = 0
			limit = self.rowcount
		else: # In a worker process, need to split workload
			per_worker = int(math.ceil(self.rowcount / float(worker_info.num_workers)))
			worker_idx = worker_info.id

			offset = worker_idx * per_worker
			limit = per_worker

		# Can use check_same_thread=False here, as this is a read-only connection
		# (and the same thread requirement is intended for write operations)
		connection = sqlite3.connect('file:'+self.databasepath+'?mode=ro', uri=True, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
		cursor = connection.cursor()

		cursor.execute('SELECT data FROM normalized LIMIT ? OFFSET ?', (limit, offset))
		rows = (torch.tensor(i[0], dtype=torch.float) for i in iter(cursor))

		return rows

	def collate_fn(self, data):
		''' Convert a list of datapoints into a single tensor. '''
		collated = torch.stack(data) # (batch_size, num_wavelengths)
		return collated

	def dataloader(self, batch_size=1, shuffle=False):
		''' Produce a dataloader instance based on this dataset. '''
		return torch.utils.data.DataLoader(
				self,
				collate_fn = self.collate_fn,
				batch_size = batch_size,
				shuffle = False
			)

def train(encoder, decoder, dataset, batch_size=10, epochs=1, print_every=1000, plot_every=100, plotfile=None, learning_rate=1e-3, beta_period=5000, beta_ramp_frac=0.5, beta=1.0, device=None):
	start = time.time()

	stats = ['recon', 'enc_kl']

	# Reset every print_every
	print_totals = {k:0 for k in stats}
	print_every_batch = int(max(1, round(print_every / batch_size)))
	# Reset every plot_every
	plot_totals = {k:0 for k in stats}
	plots = {k:[] for k in (list(plot_totals.keys())+['samples'])}
	plot_every_batch = int(max(1, round(plot_every / batch_size)))
	if plotfile is not None:
		plotfile.write(','.join(sorted(plots.keys()))+'\n')

	# Initialise optimizers
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
	criterion = nn.MSELoss(reduction='mean')

	n_iters = epochs * len(dataset)

	beta = max(0.0, float(beta))
	beta_ramp_frac = min(max(beta_ramp_frac, 0.0), 1.0)

	data = dataset.dataloader(batch_size, True)

	batch_count = 0
	sample_count = 0
	for _ in range(epochs):
		for batch_i,batch_spectra in enumerate(data):

			batch_spectra[~torch.isfinite(batch_spectra)] = 0.0
			batch_spectra = torch.clamp(batch_spectra, min=-2, max=2)

			beta_i = min(1.0, (sample_count % beta_period) / (beta_ramp_frac * beta_period)) * beta

			# spectra: (batch_size, num_wavelengths)
			input_spectra = batch_spectra.to(device)

			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			# Both of these: (batch_size, encoding_size)
			enc_mean, enc_logvar = encoder(input_spectra)

			# KL Divergence loss
			enc_kl_loss = torch.mean(-0.5 * torch.sum(1 + enc_logvar - enc_mean.pow(2) - enc_logvar.exp(), dim=1), dim=0)

			enc_std = torch.exp(0.5 * enc_logvar)
			enc_eps = torch.randn_like(enc_std)
			spectra_encoding = enc_mean + enc_eps * enc_std # (batch_size, encoding_size)

			decoder_output = decoder(spectra_encoding)
			recon_loss = criterion(decoder_output, input_spectra) # Reconstruction loss

			for n,v in {'input': input_spectra, 'output': decoder_output}.items():
				print(n, v.min().item(), v.max().item(), v.mean().item())
			print('recon_loss', recon_loss.item())

			loss = recon_loss + (beta_i * enc_kl_loss)

			loss.backward()

			encoder_optimizer.step()
			decoder_optimizer.step()

			for k,v in {'recon': recon_loss.item(), 'enc_kl': enc_kl_loss.item()}.items():
				print_totals[k] += v
				if plotfile is not None: plot_totals[k] += v

			batch_count += 1
			sample_count += input_spectra.size(0)

			if batch_count % print_every_batch == 0:
				print(f'{timeSince(start, sample_count / n_iters)} ({sample_count:d} {100*(sample_count)/n_iters:.0f}%) ' +
						', '.join([f'{k}: {v/print_every_batch:.4f}' for k,v in print_totals.items()]), flush=True)
				for k in print_totals.keys(): print_totals[k] = 0

			if batch_count % plot_every_batch == 0:
				plots['samples'].append(sample_count)
				for k in plot_totals.keys():
					plots[k].append(plot_totals[k] / plot_every_batch)
					plot_totals[k] = 0
				if plotfile is not None:
					plotfile.write(','.join(str(plots[k][-1]) for k in sorted(plots.keys()))+'\n')
					plotfile.flush()

	return {k:numpy.array(v) for k,v in plots.items()}

def makePlot(data):
	fig, axs = plt.subplots(nrows=len(data)-1, ncols=1, sharex='col')
	for i,(label,values) in enumerate((k,v) for k,v in data.items() if k!='samples'):
		axs[i].plot(data['samples'], values, label=label)
	return fig


if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import os

	from utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from utilities.sqliteutils import register_numpy_array_type, query_db
	from utilities.jsonutils import dump_json

	parser = ArgumentParser(description='Train name autoencoder.')
	parser.add_argument('database', action=FileAction, mustexist=True, help='Database file.')
	parser.add_argument('modeldir', action=DirectoryAction, mustexist=False, mkdirs=True, help='Directory to use when saving outputs.')
	parser.add_argument('--encoding-size', type=int, default=64, help='Size of the name encoding vectors.')
	parser.add_argument('--kernel-size', type=int, default=5, help='CNN kernel size for the encoder/decoder networks.')
	parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for this training run.')
	parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs to conduct.')
	parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training loop.')
	parser.add_argument('--beta', type=float, default=0.01, help='Beta parameter for this training run.')
	parser.add_argument('--beta-period', type=int, default=50000, help='Beta schedule period for this training run.')
	parser.add_argument('--beta-ramp', type=float, default=0.5, help='Beta ramp period for this training run.')
	parser.add_argument('--seed', type=int, default=42, help='Random number seed for this training run.')
	args = parser.parse_args()

	print(f'[{datetime.now().strftime("%H:%M")}]',end=' ')
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print('Using CUDA')
	else:
		device = torch.device("cpu")
		print('Using CPU')

	torch.manual_seed(args.seed)

	register_numpy_array_type()

	def make_file(filename):
		return os.path.join(args.modeldir, filename)

	num_samples = 1000000 # int(query_db(args.database, 'SELECT numrows FROM rowcounts WHERE tablename IS ?', ('normalized,'))['numrows'][0])
	dataset = SQLiteDataset(args.database, num_samples)

	num_wavelengths = int(query_db(args.database, 'SELECT COUNT(1) AS count FROM wavelengths')['count'][0])
	encoding_size = args.encoding_size
	kernel_size = args.kernel_size

	encoder1 = EncoderCNN(num_wavelengths, encoding_size, kernel_size)
	decoder1 = DecoderCNN(num_wavelengths, encoding_size, kernel_size)

	stopwatch.tick('Setup complete',report=True)

	plotfile = open(make_file('loss_epochs.csv'),mode='wt')

	training_params = {'learning_rate': args.learning_rate, 'batch_size': args.batch_size, 'epochs': args.epochs, 'beta_period': args.beta_period, 'beta_ramp_frac': args.beta_ramp, 'beta': args.beta}
	training_data = train(encoder1, decoder1, dataset, print_every=10000, plot_every=1000, plotfile=None, device=None, **training_params)

	plotfile.close()

	stopwatch.tick('Finished training',report=True)

	model_metadata = {'num_wavelengths': num_wavelengths, 'encoding_size': encoding_size, 'kernel_size': kernel_size}
	dump_json(model_metadata, make_file('model_metadata.json'))
	encoder1.cpu().save(make_file('encoder.pt'))
	decoder1.cpu().save(make_file('decoder.pt'))

	stopwatch.tick('Saved models',report=True)

	fig = makePlot(training_data)
	fig.savefig(make_file('loss_plot.png'))

	stopwatch.tick('Finished evaluation',report=True)

	stopwatch.report()
