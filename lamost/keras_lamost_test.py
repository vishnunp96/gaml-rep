#from tensorflow.python import debug as tf_debug

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse #binary_crossentropy
from keras.utils import plot_model
from keras import optimizers
from keras import backend as K

from sklearn import manifold
#from sklearn import decomposition

#import numpy as np
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
	"""Reparameterization trick by sampling from an isotropic unit Gaussian.

	# Arguments
		args (tensor): mean and log of variance of Q(z|X)

	# Returns
		z (tensor): sampled latent vector
	"""

	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	# by default, random_normal has mean = 0 and std = 1.0
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models, data, batch_size=128, dir_name="vae_lamost"):
	"""Plots latent vector space

	# Arguments
		models (tuple): encoder and decoder models
		data (tuple): test data and label
		batch_size (int): prediction batch size
		model_name (string): which model is using this function
	"""

	encoder, decoder = models
	x_test, y_test = data

	z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)

	numpy.save(os.path.join(dir_name,"vae_latent.npy"), z_mean)

	assert numpy.all(numpy.isfinite(z_mean)) # Just checking

	if decoder.input_shape[1]!=2:
		print('Calculating t-SNE representation of latent space.')
		filename = os.path.join(dir_name, "vae_tsne.png")
		#print('Calculating PCA representation of latent space.')
		#filename = os.path.join(dir_name, "vae_pca.png")

		tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
		z_plot = tsne.fit_transform(z_mean)
		#pca = decomposition.PCA(n_components=2)
		#z_plot = pca.fit_transform(z_mean)
	else:
		filename = os.path.join(dir_name, "vae_mean.png")
		z_plot = z_mean

	z_mean = z_plot.mean(axis=0)
	z_std = z_plot.std(axis=0)
	x_lim = z_mean[0]-5*z_std[0], z_mean[0]+5*z_std[0]
	y_lim = z_mean[1]-5*z_std[1], z_mean[1]+5*z_std[1]

	# display a 2D plot of the data in the latent space (using t-SNE)
	plt.figure(figsize=(12, 10))
	plt.scatter(z_plot[:, 0], z_plot[:, 1], c=y_test.reshape(-1))
	plt.colorbar()
	plt.xlabel("z[0]")
	plt.ylabel("z[1]")
	plt.xlim(x_lim)
	plt.ylim(y_lim)
	plt.savefig(filename)
	#plt.show()


if __name__ == '__main__':

	import pandas
	import numpy
	from sklearn.model_selection import train_test_split

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	parser = ArgumentParser()
	parser.add_argument('data',action=FileAction, mustexist=True,help='Location of CSV dataset.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	parser.add_argument('-n','--nrows',type=int,default=None,help='Number of rows of dataset to read in. Defaults to all.')
	args = parser.parse_args()

	#sess = K.get_session()
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	#K.set_session(sess)

	# Read in dataset
	data = pandas.read_csv(args.data,nrows=args.nrows,index_col='filename').values
	x,y = data[:,:-1],data[:,-1] # This should be right...
	#x,y_unscaled = data[:,:-1],data[:,-1].reshape(-1,1) # This should be right...

	good_rows = numpy.all(numpy.isfinite(x),axis=1)

	x = x[good_rows]
	y = y[good_rows]

	numpy.clip(x,-10,10,out=x) # Not ideal - stopgap

	assert not numpy.any(numpy.isnan(x)) # Just checking
	assert numpy.all(numpy.isfinite(x)) # Just checking
	assert numpy.all(numpy.isfinite(y)) # Just checking
	assert not numpy.any(~x.any(axis=1))

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

	original_dim = x_train.shape[1]

	# network parameters
	input_shape = (original_dim, )
	intermediate_dim = 512
	batch_size = 128
	latent_dim = 10
	epochs = 50

	# VAE model = encoder + decoder
	# build encoder model
	inputs = Input(shape=input_shape, name='encoder_input')
	enc_hidden = Dense(intermediate_dim, activation='relu')(inputs)
	z_mean = Dense(latent_dim, name='z_mean')(enc_hidden)
	z_log_var = Dense(latent_dim, name='z_log_var')(enc_hidden)
	#z_log_var = Dense(latent_dim, name='z_log_var',kernel_initializer='zeros',bias_initializer='zeros')(enc_hidden)

	# use reparameterization trick to push the sampling out as input
	# note that "output_shape" isn't necessary with the TensorFlow backend
	z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

	# instantiate encoder model
	encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

	# build decoder model
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	dec_hidden = Dense(intermediate_dim, activation='relu')(latent_inputs)
	outputs = Dense(original_dim, activation='sigmoid')(dec_hidden)

	# instantiate decoder model
	decoder = Model(latent_inputs, outputs, name='decoder')

	# instantiate VAE model
	outputs = decoder(encoder(inputs)[2])
	vae = Model(inputs, outputs, name=args.name)

	models = (encoder, decoder)
	data = (x_test, y_test)

	reconstruction_loss = mse(inputs,outputs) # This seems wrong...
	#reconstruction_loss *= original_dim

	kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

	#reconstruction_loss = K.print_tensor(reconstruction_loss, message='reconstruction_loss = ')
	#kl_loss = K.print_tensor(kl_loss, message='kl_loss = ')

	vae_loss = K.mean(reconstruction_loss + kl_loss)
	vae.add_loss(vae_loss)

	opt = optimizers.Adam(lr=0.0001)
	#opt = optimizers.SGD(lr=0.01)

	vae.compile(optimizer=opt)

	# Make directory to store summaries and results
	os.makedirs(vae.name, exist_ok=True)

	# Print and save model summaries
	encoder.summary()
	plot_model(encoder, to_file=os.path.join(vae.name,vae.name+'_encoder.png'), show_shapes=True)
	decoder.summary()
	plot_model(decoder, to_file=os.path.join(vae.name,vae.name+'_decoder.png'), show_shapes=True)
	vae.summary()
	plot_model(vae, to_file=os.path.join(vae.name,vae.name+'.png'), show_shapes=True)

	# train the autoencoder
	vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))
	#vae.save_weights(os.path.join(vae.name,vae.name+'_weights.h5'))

	plot_results(models, data, batch_size=batch_size, dir_name=vae.name)
