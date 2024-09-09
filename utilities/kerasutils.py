import tensorflow as tf

from tensorflow.keras.layers import Layer, InputSpec, Embedding
from tensorflow.python.keras import backend as K
from tensorflow.keras import regularizers
#from tensorflow.keras import initializers
from tensorflow.keras import constraints

import numpy

class TemporalMaxPooling(Layer):
	"""
	This pooling layer accepts the temporal sequence output by a recurrent layer
	and performs temporal pooling, looking at only the non-masked portion of the sequence.
	The pooling layer converts the entire variable-length hidden vector sequence
	into a single hidden vector.
	Modified from https://github.com/fchollet/keras/issues/2151 so code also
	works on tensorflow backend. Updated syntax to match Keras 2.0 spec.
	Args:
		Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
		The dimensions are inferred based on the output shape of the RNN.
		3D tensor with shape: `(samples, steps, features)`.
		input shape: (nb_samples, nb_timesteps, nb_features)
		output shape: (nb_samples, nb_features)
	Examples:
		> x = Bidirectional(GRU(128, return_sequences=True))(x)
		> x = TemporalMaxPooling()(x)
	"""
	# Github: nigeljyng/TemporalMaxPooling.py
	def __init__(self, **kwargs):
		super(TemporalMaxPooling, self).__init__(**kwargs)
		self.supports_masking = True
		self.input_spec = InputSpec(ndim=3)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])

	def call(self, x, mask=None):
		if mask is None:
			mask = K.sum(K.ones_like(x), axis=-1)

		# if masked, set to large negative value so we ignore it when taking max of the sequence
		mask = K.expand_dims(mask, axis=-1)
		mask = K.tile(mask, (1, 1, K.int_shape(x)[2]))
		# if masked assume value is -inf
		masked_data = tf.where(K.equal(mask, K.zeros_like(mask)), K.ones_like(x)*-numpy.inf, x)
		return K.max(masked_data, axis=1)

	def compute_mask(self, input, mask):
		# do not pass the mask to the next layers
		return None


class TemporalMinPooling(Layer):
	"""
	This pooling layer accepts the temporal sequence output by a recurrent layer
	and performs temporal pooling, looking at only the non-masked portion of the sequence.
	The pooling layer converts the entire variable-length hidden vector sequence
	into a single hidden vector.
	Modified from https://github.com/fchollet/keras/issues/2151 so code also
	works on tensorflow backend. Updated syntax to match Keras 2.0 spec.
	Args:
		Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
		The dimensions are inferred based on the output shape of the RNN.
		3D tensor with shape: `(samples, steps, features)`.
		input shape: (nb_samples, nb_timesteps, nb_features)
		output shape: (nb_samples, nb_features)
	Examples:
		> x = Bidirectional(GRU(128, return_sequences=True))(x)
		> x = TemporalMinPooling()(x)
	"""
	# Modified from Github: nigeljyng/TemporalMaxPooling.py
	def __init__(self, **kwargs):
		super(TemporalMinPooling, self).__init__(**kwargs)
		self.supports_masking = True
		self.input_spec = InputSpec(ndim=3)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])

	def call(self, x, mask=None):
		if mask is None:
			mask = K.sum(K.ones_like(x), axis=-1)

		# if masked, set to large positive value so we ignore it when taking min of the sequence
		mask = K.expand_dims(mask, axis=-1)
		mask = K.tile(mask, (1, 1, K.int_shape(x)[2]))
		# if masked assume value is inf
		masked_data = tf.where(K.equal(mask, K.zeros_like(mask)), K.ones_like(x)*numpy.inf, x)
		return K.min(masked_data, axis=1)

	def compute_mask(self, input, mask):
		# do not pass the mask to the next layers
		return None

class MeanPool(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(MeanPool, self).__init__(**kwargs)
		self.input_spec = InputSpec(ndim=3)

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	#def call(self, x, mask=None):
	#	if mask is not None:
	#		# mask (batch, time)
	#		mask = K.cast(mask, K.floatx())
	#		# mask (batch, time, 'x')
	#		mask = mask.dimshuffle(0, 1, 'x')
	#		# to make the masked values in x be equal to zero
	#		x = x * mask
	#	return K.sum(x, axis=1) / K.sum(mask, axis=1)

	def call(self, x, mask=None):
		if mask is not None:
			# mask (batch, time)
			mask = K.cast(mask, K.floatx())
			# mask (batch, x_dim, time)
			mask = K.repeat(mask, x.shape[-1])
			# mask (batch, time, x_dim)
			mask = tf.transpose(mask, [0,2,1])
			x = x * mask
		return K.sum(x, axis=1) / K.sum(mask, axis=1)

	#def get_output_shape_for(self, input_shape):
	def compute_output_shape(self, input_shape):
		# remove temporal dimension
		return input_shape[0], input_shape[2]

class MinMaxMeanPool(Layer):
	def __init__(self,**kwargs):
		self.supports_masking = True
		self.min = TemporalMinPooling(**kwargs)
		self.max = TemporalMaxPooling(**kwargs)
		self.mean = MeanPool(**kwargs)
		super(MinMaxMeanPool, self).__init__(**kwargs)
		self.input_spec = InputSpec(ndim=3)

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		return tf.concat([self.min(x,mask=mask), self.max(x,mask=mask), self.mean(x,mask=mask)], axis=1)

	def compute_output_shape(self, input_shape):
		return input_shape[0], 3*input_shape[2]

class Projection(Layer):
	"""Linear projection layer.

	`Projection` implements the operation:
	`output = dot(input, kernel)`
	where `kernel` is a projection matrix created by the layer.

	# Example

	```python
		# as first layer in a sequential model:
		model = Sequential()
		model.add(Projection(input_shape=(16,128)))
		# now the model will take as input arrays of shape (16,128)
		# and output arrays of shape (16,128)
		# after the first layer, you don't need to specify
		# the size of the input anymore:
		model.add(Projection())
	```

	# Arguments
		kernel_initializer: Initializer for the `kernel` weights matrix
			(see [initializers](../initializers.md)).
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix
			(see [regularizer](../regularizers.md)).
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation").
			(see [regularizer](../regularizers.md)).
		kernel_constraint: Constraint function applied to
			the `kernel` weights matrix
			(see [constraints](../constraints.md)).

	# Input shape
		3D tensor with shape: `(batch_size, timesteps, input_dim)`.

	# Output shape
		3D tensor with shape: `(batch_size, timesteps, input_dim)`.
	"""

	def __init__(self,
				#kernel_initializer='glorot_uniform',
				kernel_regularizer=None,
				activity_regularizer=None,
				kernel_constraint=None,
				**kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (None, kwargs.pop('input_dim')) # None indicates unknown series length

		super(Projection, self).__init__(**kwargs)

		#self.kernel_initializer = initializers.get(kernel_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.input_spec = InputSpec(ndim=3)
		self.supports_masking = True

	def build(self, input_shape):
		assert len(input_shape) == 3
		input_dim = int(input_shape[-1])

		self.kernel = self.add_weight(shape=(input_dim, input_dim),
									#initializer=self.kernel_initializer,
									initializer=initialize_projection,
									name='kernel',
									regularizer=self.kernel_regularizer,
									constraint=self.kernel_constraint)
		#self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
		self.built = True

	def call(self, inputs):
		output = K.dot(inputs, self.kernel)
		return output

	def compute_output_shape(self, input_shape):
		#assert input_shape and len(input_shape) >= 2
		#assert input_shape[-1]
		return input_shape

	def compute_mask(self, input, mask=None):
		return mask

	def get_config(self):
		config = {
			#'kernel_initializer': initializers.serialize(self.kernel_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
		}
		base_config = super(Projection, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def initialize_projection(shape, dtype=None, partition_info=None):
	assert len(shape)==2 and shape[0]==shape[1]
	# init_projection(dims...) = Matrix(1.0*I,dims...)/100 .+ randn(dims...)/1000
	return tf.diag(tf.random_normal((shape[0],),dtype=dtype,stddev=0.01)) + tf.random_normal(shape,dtype=dtype,stddev=0.001)

class PretrainedEmbedding(Embedding):
	def __init__(self, weights, **kwargs):
		vocab_size, dim = weights.shape
		super(PretrainedEmbedding, self).__init__(vocab_size, dim, weights=[weights], trainable=False, **kwargs)

	def get_config(self):
		base_config = super(Projection, self).get_config()
		base_config['weights'] = self.get_weights()[0]
		return base_config
