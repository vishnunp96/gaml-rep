from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import constraints

from tensorflow.keras.layers import Layer,InputSpec


from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

#from tensorflow import boolean_mask


class MyLayer(Layer):

	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MyLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = tensor_shape.TensorShape((input_shape[1], self.output_dim))
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='kernel', shape=shape, initializer='uniform', trainable=True)
		# Make sure to call the `build` method at the end
		super(MyLayer, self).build(input_shape)

	def call(self, inputs):
		return K.dot(inputs, self.kernel)

	def compute_output_shape(self, input_shape):
		shape = tensor_shape.TensorShape(input_shape).as_list()
		shape[-1] = self.output_dim
		return tensor_shape.TensorShape(shape)

	def get_config(self):
		base_config = super(MyLayer, self).get_config()
		base_config['output_dim'] = self.output_dim
		return base_config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

class MyProjection(Layer):
	"""Just your regular densely-connected NN layer.
	`Dense` implements the operation:
	`output = activation(dot(input, kernel) + bias)`
	where `activation` is the element-wise activation function
	passed as the `activation` argument, `kernel` is a weights matrix
	created by the layer, and `bias` is a bias vector created by the layer
	(only applicable if `use_bias` is `True`).
	Note: If the input to the layer has a rank greater than 2, then
	it is flattened prior to the initial dot product with `kernel`.
	Example:
	```python
	# as first layer in a sequential model:
	model = Sequential()
	model.add(Dense(32, input_shape=(16,)))
	# now the model will take as input arrays of shape (*, 16)
	# and output arrays of shape (*, 32)
	# after the first layer, you don't need to specify
	# the size of the input anymore:
	model.add(Dense(32))
	```
	Arguments:
		units: Positive integer, dimensionality of the output space.
		activation: Activation function to use.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix.
		bias_initializer: Initializer for the bias vector.
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix.
		bias_regularizer: Regularizer function applied to the bias vector.
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation")..
		kernel_constraint: Constraint function applied to
			the `kernel` weights matrix.
		bias_constraint: Constraint function applied to the bias vector.
	Input shape:
		N-D tensor with shape: `(batch_size, ..., input_dim)`.
		The most common situation would be
		a 2D input with shape `(batch_size, input_dim)`.
	Output shape:
		N-D tensor with shape: `(batch_size, ..., units)`.
		For instance, for a 2D input with shape `(batch_size, input_dim)`,
		the output would have shape `(batch_size, units)`.
	"""

	def __init__(self,
			activation=None,
			use_bias=True,
			kernel_initializer='glorot_uniform',
			bias_initializer='zeros',
			kernel_regularizer=None,
			bias_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			**kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)

		super(MyProjection, self).__init__(activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
		self.activation = activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)

		self.supports_masking = True
		self.input_spec = InputSpec(min_ndim=2)

	def build(self, input_shape):
		dtype = dtypes.as_dtype(self.dtype or K.floatx())
		if not (dtype.is_floating or dtype.is_complex):
			raise TypeError('Unable to build `Projection` layer with non-floating point dtype %s' % (dtype,))
		input_shape = tensor_shape.TensorShape(input_shape)
		if tensor_shape.dimension_value(input_shape[-1]) is None:
			raise ValueError('The last dimension of the inputs to `Dense` '
						'should be defined. Found `None`.')
		last_dim = tensor_shape.dimension_value(input_shape[-1])
		self.input_spec = InputSpec(min_ndim=2,
									axes={-1: last_dim})
		self.kernel = self.add_weight(
			'kernel',
			shape=[last_dim, self.units],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
			dtype=self.dtype,
			trainable=True)
		if self.use_bias:
			self.bias = self.add_weight(
				'bias',
				shape=[self.units,],
				initializer=self.bias_initializer,
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint,
				dtype=self.dtype,
				trainable=True)
		else:
			self.bias = None
		self.built = True

	def call(self, inputs):
		inputs = ops.convert_to_tensor(inputs)
		rank = common_shapes.rank(inputs)
		if rank > 2:
			# Broadcasting is required for the inputs.
			outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
			# Reshape the output back to the original ndim of the input.
			if not context.executing_eagerly():
				shape = inputs.shape.as_list()
				output_shape = shape[:-1] + [self.units]
				outputs.set_shape(output_shape)
		else:
			# Cast the inputs to self.dtype, which is the variable dtype. We do not
			# cast if `should_cast_variables` is True, as in that case the variable
			# will be automatically casted to inputs.dtype.
			if not self._mixed_precision_policy.should_cast_variables:
				inputs = math_ops.cast(inputs, self.dtype)
			outputs = gen_math_ops.mat_mul(inputs, self.kernel)
		if self.use_bias:
			outputs = nn.bias_add(outputs, self.bias)
		if self.activation is not None:
			return self.activation(outputs)  # pylint: disable=not-callable
		return outputs

	def compute_output_shape(self, input_shape):
		input_shape = tensor_shape.TensorShape(input_shape)
		input_shape = input_shape.with_rank_at_least(2)
		if tensor_shape.dimension_value(input_shape[-1]) is None:
			raise ValueError(
				'The innermost dimension of input_shape must be defined, but saw: %s'
				% input_shape)
		return input_shape[:-1].concatenate(self.units)

	def get_config(self):
		config = {
				'units': self.units,
				'activation': activations.serialize(self.activation),
				'use_bias': self.use_bias,
				'kernel_initializer': initializers.serialize(self.kernel_initializer),
				'bias_initializer': initializers.serialize(self.bias_initializer),
				'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
				'bias_regularizer': regularizers.serialize(self.bias_regularizer),
				'activity_regularizer': regularizers.serialize(self.activity_regularizer),
				'kernel_constraint': constraints.serialize(self.kernel_constraint),
				'bias_constraint': constraints.serialize(self.bias_constraint)
			}
		base_config = super(MyProjection, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class Projection(Layer):
	"""Just your regular densely-connected NN layer.

	`Dense` implements the operation:
	`output = activation(dot(input, kernel) + bias)`
	where `activation` is the element-wise activation function
	passed as the `activation` argument, `kernel` is a weights matrix
	created by the layer, and `bias` is a bias vector created by the layer
	(only applicable if `use_bias` is `True`).

	Note: if the input to the layer has a rank greater than 2, then
	it is flattened prior to the initial dot product with `kernel`.

	# Example

	```python
		# as first layer in a sequential model:
		model = Sequential()
		model.add(Dense(32, input_shape=(16,)))
		# now the model will take as input arrays of shape (*, 16)
		# and output arrays of shape (*, 32)
		# after the first layer, you don't need to specify
		# the size of the input anymore:
		model.add(Dense(32))
	```

	# Arguments
		units: Positive integer, dimensionality of the output space.
		activation: Activation function to use
			(see [activations](../activations.md)).
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix
			(see [initializers](../initializers.md)).
		bias_initializer: Initializer for the bias vector
			(see [initializers](../initializers.md)).
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix
			(see [regularizer](../regularizers.md)).
		bias_regularizer: Regularizer function applied to the bias vector
			(see [regularizer](../regularizers.md)).
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation").
			(see [regularizer](../regularizers.md)).
		kernel_constraint: Constraint function applied to
			the `kernel` weights matrix
			(see [constraints](../constraints.md)).
		bias_constraint: Constraint function applied to the bias vector
			(see [constraints](../constraints.md)).

	# Input shape
		nD tensor with shape: `(batch_size, ..., input_dim)`.
		The most common situation would be
		a 2D input with shape `(batch_size, input_dim)`.

	# Output shape
		nD tensor with shape: `(batch_size, ..., units)`.
		For instance, for a 2D input with shape `(batch_size, input_dim)`,
		the output would have shape `(batch_size, units)`.
	"""

	def __init__(self,
				activation=None,
				use_bias=True,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				kernel_regularizer=None,
				bias_regularizer=None,
				activity_regularizer=None,
				kernel_constraint=None,
				bias_constraint=None,
				**kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)

		super(Projection, self).__init__(**kwargs)

		self.activation = activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(min_ndim=2)
		self.supports_masking = True

	def build(self, input_shape):
		assert len(input_shape) >= 2
		input_dim = input_shape[-1]

		self.kernel = self.add_weight(shape=(input_dim, input_dim),
									initializer=self.kernel_initializer,
									name='kernel',
									regularizer=self.kernel_regularizer,
									constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(input_dim,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None
		self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
		self.built = True

	def call(self, inputs):
		output = K.dot(inputs, self.kernel)
		if self.use_bias:
			output = K.bias_add(output, self.bias, data_format='channels_last')
		if self.activation is not None:
			output = self.activation(output)
		return output

	def compute_output_shape(self, input_shape):
		assert input_shape and len(input_shape) >= 2
		assert input_shape[-1]
		output_shape = list(input_shape)
		#output_shape[-1] = self.units # Input shape is unchanged - unless we alter so that output dim is changed? But why would we do that...?
		return tuple(output_shape)

	def get_config(self):
		config = {
			'activation': activations.serialize(self.activation),
			'use_bias': self.use_bias,
			'kernel_initializer': initializers.serialize(self.kernel_initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'activity_regularizer':
				regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint)
		}
		base_config = super(Projection, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



class MinMaxMean(Layer):
	"""Just your regular densely-connected NN layer.

	`Dense` implements the operation:
	`output = activation(dot(input, kernel) + bias)`
	where `activation` is the element-wise activation function
	passed as the `activation` argument, `kernel` is a weights matrix
	created by the layer, and `bias` is a bias vector created by the layer
	(only applicable if `use_bias` is `True`).

	Note: if the input to the layer has a rank greater than 2, then
	it is flattened prior to the initial dot product with `kernel`.

	# Example

	```python
		# as first layer in a sequential model:
		model = Sequential()
		model.add(Dense(32, input_shape=(16,)))
		# now the model will take as input arrays of shape (*, 16)
		# and output arrays of shape (*, 32)
		# after the first layer, you don't need to specify
		# the size of the input anymore:
		model.add(Dense(32))
	```

	# Arguments
		activation: Activation function to use
			(see [activations](../activations.md)).
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation").
			(see [regularizer](../regularizers.md)).

	# Input shape
		nD tensor with shape: `(batch_size, sequence_length, timestep_dim)`.

	# Output shape
		nD tensor with shape: `(batch_size, 3*timestep_dim)`.
	"""

	def __init__(self, activation=None, activity_regularizer=None, **kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (None,kwargs.pop('input_dim'))

		super(MinMaxMean, self).__init__(**kwargs)

		self.activation = activations.get(activation)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.input_spec = InputSpec(ndim=3)
		# self.supports_masking = True # How to deal with this?

	def build(self, input_shape):
		assert len(input_shape) == 3
		input_dim = input_shape[-1]
		self.input_spec = InputSpec(ndim=3, axes={-1: input_dim})
		self.built = True

	def call(self, inputs):
		# boolean_mask(inputs,mask,axis=1) ?
		i_min = K.min(inputs, axis=1)
		i_max = K.max(inputs, axis=1)
		i_mean = K.mean(inputs, axis=1)

		output = K.concatenate((i_min,i_max,i_mean),axis=1)

		if self.activation is not None:
			output = self.activation(output)
		return output

	def compute_output_shape(self, input_shape):
		assert input_shape and len(input_shape) == 3
		assert input_shape[-1]
		output_shape = (input_shape[0], 3*input_shape[-1])
		return tuple(output_shape)

	def get_config(self):
		config = {
			'activation': activations.serialize(self.activation),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer)
		}
		base_config = super(MinMaxMean, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class MeanPool(Layer):
	# https://stackoverflow.com/a/42943769/11002708
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(MeanPool, self).__init__(**kwargs)

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		if mask is not None:
			# mask (batch, time)
			mask = K.cast(mask, K.floatx())
			# mask (batch, time, 'x')
			mask = mask.dimshuffle(0, 1, 'x')
			# to make the masked values in x be equal to zero
			x = x * mask
		return K.sum(x, axis=1) / K.sum(mask, axis=1)

	def get_output_shape_for(self, input_shape):
		# remove temporal dimension
		return input_shape[0], input_shape[2]
