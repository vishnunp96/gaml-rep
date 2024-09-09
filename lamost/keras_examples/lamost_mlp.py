if __name__ == '__main__':

	from tensorflow.keras.layers import Dense,BatchNormalization,Input
	from tensorflow.keras.models import Model #Sequential
	#from tensorflow.keras.losses import mse #binary_crossentropy
	from tensorflow.keras.utils import plot_model
	from tensorflow.keras import optimizers
	from tensorflow.keras.regularizers import l2
	#from tensorflow.keras import backend as K

	from sklearn import preprocessing

	#import numpy as np
	import matplotlib.pyplot as plt

	plt.switch_backend('agg')

	import pandas
	import numpy
	from sklearn.model_selection import train_test_split

	from gaml.utilities.argparseactions import ArgumentParser,FileAction
	import os

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
	x,y_unscaled = data[:,:-1],data[:,-1].reshape(-1,1) # This should be right...

	good_rows = numpy.all(numpy.isfinite(x),axis=1)

	x = x[good_rows]
	y_unscaled = y_unscaled[good_rows]

	assert not numpy.any(numpy.isnan(x)) # Just checking
	assert numpy.all(numpy.isfinite(x)) # Just checking
	assert not numpy.any(~x.any(axis=1))

	#scaler = preprocessing.StandardScaler()
	scaler = preprocessing.MinMaxScaler()
	y = scaler.fit_transform(y_unscaled)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

	original_dim = x_train.shape[1]

	print(f'Data shape: {x.shape}/{y.shape}')
	print(f'Train data shape: {x_train.shape}/{y_train.shape}')
	print(f'Test data shape: {x_test.shape}/{y_test.shape}')
	print(f'Data dimensionality: {original_dim}')

	#model = Sequential(name=args.name)
	#model.add(Input(shape=(original_dim,)))
	#model.add(BatchNormalization(axis=1,input_shape=(original_dim,)))
	#model.add(Dense(128, input_shape=(original_dim,), activation='tanh',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
	#model.add(BatchNormalization(axis=1))
	#model.add(Dense(32, activation='tanh',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
	#model.add(BatchNormalization(axis=1))
	#model.add(Dense(1, activation='linear',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

	inputs = Input(shape=(original_dim,))
	#x = BatchNormalization(axis=1,input_shape=(original_dim,))(inputs)
	x = Dense(128, input_shape=(original_dim,), activation='tanh',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs)
	x = BatchNormalization(axis=1)(x)
	x = Dense(32, activation='tanh',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
	x = BatchNormalization(axis=1)(x)
	prediction = Dense(1, activation='linear',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
	model = Model(inputs=inputs,outputs=prediction,name=args.name)

	opt = optimizers.Adam(lr=0.0001,clipnorm=0.1)
	#opt = optimizers.SGD(lr=0.01)

	model.compile(loss='mse', optimizer='adam')

	# Make directory to store summaries and results
	os.makedirs(model.name, exist_ok=True)

	plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	batch_size = 128
	epochs = 50

	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

	y_test_unscaled = scaler.inverse_transform(y_test)
	y_pred = model.predict(x_test,batch_size=batch_size)

	print(f'y_test shape: {y_test.shape}')
	print(f'y_test_unscaled shape: {y_test_unscaled.shape}')
	print(f'y_pred shape: {y_pred.shape}')

	print(repr(y_test[:5]))
	print(repr(y_test_unscaled[:5]))
	print(repr(y_pred[:5]))

	y_pred_unscaled = scaler.inverse_transform(y_pred)
	print(f'y_pred_unscaled shape: {y_pred_unscaled.shape}')
	print(repr(y_pred_unscaled[:5]))

	plt.figure()
	plt.scatter(y_test_unscaled, y_pred_unscaled)
	plt.xlabel("Truth")
	plt.ylabel("Prediction")
	plt.savefig(os.path.join(model.name,'test_predictions.png'))

