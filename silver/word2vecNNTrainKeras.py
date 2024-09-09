from lxml import etree as let
from gaml.preprocessing import latexmlpy as latexml
#from gaml.utilities.gxml import fastxmliter
import numpy

def make_dataset(manifest, arXiv_ids, label, embeddings, force=False):
	xs = []
	for arXiv in arXiv_ids:
		path = manifest[arXiv]
		if path:
			#text = ''
			#for event,p in fastxmliter(path, events=("end",), tag='abstract'):
			#	text += latexml.tostring(p)[0] + '\n'
			root = let.parse(path).getroot()
			elems = root.findall('.//abstract')
			if elems or force:
				text = ''
				for e in elems:
					text += latexml.tostring(e)[0] + '\n'
				if text or force:
					xs.append(embeddings.tokenise(text))
		elif force:
			xs.append([])

	return xs,numpy.full(len(xs), label)


if __name__ == '__main__':

	from tensorflow.keras.layers import Dense,Embedding
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.utils import plot_model
	from tensorflow.keras import optimizers
	from tensorflow.keras.regularizers import l2
	from tensorflow.keras.preprocessing.sequence import pad_sequences
	from tensorflow.keras.utils import Sequence

	from gaml.utilities.kerasutils import Projection,MinMaxMeanPool#,PretrainedEmbedding

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import pandas
	from sklearn.model_selection import train_test_split

	from gaml.utilities.argparseactions import ArgumentParser,FileAction
	import os

	from gaml.preprocessing.manifest import ManifestAction
	from gaml.annotations.wordembeddings import WordEmbeddings

	parser = ArgumentParser(description='Train Keras ANN to predict probability of new measurement being reported in text.')
	parser.add_argument('scores',action=FileAction, mustexist=True,help='Location of scores to use when selecting training data.')
	parser.add_argument('source',action=ManifestAction,help='Source directory containing manifest.json.')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	parser.add_argument('-s','--minscore',type=int,default=1,help='Minimum score for data to be considered a positive sample. Defaults to 1.')
	parser.add_argument('-l','--lossdata',action=FileAction,default='lossdata.csv',mustexist=False,help='Location of loss data csv file. Defaults to \'lossdata.csv\'.')
	parser.add_argument('-t','--testfrac',type=float,default=0.1,help='Fraction of sampled positive data to use as testing samples. Defaults to 0.1.')
	parser.add_argument('-y','--predictions',action=FileAction,mustexist=False,help='Path at which to store model predictions.')
	parser.add_argument('-x','--exclude',action=FileAction,mustexist=False,help='File containing ids to exclude from available data.')
	args = parser.parse_args()

	numpy.random.seed(42)

	# Read in data
	scores = pandas.read_csv(args.scores)

	if args.exclude:
		with open(args.exclude,'r') as exclude:
			ids = [l.strip() for l in exclude]
		scores = scores[~scores['id'].isin(ids)]

	embeddings = WordEmbeddings.open(args.embeddings)

	assert numpy.all(numpy.isfinite(embeddings.values))

	print('Opened files.')

	positive_ids = scores[scores['score'] >= args.minscore]['id'].values
	negative_ids = scores[scores['score'] == 0]['id'].values

	pos_train_ids, pos_test_ids = train_test_split(positive_ids, test_size=args.testfrac, random_state=42)
	neg_remaining, neg_test_ids = train_test_split(negative_ids, test_size=len(pos_test_ids), random_state=42)

	print('pos_train_ids',pos_train_ids.shape)
	print('pos_test_ids',pos_test_ids.shape)
	print('neg_test_ids',neg_test_ids.shape)
	print('neg_remaining',neg_remaining.shape)

	print('Compiled ids.')


	class DataGenerator(Sequence):
		'Generates data for Keras'
		def __init__(self, pos_ids, neg_ids, batch_size=32, shuffle=True):
			'Initialization'
			self.pos_ids = pos_ids
			self.all_neg_ids = neg_ids

			self.pos_ids_count = len(pos_ids)

			self.batch_size = batch_size
			self.shuffle = shuffle

			self.pos_x, self.pos_y = make_dataset(args.source, self.pos_ids, 1.0, embeddings)
			self.pos_indexes = numpy.arange(len(self.pos_x))

			self.on_epoch_end()

		def __len__(self):
			'Denotes the number of batches per epoch'
			return int(numpy.ceil(2*len(self.pos_ids) / float(self.batch_size))) ## Estimate?

		def __getitem__(self, index):
			'Generate one batch of data'
			# Generate indexes of the batch
			pos_batch_size = int(numpy.floor(self.batch_size/2))
			neg_batch_size = self.batch_size - pos_batch_size
			pos_indexes = self.pos_indexes[index*pos_batch_size:(index+1)*pos_batch_size]
			neg_indexes = self.neg_indexes[index*neg_batch_size:(index+1)*neg_batch_size]

			#print('pos_batch_size',pos_batch_size)
			#print('neg_batch_size',neg_batch_size)
			#print('self.pos_indexes',type(self.pos_indexes))
			#print('self.neg_indexes',type(self.neg_indexes))
			#print('pos_indexes',type(pos_indexes))
			#print('neg_indexes',type(neg_indexes))
			#print('self.pos_x',type(self.pos_x))
			#print('self.pos_y',type(self.pos_y))

			X = pad_sequences([self.pos_x[i] for i in pos_indexes] + [self.neg_x[j] for j in neg_indexes], padding='post')
			y = numpy.concatenate([self.pos_y[pos_indexes],self.neg_y[neg_indexes]],axis=0)

			return X, y

		def on_epoch_end(self):
			'Updates indexes after each epoch'
			self.neg_ids = numpy.random.choice(self.all_neg_ids,size=self.pos_ids_count)

			self.neg_x, self.neg_y = make_dataset(args.source, self.neg_ids, 0.0, embeddings)

			self.neg_indexes = numpy.arange(len(self.neg_x))
			if self.shuffle == True:
				numpy.random.shuffle(self.pos_indexes)
				numpy.random.shuffle(self.neg_indexes)

	#pos_train_x, pos_train_y = make_dataset(pos_train_ids, 1.0)
	pos_test_x, pos_test_y = make_dataset(args.source, pos_test_ids, 1.0, embeddings)
	neg_test_x, neg_test_y = make_dataset(args.source, neg_test_ids, 0.0, embeddings)

	x_test = pad_sequences(pos_test_x + neg_test_x,padding='post')
	y_test = numpy.concatenate([pos_test_y,neg_test_y], axis=0)

	# Model parameters
	r = 0.01

	# Build model
	model = Sequential(name=args.name)

	model.add(Embedding(embeddings.vocab_size, embeddings.dim, weights=[embeddings.values], trainable=False, mask_zero=True))
	#model.add(PretrainedEmbedding(embeddings.values, mask_zero=True))
	model.add(Projection(kernel_regularizer=l2(r)))
	model.add(MinMaxMeanPool())
	model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(r), bias_regularizer=l2(r)))

	# Compile model
	opt = optimizers.Adam(lr=0.00025)
	model.compile(loss='binary_crossentropy', metrics=['accuracy','mse'], optimizer=opt)

	# Make directory to store summaries and results
	os.makedirs(model.name, exist_ok=True)

	plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 25

	train_generator = DataGenerator(pos_train_ids, neg_remaining, batch_size=batch_size)

	history = model.fit_generator(train_generator, validation_data=(x_test,y_test),epochs=epochs)

	# Train model
	#for i in range(epochs):
	#	numpy.random.seed(i+1)
	#	neg_train_x, neg_train_y = make_dataset(numpy.random.choice(neg_remaining,size=len(pos_train_x)),0.0)
	#	x_train = pad_sequences(pos_train_x + neg_train_x, padding='post')
	#	y_train = numpy.concatenate([pos_train_y,neg_train_y],axis=0)
	#	#print(numpy.sum(numpy.all(x_train==0,axis=1)))
	#	#print(numpy.sum(numpy.all(x_test==0,axis=1)))
	#	#print('x_train',x_train.shape)
	#	#print('y_train',y_train.shape)
	#	#print('x_test',x_test.shape)
	#	#print('y_test',y_test.shape)
	#	model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=1,batch_size=batch_size,shuffle=True)
	#	#r = model.predict(x_train)
	#	#print('isfinite',numpy.all(numpy.isfinite(r)))
	#	#print(r[~numpy.isfinite(r)])
	#	#print(len(r[~numpy.isfinite(r)]))
	#	#print(r[0])

	model.save(os.path.join(model.name,model.name+'.h5'))

	# Plot training & validation accuracy values
	plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(os.path.join(model.name,'accuracy_curve.png'))

	# Plot training & validation loss values
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(os.path.join(model.name,'loss_curve.png'))
