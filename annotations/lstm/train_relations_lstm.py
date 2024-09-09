if __name__ == '__main__':

	from tensorflow.keras.layers import Dense,Embedding,Bidirectional,LSTM
	from tensorflow.keras.models import Sequential
	#from tensorflow.keras.callbacks import EarlyStopping
	from tensorflow.keras.utils import plot_model
	from tensorflow.keras import optimizers
	from tensorflow.keras.regularizers import l2
	from tensorflow.keras.utils import Sequence

	from utilities.kerasutils import Projection

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas
	from sklearn.model_selection import train_test_split
	from tensorflow.keras.preprocessing.sequence import pad_sequences

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os
	import itertools

	from annotations.bratnormalisation import open_clean
	#from annotations.bratutils import Standoff
	#from annotations.brattowindow import StandoffLabels
	from annotations.wordembeddings import WordEmbeddings

	from sklearn.preprocessing import LabelBinarizer
	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
	from sklearn.externals import joblib

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	args = parser.parse_args()

	# Read in data
	entity_types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName']
	relation_types = ['Measurement','Confidence','Name','Property']
	anns = []
	for path in args.ann:
		try:
			#anns.append(StandoffLabels.open(path,types=types,include_bio=True))
			standoff = open_clean(path,check_repetitions=('ParameterName','ParameterSymbol','ObjectName'))
			# Convert Constraints to MeasuredValues (we can recover Constraints using Attributes)
			for c in [e for e in standoff.entities if e.type=='Constraint']:
				c.type = 'MeasuredValue'
			if standoff:
				#anns.append(StandoffLabels(standoff,types=types,include_bio=True))
				anns.append(standoff)
		except KeyError:
			print(f'Error opening {path}')
	embeddings = WordEmbeddings.open(args.embeddings)

	# Make test/train split
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
	ann_train,ann_test = train_test_split(anns, test_size=0.1, random_state=42)

	#labels = LabelBinarizer().fit(list(set(r.type for a in anns for r in a.relations))+['none'])
	labels = LabelBinarizer().fit(list(set(relation_types))+['none'])

	def make_dataset(ann_list):
		xs = []
		ys = []
		for a in ann_list:
			#for start,end in itertools.permutations(a.entities,2):
			#	rel = next((r.type for r in a.relations if r.arg1==start and r.arg2==end),None)
			for start,end in itertools.combinations([e for e in a.entities if e.type in entity_types],2):
				rel = next((r.type for r in a.relations if set([start,end])==set([r.arg1,r.arg2]) and r.type in relation_types),'none')
				#tokens = numpy.array([embeddings.getindex(t) for t in a.text[start.end:end.start].split()])
				#ent1 = numpy.stack(embeddings.getindex(t) for t in start.text.split())
				#ent2 = numpy.stack(embeddings.getindex(t) for t in end.text.split())

				tokens = [embeddings.getindex(t) for t in a.text[start.start:end.end].split()]

				xs.append(tokens)
				ys.append(rel)

		xs = pad_sequences(xs,padding='post')

		return xs,labels.transform(ys)

	x_train,y_train = make_dataset(ann_train)
	x_test,y_test = make_dataset(ann_test)

	# Model parameters
	r = 0.01
	output_num = len(labels.classes_)

	# Build model
	model = Sequential(name=args.name)
	model.add(Embedding(embeddings.vocab_size, embeddings.dim, weights=[embeddings.values], trainable=False, mask_zero=True))
	model.add(Projection(kernel_regularizer=l2(r)))
	model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
	model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))
	model.add(Dense(output_num, activation='softmax', kernel_regularizer=l2(r), bias_regularizer=l2(r)))

	# Compile model
	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', metrics=['accuracy','mse'], optimizer=opt)

	# Make directory to store summaries and results
	os.makedirs(model.name, exist_ok=True)

	plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 10 # 50

	class DataGenerator(Sequence):
		'Generates data for Keras'
		def __init__(self, x, y, none_count=2000, batch_size=32, shuffle=True):
			'Initialization'
			self.x, self.y = x, y

			none_idxs = (labels.inverse_transform(y) == 'none')

			self.indexes = numpy.arange(self.x.shape[0])
			self.other_idxs = self.indexes[~none_idxs]
			self.none_idxs = self.indexes[none_idxs]

			self.none_count = none_count
			self.other_count = self.other_idxs.shape[0]
			self.batch_size = batch_size
			self.shuffle = shuffle

			self.on_epoch_end()

		def __len__(self):
			'Denotes the number of batches per epoch'
			return int(numpy.ceil((self.none_count + self.other_count) / float(self.batch_size)))

		def __getitem__(self, index):
			'Generate one batch of data'
			batch_indexes = self.epoch_indexes[index*self.batch_size:(index+1)*self.batch_size]
			#print(f'batch_indexes.shape: {batch_indexes.shape}, batch_size = {self.batch_size}, index = {index}')
			return self.x[batch_indexes], self.y[batch_indexes]

		def on_epoch_end(self):
			'Updates indexes after each epoch'
			self.epoch_indexes = numpy.concatenate([self.other_idxs, numpy.random.choice(self.none_idxs, size=self.none_count)])
			if self.shuffle == True:
				numpy.random.shuffle(self.epoch_indexes)

			#print(f'self.epoch_indexes.shape: {self.epoch_indexes.shape}')

	# Train model
	generator = DataGenerator(x_train, y_train, none_count=4000, batch_size=batch_size, shuffle=True)

	history = model.fit_generator(generator, validation_data=(x_test, y_test), epochs=epochs)

	#callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, baseline=0.5, verbose=1, patience=25, restore_best_weights=True)]
	# Train model
	#history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

	# Print validation statistics
	test_truth = labels.inverse_transform(y_test)
	test_labels = labels.inverse_transform(model.predict(x_test))

	confusion = confusion_matrix(test_truth, test_labels, labels=labels.classes_)
	print(confusion)
	print(confusion.shape)
	p,r,f,s = precision_recall_fscore_support(test_truth, test_labels, labels=labels.classes_)

	print_len = max(len(c) for c in labels.classes_) + 2
	for i,label in enumerate(labels.classes_):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

	# Save model
	model.save(os.path.join(model.name,model.name+'.h5'))
	joblib.dump(labels,os.path.join(model.name,model.name+'_labels.joblib'))
	pandas.DataFrame(history.history,index=pandas.Index(history.epoch,name='epoch')).to_csv(os.path.join(model.name,'training_logs.csv'))

	# Make training graphs
	handles = set((k if not k.startswith('val_') else k[4:]) for k in history.history.keys())
	for h in handles:
		if ('val_'+h) in history.history and h in history.history:
			plt.figure()
			plt.plot(history.history[h])
			plt.plot(history.history['val_'+h])
			plt.title('Model ' + h)
			plt.ylabel(h)
			plt.xlabel('Epoch')
			plt.legend(['Train', 'Test'], loc='upper left')
			plt.savefig(os.path.join(model.name,h+'_curve.png'))
