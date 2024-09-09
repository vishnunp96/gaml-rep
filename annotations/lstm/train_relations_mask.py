if __name__ == '__main__':

	from tensorflow.keras.layers import Dense,Embedding,Bidirectional,LSTM,Input,concatenate
	from tensorflow.keras.models import Model
	from tensorflow.keras.utils import plot_model
	from tensorflow.keras import optimizers
	from tensorflow.keras.regularizers import l2
	from tensorflow.keras.utils import Sequence

	from gaml.utilities.kerasutils import Projection

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	from sklearn.model_selection import train_test_split
	from tensorflow.keras.preprocessing.sequence import pad_sequences

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os
	import itertools

	from gaml.annotations.bratnormalisation import open_clean
	from gaml.annotations.brattowindow import StandoffLabels
	from gaml.annotations.bratutils import StandoffConfig
	from gaml.annotations.wordembeddings import WordEmbeddings

	from sklearn.preprocessing import LabelBinarizer
	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
	from sklearn.externals import joblib

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('conf',action=FileAction, mustexist=True,help='annotation.conf file for .ann files.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	args = parser.parse_args()

	numpy.random.seed(42)

	# Read in data
	entity_types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName']
	#types = None
	#anns = [StandoffLabels.open(path,types=types) for path in args.ann]
	anns = []
	for path in args.ann:
		try:
			#anns.append(StandoffLabels.open(path,types=types,include_bio=True))
			anns.append(StandoffLabels(open_clean(path,check_repetitions=('ParameterName','ParameterSymbol','ObjectName')),types=entity_types,include_bio=True))
		except KeyError:
			print(f'Error opening {path}')
	anns = [labelled for labelled in anns if labelled.standoff]
	conf = StandoffConfig.open(args.conf)
	embeddings = WordEmbeddings.open(args.embeddings)

	# Make test/train split
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
	ann_train,ann_test = train_test_split(anns, test_size=0.1, random_state=42)

	labels = LabelBinarizer().fit(list(set(r.type for a in anns for r in a.standoff.relations))+['none'])
	entity_labels = LabelBinarizer().fit(list(set(e.type for a in anns for e in a.standoff.entities)))

	def make_dataset(ann_list):
		all_tokens = []
		entity_masks = []
		ys = []
		#total = 0
		for a in ann_list:
			tokens = [embeddings.getindex(t) for t in a.tokens]
			#count = 0
			for start,end in itertools.permutations([e for e in a.standoff.entities if e.type in entity_types],2):
				if conf.allowed_relation(start,end):
					ys.append(next((r.type for r in a.standoff.relations if r.arg1==start and r.arg2==end),'none'))
					entity_masks.append(numpy.stack([a.entity_mask(start), a.entity_mask(end)], axis=1))
					all_tokens.append(tokens)
					#total += 1
					#count += 1
			#print(f'Count for {a.standoff.path} = {count} ({total}) with {len(a.standoff.entities)} entities')
		#print(f'Total = {total}')

		all_tokens = pad_sequences(all_tokens, padding='post')
		padded_length = all_tokens.shape[-1]

		entity_masks = numpy.stack([numpy.pad(m, ((0,padded_length-m.shape[0]),(0,0)), mode='constant') for m in entity_masks])

		#print(Counter(ys))

		return all_tokens, entity_masks, numpy.array(ys)

	x_train_1,x_train_2,y_train_labels = make_dataset(ann_train)
	x_test_1,x_test_2,y_test_labels = make_dataset(ann_test)
	y_test = labels.transform(y_test_labels)

	print(y_train_labels[0])
	print(x_train_1[0])
	print(x_train_2[0])

	# Sanity check dataset
	assert numpy.all(numpy.isfinite(x_train_1)) and numpy.all(numpy.isfinite(x_train_2))
	assert numpy.all(numpy.isfinite(x_test_1)) and numpy.all(numpy.isfinite(x_test_2))

	# Model parameters
	r = 0.01
	output_num = len(labels.classes_)

	# Projection matrix? Can we do this in Keras? -> Custom Layer

	# Build model
	# Inputs
	token_input = Input(name='tokens',shape=(None,)) # (num_tokens,)
	#ent1_mask_input = Input(shape=(None,)) # (num_tokens,)
	#ent2_mask_input = Input(shape=(None,)) # (num_tokens,)
	ent_mask_input = Input(name='entity_mask',shape=(None,2)) # (num_tokens,2)
	# Embedding tokens
	embedding_layer = Embedding(embeddings.vocab_size, embeddings.dim, weights=[embeddings.values], trainable=False, mask_zero=True)
	projection_layer = Projection(kernel_regularizer=l2(r))
	embedded_tokens = projection_layer(embedding_layer(token_input)) # (num_tokens,dim)
	# Combining inputs
	#all_input = concatenate([embedded_tokens, ent1_mask_input, ent2_mask_input], axis=-1) #(num_tokens,dim+2)
	all_input = concatenate([embedded_tokens, ent_mask_input], axis=-1) #(num_tokens,dim+2)
	# Recurrent layers
	x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(all_input)
	#x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
	# Summary layers
	x = Bidirectional(LSTM(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
	prediction = Dense(output_num, name='output', activation='softmax', kernel_regularizer=l2(r), bias_regularizer=l2(r))(x)
	# Create model
	#model = Model(inputs=[token_input,ent1_mask_input,ent2_mask_input], outputs=prediction)
	model = Model(inputs=[token_input,ent_mask_input], outputs=prediction, name=args.name)

	# Compile model
	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', metrics=['accuracy','mse'], optimizer=opt)

	# Make directory to store summaries and results
	os.makedirs(model.name, exist_ok=True)

	plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 10

	class DataGenerator(Sequence):
		'Generates data for Keras'
		def __init__(self, x1, x2, y_labels, none_count=2000, batch_size=32, shuffle=True):
			'Initialization'
			self.x1, self.x2, self.y = x1, x2, labels.transform(y_labels)

			none_idxs = (y_labels == 'none')

			self.indexes = numpy.arange(self.x1.shape[0])
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
			return [self.x1[batch_indexes], self.x2[batch_indexes]], self.y[batch_indexes]

		def on_epoch_end(self):
			'Updates indexes after each epoch'
			self.epoch_indexes = numpy.concatenate([self.other_idxs, numpy.random.choice(self.none_idxs, size=self.none_count)])
			if self.shuffle == True:
				numpy.random.shuffle(self.epoch_indexes)

			#print(f'self.epoch_indexes.shape: {self.epoch_indexes.shape}')

	# Train model
	generator = DataGenerator(x_train_1, x_train_2, y_train_labels, none_count=4000, batch_size=batch_size, shuffle=True)

	history = model.fit_generator(generator, validation_data=([x_test_1,x_test_2], y_test), epochs=epochs)

	model.save(os.path.join(model.name,model.name+'.h5'))
	joblib.dump(labels,os.path.join(model.name,model.name+'_labels.joblib'))
	joblib.dump(entity_labels,os.path.join(model.name,model.name+'_entity_labels.joblib'))

	## Statistics
	print_len = max(len(c) for c in labels.classes_) + 2

	test_truth = y_test_labels
	test_labels = labels.inverse_transform(model.predict([x_test_1,x_test_2]))

	confusion = confusion_matrix(test_truth, test_labels, labels=labels.classes_)
	print(confusion)
	print(confusion.shape)
	p,r,f,s = precision_recall_fscore_support(test_truth, test_labels, labels=labels.classes_)

	for i,label in enumerate(labels.classes_):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

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

