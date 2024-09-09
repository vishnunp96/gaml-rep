if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	from tensorflow.keras.layers import Dense,Embedding,Bidirectional,TimeDistributed,LSTM,Input,concatenate,SpatialDropout1D
	from tensorflow.keras.models import Model
	from tensorflow.keras.callbacks import EarlyStopping
	from tensorflow.keras.utils import plot_model
	from tensorflow.keras import optimizers
	from tensorflow.keras.regularizers import l2

	from gaml.utilities.kerasutils import Projection

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas
	from sklearn.model_selection import train_test_split
	from tensorflow.keras.preprocessing.sequence import pad_sequences

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os

	from gaml.annotations.bratnormalisation import open_clean
	from gaml.annotations.brattowindow import StandoffLabels
	from gaml.annotations.wordembeddings import WordEmbeddings

	from sklearn.preprocessing import LabelBinarizer
	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
	from sklearn.externals import joblib

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	args = parser.parse_args()

	# Read in data
	types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName']
	#types = None
	#anns = [StandoffLabels.open(path,types=types) for path in args.ann]
	anns = []
	for path in args.ann:
		try:
			#anns.append(StandoffLabels.open(path,types=types,include_bio=True))
			standoff = open_clean(path,check_repetitions=('ParameterName','ParameterSymbol','ObjectName'))
			# Convert Constraints to MeasuredValues (we can recover Constraints using Attributes)
			for c in [e for e in standoff.entities if e.type=='Constraint']:
				c.type = 'MeasuredValue'
			if standoff:
				anns.append(StandoffLabels(standoff,types=types,include_bio=True))
		except KeyError:
			print(f'Error opening {path}')
	stopwatch.tick('Loaded dataset.',report=True)
	embeddings = WordEmbeddings.open(args.embeddings)
	stopwatch.tick('Loaded embeddings.',report=True)

	# Make test/train split
	ann_train,ann_test = train_test_split(anns, test_size=0.1, random_state=42)

	labels = LabelBinarizer().fit(list(set(l for a in anns for t,l in a)))

	# Get all characters used
	charset = set()
	for a in anns: charset.update(a.standoff.text)
	charindexdict = {c:(i+1) for i,c in enumerate(sorted(charset))} # +1 for mask index
	chardict = {i:c for c,i in charindexdict.items()}

	max_word_len = 26 # max(len(w) for w in embeddings.indexes.keys() if w is not None)
	char_vocab = len(charindexdict) + 1

	stopwatch.tick('Processed characters.',report=True)

	def make_dataset(ann_list):
		xs = []
		cs = []
		ys = []
		for a in ann_list:
			tokens,tags = zip(*a)
			xs.append([embeddings.getindex(t) for t in tokens])
			cs.append(pad_sequences([[charindexdict[c] for c in t] for t in tokens], padding='post', maxlen=max_word_len))
			ys.append(labels.transform(tags))

		xs = pad_sequences(xs,padding='post')
		padded_length = xs.shape[-1]

		cs = numpy.stack([numpy.pad(c, ((0,padded_length-c.shape[0]),(0,0)), mode='constant') for c in cs])
		ys = numpy.stack([numpy.pad(y, ((0,padded_length-y.shape[0]),(0,0)), mode='constant') for y in ys])

		return [xs,cs], ys

	x_train,y_train = make_dataset(ann_train)
	x_test,y_test = make_dataset(ann_test)

	print(x_train[0].shape)
	print(x_train[1].shape)
	print(y_train.shape)
	print(char_vocab)
	print(max_word_len)

	stopwatch.tick('Made datasets.',report=True)

	# Model parameters
	r = 0.01
	output_num = len(labels.classes_)


	### Build model
	# Word input and embedding/projection
	word_input = Input(shape=(None,))
	word_embedding = Embedding(embeddings.vocab_size, embeddings.dim, weights=[embeddings.values], trainable=False, mask_zero=True)(word_input)
	word_embedding = Projection(kernel_regularizer=l2(r))(word_embedding)

	# Character input and embedding
	chars_input = Input(shape=(None,max_word_len))
	chars_embedding = TimeDistributed(Embedding(char_vocab, 10, input_length=max_word_len, mask_zero=True))(chars_input)
	# LSTM to get word encodings from characters
	char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(chars_embedding)

	# Combine inputs (add dropout)
	x = concatenate([word_embedding, char_enc])
	x = SpatialDropout1D(0.3)(x)

	# Main LSTM network
	x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
	x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
	outputs = TimeDistributed(Dense(output_num, activation='softmax', kernel_regularizer=l2(r), bias_regularizer=l2(r)))(x)

	# Final model
	model = Model(inputs=[word_input,chars_input],outputs=outputs, name=args.name)


	# Compile model
	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', metrics=['accuracy','mse'], optimizer=opt)

	# Make directory to store summaries and results
	os.makedirs(model.name, exist_ok=True)

	plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	stopwatch.tick('Built model.',report=True)

	# Training parameters
	batch_size = 128
	epochs = 500 # 50

	callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, baseline=0.5, verbose=1, patience=25, restore_best_weights=True)]

	# Train model
	history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

	stopwatch.tick('Trained model.',report=True)

	test_subjects = ann_test[10:11]

	print_len = max(len(c) for c in labels.classes_) + 2
	for i,a in enumerate(test_subjects):
		x_subject, y_subject = make_dataset([a])
		predict_subject = model.predict(x_subject)[0]
		predicted_labels = labels.inverse_transform(predict_subject)
		for j,(token,label) in enumerate(a):
			vec_str = ', '.join(f'{p:.2f}' for p in predict_subject[j])
			print(('{0:20} {1:'+str(print_len)+'} {2:'+str(print_len)+'} [{3}]').format(token,label,predicted_labels[j],vec_str))
	print(f'[{", ".join(str(i) for i in labels.classes_)}]')

	test_labels = []
	test_truth = []
	for a in ann_test:
		pred = labels.inverse_transform(model.predict(make_dataset([a])[0])[0])
		_,truth = zip(*a)
		test_labels += list(pred)
		test_truth += list(truth)

	confusion = confusion_matrix(test_truth, test_labels, labels=labels.classes_)
	print(confusion)
	print(confusion.shape)
	p,r,f,s = precision_recall_fscore_support(test_truth, test_labels, labels=labels.classes_)

	for i,label in enumerate(labels.classes_):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

	model.save(os.path.join(model.name,model.name+'.h5'))
	joblib.dump(labels,os.path.join(model.name,model.name+'_labels.joblib'))
	pandas.DataFrame(history.history,index=pandas.Index(history.epoch,name='epoch')).to_csv(os.path.join(model.name,'training_logs.csv'))

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


	stopwatch.report()
