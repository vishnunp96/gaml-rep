if __name__ == '__main__':

	from tensorflow.keras.layers import Dense,BatchNormalization
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.utils import plot_model
	from tensorflow.keras import optimizers
	from tensorflow.keras.regularizers import l2

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	from sklearn.model_selection import train_test_split

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os
	import itertools
	from collections import Counter

	#from annotations.bratutils import Standoff
	from annotations.brattowindow import StandoffLabels
	from annotations.wordembeddings import WordEmbeddings

	from sklearn.preprocessing import LabelBinarizer
	from sklearn.metrics import confusion_matrix
	from sklearn.externals import joblib

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	args = parser.parse_args()

	# Read in data
	types = None # ['MeasuredValue','Constraint','ParameterSymbol','ParameterName']
	anns = [StandoffLabels.open(path,types=types) for path in args.ann]
	anns = [labelled for labelled in anns if labelled.standoff]
	embeddings = WordEmbeddings.open(args.embeddings)

	window_size = 11
	assert window_size%2 == 1
	window_pad = window_size//2

	numpy.random.seed(42)
	padding_vector = numpy.random.normal(size=(embeddings.dim,))

	# Make test/train split
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
	ann_train,ann_test = train_test_split(anns, test_size=0.1, random_state=42)

	labels = LabelBinarizer().fit(list(set(l for a in anns for t,l in a)))

	def make_dataset(ann_list):
		xs = []
		ys = []
		previous = []
		for a in ann_list:
			tokens,tags = zip(*a)
			document_matrix = numpy.vstack([padding_vector]*window_pad + [embeddings[t] for t in tokens] + [padding_vector]*window_pad)

			for i,tag in enumerate(tags):
				j = i + window_pad
				window = document_matrix[(j-window_pad):(j+window_pad+1)].flatten()
				previous_tag = tags[i-1] if i>0 else StandoffLabels.outside_label
				xs.append(window)
				ys.append(tag)
				previous.append(previous_tag)

		return numpy.stack(xs),labels.transform(ys),labels.transform(previous)

	x_train,y_train,p_train = make_dataset(ann_train)
	x_test,y_test,p_test = make_dataset(ann_test)

	x_train = numpy.concatenate((x_train,p_train),axis=1)
	x_test = numpy.concatenate((x_test,p_test),axis=1)

	# Sanity check dataset
	assert numpy.all(numpy.isfinite(x_train)) and numpy.all(numpy.isfinite(x_test)) # Can we check if any entry is all zeroes?

	# Model parameters
	r = 0.01
	input_shape = x_train[0].shape
	output_num = len(labels.classes_)

	# Provide previous label to model

	# Build model
	model = Sequential(name=args.name)
	model.add(Dense(128, input_shape=input_shape, activation='relu',kernel_regularizer=l2(r), bias_regularizer=l2(r)))
	model.add(BatchNormalization(axis=-1))
	model.add(Dense(32, activation='relu',kernel_regularizer=l2(r), bias_regularizer=l2(r)))
	model.add(BatchNormalization(axis=-1))
	model.add(Dense(output_num, activation='softmax',kernel_regularizer=l2(r), bias_regularizer=l2(r)))

	# Compile model
	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', metrics=['accuracy','mse'], optimizer=opt)

	# Make directory to store summaries and results
	os.makedirs(model.name, exist_ok=True)

	plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 50

	# Train model
	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

	test_subjects = ann_test[10:15]
	x_subjects,y_subjects,p_subjects = make_dataset(test_subjects)

	#predict_subjects = model.predict(x_subjects)

	predict_subjects = []
	previous_pred = labels.transform([StandoffLabels.outside_label])
	for i,xi in enumerate(x_subjects):
		pred = model.predict(numpy.concatenate((xi.reshape(1,-1),previous_pred),axis=1))
		predict_subjects.append(pred.reshape(-1))
		previous_pred = pred
	predict_subjects = numpy.stack(predict_subjects)

	predicted_labels = labels.inverse_transform(predict_subjects)

	print_len = max(len(c) for c in labels.classes_) + 2
	for i,(token,label) in enumerate(itertools.chain.from_iterable(test_subjects)):
		#print(f'{token:20} {label:25} {predicted_labels[i]:25} [{", ".join(f"{p:.2f}" for p in predict_subjects[i])}]')
		vec_str = ', '.join(f'{p:.2f}' for p in predict_subjects[i])
		print(('{0:20} {1:'+str(print_len)+'} {2:'+str(print_len)+'} [{3}]').format(token,label,predicted_labels[i],vec_str))
	print(f'[{", ".join(str(i) for i in labels.classes_)}]')

	confusion = confusion_matrix(labels.inverse_transform(y_test), labels.inverse_transform(model.predict(x_test)),labels=labels.classes_)
	class_accuracies = confusion.diagonal()/confusion.sum(axis=1)
	counter = Counter(labels.inverse_transform(y_test))

	print(confusion)
	print(confusion.shape)
	print(class_accuracies)
	print(class_accuracies.shape)

	for label,acc in zip(labels.classes_,class_accuracies):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:6d}').format(label,acc,counter[label]))

	model.save(os.path.join(model.name,model.name+'.h5'))
	joblib.dump(labels,os.path.join(model.name,model.name+'_labels.joblib'))
