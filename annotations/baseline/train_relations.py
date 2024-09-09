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

	from annotations.bratutils import Standoff
	#from annotations.brattowindow import StandoffLabels
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
	anns = [Standoff.open(path) for path in args.ann]
	anns = [a for a in anns if a]
	embeddings = WordEmbeddings.open(args.embeddings)

	# Make test/train split
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
	ann_train,ann_test = train_test_split(anns, test_size=0.1, random_state=42)

	def minmaxmean(x):
		return numpy.concatenate([x.min(axis=0),x.max(axis=0),x.mean(axis=0)])

	numpy.random.seed(42)
	padding_vector = numpy.random.normal(size=(embeddings.dim,))

	labels = LabelBinarizer().fit(list(set(r.type for a in anns for r in a.relations))+['none'])
	entity_labels = LabelBinarizer().fit(list(set(e.type for a in anns for e in a.entities)))

	def make_dataset(ann_list):
		xs = []
		ys = []
		for a in ann_list:
			#for start,end in itertools.permutations(a.entities,2):
			#	rel = next((r.type for r in a.relations if r.arg1==start and r.arg2==end),None)
			for start,end in itertools.combinations(a.entities,2):
				rel = next((r.type for r in a.relations if set([start,end])==set([r.arg1,r.arg2])),'none')
				tokens = numpy.stack([padding_vector] + [embeddings[t] for t in a.text[start.end:end.start].split()] + [padding_vector])
				ent1 = numpy.stack(embeddings[t] for t in start.text.split())
				ent2 = numpy.stack(embeddings[t] for t in end.text.split())

				xs.append(numpy.concatenate([minmaxmean(tokens),minmaxmean(ent1),minmaxmean(ent2),entity_labels.transform([start.type,end.type]).reshape(-1)]))
				ys.append(rel)

		return numpy.stack(xs),labels.transform(ys)

	x_train,y_train = make_dataset(ann_train)
	x_test,y_test = make_dataset(ann_test)

	# Sanity check dataset
	assert numpy.all(numpy.isfinite(x_train)) and numpy.all(numpy.isfinite(x_test)) # Can we check if any entry is all zeroes?

	# Model parameters
	r = 0.01
	input_shape = x_train[0].shape
	output_num = len(labels.classes_)

	# Projection matrix? Can we do this in Keras? -> Custom Layer

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

	print_len = max(len(c) for c in labels.classes_) + 2
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
	joblib.dump(entity_labels,os.path.join(model.name,model.name+'_entity_labels.joblib'))
