if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	from sklearn.model_selection import train_test_split

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os
	import itertools
	from collections import defaultdict

	from annotations.bratnormalisation import open_clean
	#from annotations.bratutils import Standoff
	from annotations.brattowindow import StandoffLabels
	from annotations.wordembeddings import WordEmbeddings

	from sklearn.preprocessing import LabelBinarizer,LabelEncoder
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
	labels_int = LabelEncoder().fit(list(set(l for a in anns for t,l in a)))

	class BaselineEntityDataset(torch.utils.data.Dataset):
		def __init__(self, ann_list, transform=None):
			self.anns = ann_list
			self.transform = transform

			xs = []
			ys = []
			previous = []
			for a in self.anns:
				tokens,tags = zip(*a)
				document_matrix = numpy.vstack([padding_vector]*window_pad + [embeddings[t] for t in tokens] + [padding_vector]*window_pad)

				for i,tag in enumerate(tags):
					j = i + window_pad
					window = document_matrix[(j-window_pad):(j+window_pad+1)].flatten()
					previous_tag = tags[i-1] if i>0 else StandoffLabels.outside_label
					xs.append(window)
					ys.append(tag)
					previous.append(previous_tag)

			all_x = numpy.stack(xs)
			all_previous = labels.transform(previous)

			self.y = labels_int.transform(ys)
			self.y_labels = ys
			self.x_raw = all_x
			self.x = numpy.concatenate((all_x,all_previous),axis=1)

			assert numpy.all(numpy.isfinite(self.x)) # Can we check if any entry is all zeroes?

		def __len__(self):
			return len(self.x)

		def __getitem__(self, idx):
			if torch.is_tensor(idx):
				idx = idx.tolist()

			x,y = self.x[idx], self.y[idx]

			x,y = torch.from_numpy(x).float(), torch.tensor(y).long()

			return x,y

		def get_data(self):
			return torch.from_numpy(self.x).float(), torch.tensor(self.y).long()

	train_dataset = BaselineEntityDataset(ann_train)
	test_dataset = BaselineEntityDataset(ann_test)


	# Sanity check dataset

	# Model parameters
	r = 0.01
	input_len = embeddings.dim * window_size + len(labels.classes_)
	output_num = len(labels.classes_)

	# Provide previous label to model

	# Define model
	class BaselineEntityModel(nn.Module):
		def __init__(self, in_features, output_num):
			super(BaselineEntityModel,self).__init__()
			self.dense1 = nn.Linear(in_features, 128, bias=True)
			self.dense2 = nn.Linear(128, 32)
			self.output_dense = nn.Linear(32,output_num)

		def forward(self, x):
			x = F.relu(self.dense1(x))
			x = F.relu(self.dense2(x))
			#x = F.softmax(self.output_dense(x), dim=1)
			x = self.output_dense(x)
			return x

	# Create model
	model = BaselineEntityModel(input_len, output_num).float().cuda()
	print(model)

	opt = optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	# Make directory to store summaries and results
	os.makedirs(args.name, exist_ok=True)

	#plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 2

	# Setup training
	dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	history = defaultdict(list)

	# Train model
	val_x,val_y = test_dataset.get_data()
	for epoch in range(epochs):
		print(f'Epoch {epoch}')
		epoch_loss = 0.0
		for batch_i,(batch_x,batch_y) in enumerate(dataloader):
			#print(f'Batch: {batch_i}')
			#batch_y = batch_y.long()
			opt.zero_grad()
			output = model(batch_x.cuda())
			loss = criterion(output, batch_y.cuda())
			loss.backward()
			opt.step()
			epoch_loss += output.shape[0] * loss.item()
		epoch_loss = epoch_loss/len(train_dataset)
		with torch.no_grad():
			model.eval()
			val_output = model(val_x.cuda())
			val_loss = criterion(val_output, val_y.cuda()).item()
			model.train()
		history['loss'].append(epoch_loss)
		history['val_loss'].append(val_loss)

		print(f'Epoch loss: {epoch_loss:.4f}, Validation loss: {val_loss:.4f}')

	torch.save(model.state_dict(), os.path.join(args.name,args.name+'.pt'))
	joblib.dump(labels,os.path.join(args.name,args.name+'_labels.joblib'))

	model.eval()

	test_subjects = ann_test[10:11]
	subjects_dataset = BaselineEntityDataset(test_subjects)

	#predict_subjects = model.predict(x_subjects)

	predict_subjects = []
	previous_pred = labels.transform([StandoffLabels.outside_label])

	for i,xi in enumerate(subjects_dataset.x_raw):
		xi_in = numpy.concatenate((xi.reshape(1,-1),previous_pred),axis=1)
		pred = F.softmax(model(torch.from_numpy(xi_in).float().cuda()),dim=1).cpu().detach().numpy()
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



	predict_test = []
	previous_pred = labels.transform([StandoffLabels.outside_label])
	for i,xi in enumerate(test_dataset.x_raw):
		xi_in = numpy.concatenate((xi.reshape(1,-1),previous_pred),axis=1)
		pred = F.softmax(model(torch.from_numpy(xi_in).float().cuda()),dim=1).cpu().detach().numpy()
		predict_test.append(pred.reshape(-1))
		previous_pred = pred
	predict_test = numpy.stack(predict_test)
	predicted_labels = labels.inverse_transform(predict_test)


	confusion = confusion_matrix(test_dataset.y_labels, predicted_labels, labels=labels.classes_)
	print(confusion)
	print(confusion.shape)
	p,r,f,s = precision_recall_fscore_support(test_dataset.y_labels, predicted_labels, labels=labels.classes_)

	print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
	for i,label in enumerate(labels.classes_):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

	handles = set((k if not k.startswith('val_') else k[4:]) for k in history.keys())
	for h in handles:
		if ('val_'+h) in history and h in history:
			plt.figure()
			plt.plot(history[h])
			plt.plot(history['val_'+h])
			plt.title('Model ' + h)
			plt.ylabel(h)
			plt.xlabel('Epoch')
			plt.legend(['Train', 'Test'], loc='upper left')
			plt.savefig(os.path.join(args.name,h+'_curve.png'))

	stopwatch.report()
