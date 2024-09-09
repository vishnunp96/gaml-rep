if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	import torch.nn.utils.rnn as rnn

	from utilities.torchutils import unpack_sequence,train # predict_from_dataloader

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas
	from sklearn.model_selection import train_test_split
	from collections import defaultdict

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os

	#from annotations.bratnormalisation import open_clean
	#from annotations.brattowindow import StandoffLabels
	from annotations.wordembeddings import WordEmbeddings
	from annotations.annmlutils import open_anns

	from sklearn.preprocessing import LabelEncoder
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
	anns = open_anns(args.ann,types=types,use_labelled=True)
	embeddings = WordEmbeddings.open(args.embeddings)

	stopwatch.tick('Opened all files',report=True)

	# Make test/train split
	# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics
	ann_train,ann_other = train_test_split(anns, test_size=0.2, random_state=42)
	ann_dev,ann_test = train_test_split(ann_other, test_size=0.5, random_state=42)

	labels = LabelEncoder().fit(list(set(l for a in anns for t,l in a)))

	class BaselineRNNEntityDataset(torch.utils.data.Dataset):
		def __init__(self, ann_list):
			self.anns = ann_list

			xs = []
			ys = []
			for a in ann_list:
				tokens,tags = zip(*a)
				xs.append(numpy.array([embeddings.getindex(t) for t in tokens]))
				ys.append(labels.transform(tags))

			self.x = xs
			self.y = ys

		def __len__(self):
			return len(self.x)

		def __getitem__(self, idx):
			if torch.is_tensor(idx):
				idx = idx.tolist()

			x,y = self.x[idx], self.y[idx]

			x,y = torch.from_numpy(x).long(), torch.tensor(y).long()

			return x,y

		def get_data(self):
			xs = rnn.pack_sequence([torch.from_numpy(i).long() for i in self.x],False)
			ys = rnn.pack_sequence([torch.from_numpy(i).long() for i in self.y],False)
			return xs,ys

	train_dataset = BaselineRNNEntityDataset(ann_train)
	dev_dataset = BaselineRNNEntityDataset(ann_dev)
	test_dataset = BaselineRNNEntityDataset(ann_test)

	stopwatch.tick(f'Constructed datasets: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} testing samples',report=True)

	# Model parameters
	r = 0.01
	output_num = len(labels.classes_)

	# Define model
	class BaselineRNNEntityModel(nn.Module):
		def __init__(self, embeddings, output_num):
			super(BaselineRNNEntityModel,self).__init__()

			self.output_num = output_num

			self.initial_state = torch.zeros(self.output_num)

			self.word_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
			self.dense1 = nn.Linear(self.word_embeddings.embedding_dim + self.output_num, 64)
			self.output_dense = nn.Linear(64, self.output_num)

		def forward(self, idxs):
			''' Accepts PackedSequence of token indexes, representing a batch of sequences of tokens. '''

			idxs,lengths = rnn.pad_packed_sequence(idxs, batch_first=True)
			x = self.word_embeddings(idxs) # batch_size * sequence_length * num_features

			y = torch.empty(x.shape[0], x.shape[1], self.output_num, device=x.device)
			previous_output = self.initial_state.repeat(x.shape[0], 1).to(device=x.device)
			for i in range(x.shape[1]): # Over sequence_length dimension
				x_i = torch.cat((x[:,i,:], previous_output), dim=1)
				y_i = F.relu(self.dense1(x_i))
				y_i = self.output_dense(y_i)
				previous_output = y_i
				y[:,i,:] = y_i

			x = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
			return x

	# Create model
	model = BaselineRNNEntityModel(torch.from_numpy(embeddings.values), output_num).float().cuda()
	print(model)

	opt = optim.Adam(model.parameters(), lr=0.001)
	ignore_index = 999
	criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

	# Make directory to store summaries and results
	os.makedirs(args.name, exist_ok=True)

	#plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 100

	def collate_fn(sequences):
		xs,ys = zip(*sequences)
		return rnn.pack_sequence(xs,False).cuda(),rnn.pack_sequence(ys,False).cuda()

	# Setup training
	dataloader = torch.utils.data.DataLoader(
			train_dataset,
			collate_fn=collate_fn,
			batch_size=batch_size,
			shuffle=True
		)
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, collate_fn=collate_fn, batch_size=batch_size)
	history = defaultdict(list)

	stopwatch.tick('Setup training',report=True)

	history = train(model, dataloader, epochs, opt, criterion, dev_dataloader, patience=25, min_delta=0.001, verbose=1, padding_value=ignore_index)

	torch.save(model.state_dict(), os.path.join(args.name,args.name+'.pt'))
	joblib.dump(labels,os.path.join(args.name,args.name+'_labels.joblib'))
	pandas.DataFrame(history).to_csv(os.path.join(args.name,'logs.csv'))

	stopwatch.tick('Completed training',report=True)

	### TEST MODEL
	model.eval()

	test_subjects = ann_test[10:12]
	subjects_dataset = BaselineRNNEntityDataset(test_subjects)

	subject_x, subject_y = subjects_dataset.get_data()
	predict_subjects = model(subject_x.cuda())
	subject_preds = [F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_subjects)]

	print_len = max(len(c) for c in labels.classes_) + 2
	for subject,preds in zip(test_subjects,subject_preds):
		print(f'Lengths: {subject.labels.shape}, {preds.shape}')
		sub_labels = labels.inverse_transform(preds.argmax(1))
		for (token,label),pred,pred_lab in zip(subject,preds,sub_labels):
			#print(f'{token:20} {label:25} {predicted_labels[i]:25} [{", ".join(f"{p:.2f}" for p in predict_subjects[i])}]')
			vec_str = ', '.join(f'{p:.2f}' for p in pred)
			print(('{0:20} {1:'+str(print_len)+'} {2:'+str(print_len)+'} [{3}]').format(token,label,pred_lab,vec_str))
	print(f'[{", ".join(str(i) for i in labels.classes_)}]')

	test_x, test_y = test_dataset.get_data()
	predict_test = model(test_x.cuda())
	test_preds = numpy.vstack([F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_test)])
	print(test_preds.shape)
	predicted_test_labels = labels.inverse_transform(test_preds.argmax(1))

	val_target = numpy.hstack([i.cpu().detach().numpy() for i in unpack_sequence(test_y)])
	print(val_target.shape)
	true_test_labels = labels.inverse_transform(val_target)

	confusion = confusion_matrix(true_test_labels, predicted_test_labels, labels=labels.classes_)
	print(confusion)
	print(confusion.shape)
	p,r,f,s = precision_recall_fscore_support(true_test_labels, predicted_test_labels, labels=labels.classes_)

	print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
	for i,label in enumerate(labels.classes_):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

	stopwatch.tick('Finished evaluation',report=True)

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
