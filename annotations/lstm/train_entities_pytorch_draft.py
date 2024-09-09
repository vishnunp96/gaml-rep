if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	import torch.nn.utils.rnn as rnn

	from gaml.utilities.torchutils import unpack_sequence

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas
	from sklearn.model_selection import train_test_split
	from collections import defaultdict

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os

	from gaml.annotations.bratnormalisation import open_clean
	from gaml.annotations.brattowindow import StandoffLabels
	from gaml.annotations.wordembeddings import WordEmbeddings

	from sklearn.preprocessing import LabelEncoder
	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
	from sklearn.externals import joblib

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	args = parser.parse_args()

	# Read in data
	types = ['MeasuredValue','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName']
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

	# Make test/train split
	ann_train,ann_test = train_test_split(anns, test_size=0.1, random_state=42)

	labels = LabelEncoder().fit(list(set(l for a in anns for t,l in a)))

	class LSTMEntityDataset(torch.utils.data.Dataset):
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

	train_dataset = LSTMEntityDataset(ann_train)
	test_dataset = LSTMEntityDataset(ann_test)

	# Model parameters
	r = 0.01
	output_num = len(labels.classes_)

	# Define model
	class LSTMEntityModel(nn.Module):
		def __init__(self, embeddings, output_num):
			super(LSTMEntityModel,self).__init__()

			self.word_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
			self.lstm1 = nn.LSTM(self.word_embeddings.embedding_dim, 128, bidirectional=True)
			self.lstm2 = nn.LSTM(128*2, 32, bidirectional=True)

			self.output_dense = nn.Linear(32*2,output_num)

		def forward(self, idxs):
			''' Accepts PackedSequence of token indexes. '''

			idxs,lengths = rnn.pad_packed_sequence(idxs)
			emb = self.word_embeddings(idxs)
			x = rnn.pack_padded_sequence(emb, lengths, enforce_sorted=False)
			x,_ = self.lstm1(x)
			x,_ = self.lstm2(x)

			x,lengths = rnn.pad_packed_sequence(x)
			x = self.output_dense(x)
			x = rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
			return x

	# Create model
	model = LSTMEntityModel(torch.from_numpy(embeddings.values), output_num).float().cuda()
	print(model)

	opt = optim.Adam(model.parameters(), lr=0.001)
	ignore_index = 999
	criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

	# Make directory to store summaries and results
	os.makedirs(args.name, exist_ok=True)

	#plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 150

	def collate_fn(sequences):
		xs,ys = zip(*sequences)
		return rnn.pack_sequence(xs,False),rnn.pack_sequence(ys,False)

	# Setup training
	dataloader = torch.utils.data.DataLoader(
			train_dataset,
			collate_fn=collate_fn,
			batch_size=batch_size,
			shuffle=True
		)
	history = defaultdict(list)

	# Train model
	val_x,val_y = test_dataset.get_data()
	for epoch in range(epochs):
		print(f'Epoch {epoch}')
		epoch_loss = 0.0
		epoch_samples = 0
		for batch_i,(batch_x,batch_y) in enumerate(dataloader):
			#print(f'Batch: {batch_i}')
			#batch_y = batch_y.long()
			opt.zero_grad()
			output = model(batch_x.cuda())
			output,_ = rnn.pad_packed_sequence(output)
			output = output.view(-1,output_num)
			target,_ = rnn.pad_packed_sequence(batch_y,padding_value=ignore_index)
			target = target.view(-1)
			loss = criterion(output, target.cuda())
			loss.backward()
			opt.step()
			epoch_loss += output.shape[0] * loss.item()
			epoch_samples += output.shape[0]
		epoch_loss = epoch_loss/epoch_samples
		with torch.no_grad():
			model.eval()
			val_output = model(val_x.cuda())
			val_output = rnn.pad_packed_sequence(val_output)[0].view(-1,output_num)
			val_target = rnn.pad_packed_sequence(val_y,padding_value=ignore_index)[0].view(-1)
			val_loss = criterion(val_output, val_target.cuda()).item()
			model.train()
		history['loss'].append(epoch_loss)
		history['val_loss'].append(val_loss)

		print(f'Epoch loss: {epoch_loss:.4f}, Validation loss: {val_loss:.4f}')

	# Save model
	torch.save(model.state_dict(), os.path.join(args.name,args.name+'.pt'))
	joblib.dump(labels,os.path.join(args.name,args.name+'_labels.joblib'))

	pandas.DataFrame(history).to_csv(os.path.join(args.name,'logs.csv'))


	### TEST MODEL
	model.eval()

	test_subjects = ann_test[10:12]
	subjects_dataset = LSTMEntityDataset(test_subjects)

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
