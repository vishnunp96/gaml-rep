if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.nn.utils.rnn as rnn

	from gaml.utilities.torchutils import unpack_sequence

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas
	from collections import defaultdict
	from sklearn.model_selection import train_test_split

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction
	import os
	import itertools

	from gaml.annotations.bratnormalisation import open_clean
	from gaml.annotations.brattowindow import StandoffLabels

	from sklearn.preprocessing import LabelEncoder
	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
	from sklearn.externals import joblib

	#import transformers
	from transformers import BertTokenizer,BertModel,AdamW

	parser = ArgumentParser(description='Train PyTorch/transformers bert model to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	args = parser.parse_args()

	device = torch.device('cuda')
	#device = torch.device('cpu')

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

	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

	# Make test/train split
	ann_train,ann_test = train_test_split(anns, test_size=0.1, random_state=42)

	labels = LabelEncoder().fit(list(set(l for a in anns for t,l in a)))

	class BertEntityDataset(torch.utils.data.Dataset):
		def __init__(self, ann_list):
			self.anns = ann_list

			xs = []
			ys = []
			all_idxs = []
			subwords = []
			for a in ann_list:
				x = []
				y = []
				idxs = []
				subw = []
				for i,(token,tag) in enumerate(a):
					split_token = tokenizer.tokenize(token)
					#token_ids = tokenizer.encode(token,add_special_tokens=True)
					token_ids = tokenizer.convert_tokens_to_ids(split_token)
					x.extend(token_ids)
					y.extend(itertools.repeat(tag,len(token_ids)))
					idxs.extend(itertools.repeat(i,len(token_ids)))
					subw.extend(split_token)
				if len(x)<512:
					xs.append(numpy.array(x))
					ys.append(labels.transform(y))
					all_idxs.append(numpy.array(idxs))
					subwords.append(subw)

			self.x = xs
			self.y = ys
			self.idxs = all_idxs
			self.subwords = subwords

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
			idxs = rnn.pack_sequence([torch.from_numpy(i).long() for i in self.idxs],False)
			return xs,ys,idxs

	train_dataset = BertEntityDataset(ann_train)
	test_dataset = BertEntityDataset(ann_test)

	# Model parameters
	r = 0.01
	output_num = len(labels.classes_)

	# Define model
	class BertEntityModel(nn.Module):
		def __init__(self, pretrained_model_name_or_path, output_num):
			super(BertEntityModel,self).__init__()

			self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
			embedding_dim = self.bert.embeddings.word_embeddings.embedding_dim
			#self.dropout = nn.Dropout() ## Not helping
			self.output_dense = nn.Linear(embedding_dim,output_num)

		def forward(self, idxs_in):
			''' Accepts PackedSequence of token indexes. '''

			idxs,lengths = rnn.pad_packed_sequence(idxs_in, batch_first=True)

			#mask = (torch.arange(idxs.shape[1]).expand(idxs.shape[0], idxs.shape[1]) < lengths.unsqueeze(1)).float()
			#mask = mask.to(device=idxs.device)

			#emb = self.bert(idxs,mask)[0]
			x = self.bert(idxs)[0]
			#x = self.dropout(x)
			x = self.output_dense(x)

			x = rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
			return x

	# Create model
	model = BertEntityModel('bert-base-cased', output_num).cuda()
	print(model)

	#opt = torch.optim.Adam(model.parameters(), lr=0.001)
	opt = AdamW(model.parameters(), lr=0.001)
	#opt = AdamW(model.output_dense.parameters(), lr=0.001)
	ignore_index = 999
	criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

	# Make directory to store summaries and results
	os.makedirs(args.name, exist_ok=True)

	# Training parameters
	batch_size = 5
	epochs = 500

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

	#callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, baseline=0.5, verbose=1, patience=25, restore_best_weights=True)]

	# Train model
	for epoch in range(epochs):
		print(f'Epoch {epoch}')
		epoch_loss = 0.0
		epoch_samples = 0
		for batch_i,(batch_x,batch_y) in enumerate(dataloader):
			#print(f'Batch: {batch_i}')
			#batch_y = batch_y.long()
			opt.zero_grad()
			output = model(batch_x.cuda())
			output,_ = rnn.pad_packed_sequence(output,batch_first=True)
			output = output.view(-1,output_num)
			target,_ = rnn.pad_packed_sequence(batch_y,padding_value=ignore_index,batch_first=True)
			target = target.view(-1)
			loss = criterion(output, target.cuda())
			loss.backward()
			opt.step()
			epoch_loss += output.shape[0] * loss.item()
			epoch_samples += output.shape[0]
			del loss
			del output
		epoch_loss = epoch_loss/epoch_samples
		with torch.no_grad():
			model.eval()
			val_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size)
			val_loss = 0.0
			val_samples = 0
			for val_x,val_y in val_dataloader:
				val_output = model(val_x.cuda())
				val_output = rnn.pad_packed_sequence(val_output)[0].view(-1,output_num)
				val_target = rnn.pad_packed_sequence(val_y,padding_value=ignore_index)[0].view(-1)
				val_loss += val_output.shape[0] * criterion(val_output, val_target.cuda()).item()
				val_samples += val_output.shape[0]
			val_loss = val_loss/val_samples
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
	subjects_dataset = BertEntityDataset(test_subjects)

	subject_x, subject_y, subject_idxs = subjects_dataset.get_data()
	predict_subjects = model(subject_x.cuda()).cpu()
	subject_preds = [F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_subjects)]

	print_len = max(len(c) for c in labels.classes_) + 2
	for subwords,true_label_idxs,preds in zip(subjects_dataset.subwords, subjects_dataset.y, subject_preds):
		print(f'Shapes: {preds.shape}')
		sub_labels = labels.inverse_transform(preds.argmax(1))
		true_labels = labels.inverse_transform(true_label_idxs)
		for subtoken,label,pred,pred_lab in zip(subwords,true_labels,preds,sub_labels):
			vec_str = ', '.join(f'{p:.2f}' for p in pred)
			print(('{0:20} {1:'+str(print_len)+'} {2:'+str(print_len)+'} [{3}]').format(subtoken,label,pred_lab,vec_str))
	print(f'[{", ".join(str(i) for i in labels.classes_)}]')


	test_x, test_y, test_idxs = test_dataset.get_data()

	with torch.no_grad():
		test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1)
		predict_test = []
		for test_x_batch,_ in test_dataloader:
			test_output = unpack_sequence(model(test_x_batch.cuda()).cpu())
			predict_test.extend(test_output)

	test_preds = numpy.vstack([F.softmax(i,dim=1).cpu().detach().numpy() for i in predict_test])
	print(test_preds.shape)
	predicted_test_labels = labels.inverse_transform(test_preds.argmax(1))

	test_target = numpy.hstack([i.cpu().detach().numpy() for i in unpack_sequence(test_y)])
	print(test_target.shape)
	true_test_labels = labels.inverse_transform(test_target)

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
