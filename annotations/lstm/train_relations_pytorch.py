if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.nn.utils.rnn as rnn
	import torch.optim as optim

	from gaml.utilities.torchutils import train,predict_from_dataloader,save_figs

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas
	from sklearn.model_selection import train_test_split

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	import os
	import itertools
	from collections import defaultdict

	#from gaml.annotations.bratutils import Standoff
	#from gaml.annotations.brattowindow import StandoffLabels
	from gaml.annotations.wordembeddings import WordEmbeddings
	from gaml.annotations.annmlutils import open_anns

	from sklearn.preprocessing import LabelEncoder,LabelBinarizer
	import sklearn.metrics
	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
	from sklearn.utils.class_weight import compute_class_weight
	from sklearn.externals import joblib

	parser = ArgumentParser(description='Train Pytorch ANN to predict relations in astrophysical text using an LSTM-based model.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('name',help='Model name to use when saving outputs.')
	args = parser.parse_args()

	torch.manual_seed(42)

	# Read in data
	types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Confidence','Measurement','Name','Property']
	#types = None
	anns = open_anns(args.ann,types=types,use_labelled=True)
	embeddings = WordEmbeddings.open(args.embeddings)

	stopwatch.tick('Opened all files',report=True)

	# Make test/train split
	# Training set for parameters, dev set for hyper-parameters, test set for evaluation metrics
	ann_train,ann_other = train_test_split(anns, test_size=0.4, random_state=42)
	ann_dev,ann_test = train_test_split(ann_other, test_size=0.5, random_state=42)

	window_pad = 5

	numpy.random.seed(42)
	padding_vector = numpy.random.normal(size=(embeddings.dim,))

	labels = LabelEncoder().fit(list(set(r.type for a in anns for r in a.relations))+['none'])
	entity_labels = LabelBinarizer().fit(list(set(e.type for a in anns for e in a.entities)))

	class LSTMRelationDataset(torch.utils.data.Dataset):
		def __init__(self, ann_list):
			self.anns = ann_list

			xs = []
			ys = []
			for a in self.anns:
				tokens,tags = zip(*a)
				document_matrix = numpy.vstack([padding_vector]*window_pad + [embeddings[t] for t in tokens] + [padding_vector]*window_pad)
				#for start,end in itertools.permutations(a.standoff.entities,2):
				#	rel = next((r.type for r in a.standoff.relations if r.arg1==start and r.arg2==end),None)
				for start,end in itertools.combinations(a.entities,2):
					rel = next((r.type for r in a.relations if set([start,end])==set([r.arg1,r.arg2])),'none')

					#print(f'Start: {start.text})')
					#print(f'End: {end.text}')
					start_s,start_e = a.get_token_idxs(start)
					end_s,end_e = a.get_token_idxs(end)
					#print(f'Start: {start.text}, ({start_s}, {start_e})')
					#print(f'End: {end.text}, ({end_s}, {end_e})')

					start_s,start_e = start_s+window_pad,start_e+window_pad
					end_s,end_e = end_s+window_pad,end_e+window_pad

					pre_window = document_matrix[(start_s-window_pad):start_s]
					start_tokens = document_matrix[start_s:start_e]
					between_tokens = numpy.vstack([padding_vector,document_matrix[start_e:end_s],padding_vector])
					end_tokens = document_matrix[end_s:end_e]
					post_window = document_matrix[end_e:(end_e+window_pad)]

					entity_encodings = entity_labels.transform([start.type,end.type]).reshape(-1)

					xs.append((pre_window,start_tokens,between_tokens,end_tokens,post_window,entity_encodings))
					ys.append(rel)

			#print('ys =',ys)
			self.y = labels.transform(ys)
			self.y_labels = ys
			self.x = xs

		def __len__(self):
			return len(self.x)

		def __getitem__(self, idx):
			if torch.is_tensor(idx):
				idx = idx.tolist()

			x,y = self.x[idx], self.y[idx]

			x,y = tuple(torch.from_numpy(i).float() for i in x), torch.tensor(y).long()

			return x,y

	train_dataset = LSTMRelationDataset(ann_train)
	dev_dataset = LSTMRelationDataset(ann_dev)
	test_dataset = LSTMRelationDataset(ann_test)

	stopwatch.tick(f'Constructed datasets: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} testing samples',report=True)

	# Model parameters
	output_num = len(labels.classes_)

	# Define model
	class LSTMRelationModel(nn.Module):
		def __init__(self, embedding_dim, entity_classes_num, hidden, output_num):
			super(LSTMRelationModel,self).__init__()
			self.pool = nn.LSTM(embedding_dim, 64, bidirectional=True, batch_first=True)
			self.dense1 = nn.Linear(64*2*5 + entity_classes_num*2, hidden)
			self.output_dense = nn.Linear(hidden, output_num)

		def forward(self, x):

			#x = tuple(F.relu(self.pool(i)[1][0]) for i in x[:-1])+(x[-1],)
			x = tuple(F.relu(self.pool(i)[1][0].permute(1,2,0).reshape(-1, 64*2)) for i in x[:-1])+(x[-1],)
			x = torch.cat(x,1)

			x = F.relu(self.dense1(x))
			x = self.output_dense(x)
			return x

	# Create model
	model = LSTMRelationModel(embeddings.dim, len(entity_labels.classes_), 32, output_num).float().cuda()
	print(model)

	opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
	## Weight classes, sum samples and take inverse weight
	#criterion = nn.CrossEntropyLoss()
	class_weights = compute_class_weight('balanced',labels.classes_,train_dataset.y_labels)
	criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().cuda())

	# Make directory to store summaries and results
	os.makedirs(args.name, exist_ok=True)

	#plot_model(model, to_file=os.path.join(model.name,model.name+'.png'), show_shapes=True)

	# Training parameters
	batch_size = 128
	epochs = 30

	def make_metric(func):
		def metric(output, target):
			output = output.cpu().detach().numpy().argmax(1)
			target = target.cpu().detach().numpy()
			return func(output, target)
		return metric
	average = 'macro'
	f1_score = make_metric(lambda o,t: sklearn.metrics.f1_score(t,o,average=average))
	precision_score = make_metric(lambda o,t: sklearn.metrics.precision_score(t,o,average=average))
	recall_score = make_metric(lambda o,t: sklearn.metrics.recall_score(t,o,average=average))

	# Setup training
	def collate_fn(sequences):
		xs,ys = zip(*sequences)
		unpack_xs = tuple(zip(*xs))
		return tuple(rnn.pack_sequence(i,False).cuda() for i in unpack_xs[:-1])+(torch.stack(unpack_xs[-1],0).cuda(),), torch.stack(ys,0).cuda()

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

	history = train(model, dataloader, epochs, opt, criterion, dev_dataloader, metrics={'f1':f1_score,'precision':precision_score,'recall':recall_score}, patience=25, min_delta=0.0001, verbose=1)

	torch.save(model.state_dict(), os.path.join(args.name,args.name+'.pt'))
	joblib.dump(labels,os.path.join(args.name,args.name+'_labels.joblib'))
	pandas.DataFrame(history).to_csv(os.path.join(args.name,'logs.csv'))

	stopwatch.tick('Completed training',report=True)

	### TEST MODEL
	model.eval()

	test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size)
	predict_test = predict_from_dataloader(model, test_dataloader)
	predicted_labels = labels.inverse_transform(predict_test.argmax(1))

	confusion = confusion_matrix(test_dataset.y_labels, predicted_labels, labels=labels.classes_)
	print(confusion)
	print(confusion.shape)
	p,r,f,s = precision_recall_fscore_support(test_dataset.y_labels, predicted_labels, labels=labels.classes_)

	print_len = max(len(c) for c in labels.classes_) + 2
	print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
	for i,label in enumerate(labels.classes_):
		print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

	pandas.DataFrame([p,r,f,s],columns=labels.classes_,index=['Precision','Recall','F1','Support']).T.to_csv(os.path.join(args.name,'test_metrics.csv'))

	stopwatch.tick('Finished evaluation',report=True)

	save_figs(history, args.name)

	stopwatch.report()
