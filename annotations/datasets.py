import itertools

import numpy
import torch
import torch.nn.utils.rnn as rnn

from annotations.brattowindow import StandoffLabels

class AnnotationDataset(torch.utils.data.Dataset):
	def __init__(self, anns, cuda=False):
		super(AnnotationDataset,self).__init__()
		self.anns = anns
		self.cuda = cuda

	def __len__(self):
		''' Return length of dataset - note, this is number of data points, not number of documents. '''
		raise NotImplementedError

	def __getitem__(self, idx):
		''' Get the ith datapoint from this dataset, and return as (x,y) tuple of tensors.'''
		raise NotImplementedError

	def collate_fn(self, sequences):
		''' Convert a list of (x,y) datapoint tuples into a single (x,y) tuple of tensors. '''
		raise NotImplementedError

	def get_data(self):
		''' Get all data and return as an (x,y) tuple of tensors. '''
		return self.collate_fn([self[i] for i in range(len(self))])

	def dataloader(self, batch_size=1, shuffle=False):
		''' Produce a dataloader instance based on this dataset. '''
		return torch.utils.data.DataLoader(
				self,
				collate_fn = self.collate_fn,
				batch_size = batch_size,
				shuffle = shuffle
			)


class EntityEmbeddingsDataset(AnnotationDataset):
	def __init__(self, ann_list, embeddings, labels, window_pad, cuda=False):
		super(EntityEmbeddingsDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		for a in self.anns:
			tokens, tags = a.tokens, a.labels
			document_matrix = numpy.vstack([embeddings.padding]*window_pad + [embeddings[t] for t in tokens] + [embeddings.padding]*window_pad)

			x = []
			for i in range(len(tokens)):
				j = i + window_pad
				window = document_matrix[(j-window_pad):(j+window_pad+1)].flatten()
				x.append(window)

			xs.append(numpy.vstack(x))
			ys.append(labels.transform(tags))

		self.x = xs
		self.y = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.x[idx], self.y[idx]
		return torch.from_numpy(x).float(), torch.from_numpy(y).long()

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		return rnn.pack_sequence(xs,False).cuda(),rnn.pack_sequence(ys,False).cuda()


class EntityIndexesDataset(AnnotationDataset):
	def __init__(self, ann_list, token_indexes, fallback_index, labels, cuda=False):
		super(EntityIndexesDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		for a in self.anns:
			tokens, tags = a.tokens, a.labels
			xs.append(numpy.array([token_indexes.get(t,fallback_index) for t in tokens]))
			ys.append(labels.transform(tags))

		self.x = xs
		self.y = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.x[idx], self.y[idx]
		return torch.from_numpy(x).long(), torch.from_numpy(y).long()

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		xs,ys = rnn.pack_sequence(xs,False),rnn.pack_sequence(ys,False)
		if self.cuda:
			return xs.cuda(), ys.cuda()
		else:
			return xs, ys


class EntityIndexesCharsDataset(AnnotationDataset):
	def __init__(self, ann_list, token_indexes, char_indexes, fallback_index, labels, cuda=False):
		super(EntityIndexesCharsDataset,self).__init__(ann_list, cuda)

		ts = [] # tokens
		cs = [] # characters
		ys = []
		for a in self.anns:
			tokens, tags = a.tokens, a.labels
			ts.append(numpy.array([token_indexes.get(t,fallback_index) for t in tokens]))
			cs.append([numpy.array([char_indexes.get(c,char_indexes[None]) for c in t]) for t in tokens])
			ys.append(labels.transform(tags))

		self.t = ts
		self.c = cs
		self.y = ys

	def __len__(self):
		return len(self.t)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		t,c,y = self.t[idx], self.c[idx], self.y[idx]
		return ((torch.from_numpy(t).long(),
					rnn.pack_sequence([torch.from_numpy(i).long() for i in c],False)),
				torch.from_numpy(y).long())

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		ts,cs = tuple(zip(*xs))
		if self.cuda:
			return (rnn.pack_sequence(ts,False).cuda(), [i.cuda() for i in cs]), rnn.pack_sequence(ys,False).cuda()
		else:
			return (rnn.pack_sequence(ts,False), [i for i in cs]), rnn.pack_sequence(ys,False)



class RelationSplitSpansDirectedLabelledIndexesDataset(AnnotationDataset):
	def __init__(self, ann_list, token_indexes, fallback_index, labels, entity_labels, allowed_relations, window_pad, cuda=False):
		super(RelationSplitSpansDirectedLabelledIndexesDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		self.origins = []

		outside_label = entity_labels.transform([StandoffLabels.outside_label])
		for a in self.anns:
			stripped_labels = [l.split('_')[0] for l in a.labels]
			tokens,elabels = a.tokens, entity_labels.transform(stripped_labels)
			idxs = numpy.array(([0]*window_pad) + [token_indexes.get(t,fallback_index) for t in tokens] + ([0]*window_pad))
			label_padding = entity_labels.transform([StandoffLabels.outside_label]*window_pad)
			elabels = numpy.concatenate((label_padding, elabels, label_padding))
			for start,end in itertools.permutations(a.entities,2):
				if (start.type,end.type) in allowed_relations:
					rel = next((r.type for r in a.relations if r.arg1==start and r.arg2==end),'none')

					self.origins.append((a, start, end))

					if start.start > end.start: # This relation goes "backwards" in the text
						start,end = end,start # Reverse start and end tokens such that start occurs before end
						reverse = 1
					else:
						reverse = 0

					#print(f'Relation: {rel}')
					#print(repr(start))
					#print(repr(end))

					start_s,start_e = a.get_token_idxs(start)
					end_s,end_e = a.get_token_idxs(end)

					start_s,start_e = start_s+window_pad,start_e+window_pad
					end_s,end_e = end_s+window_pad,end_e+window_pad

					pre_window = idxs[(start_s-window_pad):start_s]
					pre_window_labels = elabels[(start_s-window_pad):start_s]
					start_tokens = idxs[start_s:start_e]
					start_tokens_labels = elabels[start_s:start_e]
					between_tokens = numpy.concatenate(([0], idxs[start_e:end_s], [0]))
					between_tokens_labels = numpy.concatenate((outside_label, elabels[start_e:end_s], outside_label))
					end_tokens = idxs[end_s:end_e]
					end_tokens_labels = elabels[end_s:end_e]
					post_window = idxs[end_e:(end_e+window_pad)]
					post_window_labels = elabels[end_e:(end_e+window_pad)]

					entity_encodings = entity_labels.transform([start.type,end.type]).reshape(-1)

					xs.append(
							(
								(pre_window, pre_window_labels),
								(start_tokens, start_tokens_labels),
								(between_tokens, between_tokens_labels),
								(end_tokens, end_tokens_labels),
								(post_window, post_window_labels),
								entity_encodings,
								reverse
							)
						)
					ys.append(rel)

		self.x = xs
		self.y = labels.transform(ys)
		self.y_labels = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x,y = self.x[idx], self.y[idx]

		x,y = tuple((torch.from_numpy(ts).long(),torch.from_numpy(ls).long()) for ts,ls in x[:-2]) + (torch.from_numpy(x[-2]).float(), torch.tensor([x[-1]]).float()), torch.tensor(y).long()

		return x,y

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		unpack_xs = tuple(zip(*xs))
		unpacked_labels = [tuple(zip(*u)) for u in unpack_xs[:-2]]
		if self.cuda:
			return tuple((rnn.pack_sequence(ts,False).cuda(), rnn.pack_sequence(ls,False).cuda()) for ts,ls in unpacked_labels) + (torch.stack(unpack_xs[-2],0).cuda(),torch.stack(unpack_xs[-1],0).cuda()), torch.stack(ys,0).cuda()
		else:
			return tuple((rnn.pack_sequence(ts,False), rnn.pack_sequence(ls,False)) for ts,ls in unpacked_labels)+(torch.stack(unpack_xs[-2],0),torch.stack(unpack_xs[-1],0)), torch.stack(ys,0)


class RelationSplitSpansDirectedIndexesDataset(AnnotationDataset):
	def __init__(self, ann_list, token_indexes, fallback_index, labels, entity_labels, allowed_relations, window_pad, cuda=False):
		super(RelationSplitSpansDirectedIndexesDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		self.origins = []
		for a in self.anns:
			tokens = a.tokens ## Should probably find a way to include entity labels for each token?
			idxs = numpy.array(([0]*window_pad) + [token_indexes.get(t,fallback_index) for t in tokens] + ([0]*window_pad))
			for start,end in itertools.permutations(a.entities,2):
				if (start.type,end.type) in allowed_relations:
					rel = next((r.type for r in a.relations if r.arg1==start and r.arg2==end),'none')

					self.origins.append((a, start, end))

					if start.start > end.start: # This relation goes "backwards" in the text
						start,end = end,start # Reverse start and end tokens such that start occurs before end
						reverse = 1
					else:
						reverse = 0

					#print(f'Relation: {rel}')
					#print(repr(start))
					#print(repr(end))

					start_s,start_e = a.get_token_idxs(start)
					end_s,end_e = a.get_token_idxs(end)

					start_s,start_e = start_s+window_pad,start_e+window_pad
					end_s,end_e = end_s+window_pad,end_e+window_pad

					pre_window = idxs[(start_s-window_pad):start_s]
					start_tokens = idxs[start_s:start_e]
					between_tokens = numpy.concatenate(([0], idxs[start_e:end_s], [0]))
					end_tokens = idxs[end_s:end_e]
					post_window = idxs[end_e:(end_e+window_pad)]

					entity_encodings = entity_labels.transform([start.type,end.type]).reshape(-1)

					xs.append((pre_window,start_tokens,between_tokens,end_tokens,post_window,entity_encodings,reverse))
					ys.append(rel)

		self.x = xs
		self.y = labels.transform(ys)
		self.y_labels = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x,y = self.x[idx], self.y[idx]

		x,y = tuple(torch.from_numpy(i).long() for i in x[:-2]) + (torch.from_numpy(x[-2]).float(), torch.tensor([x[-1]]).float()), torch.tensor(y).long()

		return x,y

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		unpack_xs = tuple(zip(*xs))
		if self.cuda:
			return tuple(rnn.pack_sequence(i,False).cuda() for i in unpack_xs[:-2])+(torch.stack(unpack_xs[-2],0).cuda(),torch.stack(unpack_xs[-1],0).cuda()), torch.stack(ys,0).cuda()
		else:
			return tuple(rnn.pack_sequence(i,False) for i in unpack_xs[:-2])+(torch.stack(unpack_xs[-2],0),torch.stack(unpack_xs[-1],0)), torch.stack(ys,0)


class RelationSplitSpansIndexesDataset(AnnotationDataset):
	def __init__(self, ann_list, token_indexes, fallback_index, labels, entity_labels, allowed_relations, window_pad, cuda=False):
		super(RelationSplitSpansIndexesDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		self.origins = []
		for a in self.anns:
			tokens = a.tokens ## Should probably find a way to include entity labels for each token?
			idxs = numpy.array(([0]*window_pad) + [token_indexes.get(t,fallback_index) for t in tokens] + ([0]*window_pad))
			#for start,end in itertools.permutations(a.entities,2):
			#	if (start.type,end.type) in allowed_relations or (end.type,start.type) in allowed_relations:
			#		rel = next((r.type for r in a.relations if r.arg1==start and r.arg2==end),'none')
			for start,end in itertools.combinations(a.entities,2):
				if (start.type,end.type) in allowed_relations or (end.type,start.type) in allowed_relations:
					rel = next((r.type for r in a.relations if set([start,end])==set([r.arg1,r.arg2])),'none')

					start_s,start_e = a.get_token_idxs(start)
					end_s,end_e = a.get_token_idxs(end)

					start_s,start_e = start_s+window_pad,start_e+window_pad
					end_s,end_e = end_s+window_pad,end_e+window_pad

					pre_window = idxs[(start_s-window_pad):start_s]
					start_tokens = idxs[start_s:start_e]
					between_tokens = numpy.concatenate(([0], idxs[start_e:end_s], [0]))
					end_tokens = idxs[end_s:end_e]
					post_window = idxs[end_e:(end_e+window_pad)]

					entity_encodings = entity_labels.transform([start.type,end.type]).reshape(-1)

					xs.append((pre_window,start_tokens,between_tokens,end_tokens,post_window,entity_encodings))
					ys.append(rel)
					self.origins.append((a, start, end))

		self.x = xs
		self.y = labels.transform(ys)
		self.y_labels = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x,y = self.x[idx], self.y[idx]

		x,y = tuple(torch.from_numpy(i).long() for i in x[:-1]) + (torch.from_numpy(x[-1]).float(),), torch.tensor(y).long()

		return x,y

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		unpack_xs = tuple(zip(*xs))
		if self.cuda:
			return tuple(rnn.pack_sequence(i,False).cuda() for i in unpack_xs[:-1])+(torch.stack(unpack_xs[-1],0).cuda(),), torch.stack(ys,0).cuda()
		else:
			return tuple(rnn.pack_sequence(i,False) for i in unpack_xs[:-1])+(torch.stack(unpack_xs[-1],0),), torch.stack(ys,0)


class RelationSplitSpansDataset(AnnotationDataset):
	def __init__(self, ann_list, embeddings, labels, entity_labels, window_pad, cuda=False):
		super(RelationSplitSpansDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		for a in self.anns:
			tokens,tags = zip(*a)
			document_matrix = numpy.vstack([embeddings.padding]*window_pad + [embeddings[t] for t in tokens] + [embeddings.padding]*window_pad)
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
				between_tokens = numpy.vstack([embeddings.padding,document_matrix[start_e:end_s],embeddings.padding])
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

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		unpack_xs = tuple(zip(*xs))
		return tuple(rnn.pack_sequence(i,False).cuda() for i in unpack_xs[:-1])+(torch.stack(unpack_xs[-1],0).cuda(),), torch.stack(ys,0).cuda()


class AttributeSplitSpansIndexesDataset(AnnotationDataset):
	def __init__(self, ann_list, token_indexes, fallback_index, labels, entity_labels, window_pad, cuda=False):
		super(AttributeSplitSpansIndexesDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		self.origins = []
		for ann in self.anns:
			idxs = numpy.array(([0]*window_pad) + [token_indexes.get(t,fallback_index) for t in ann.tokens] + ([0]*window_pad))
			for ent in (e for e in ann.entities if e.type in entity_labels.classes_):
				att = next((att.type for att in ann.attributes if att.subject==ent and att.type in labels.classes_),'none')

				start,end = ann.get_token_idxs(ent)

				start,end = start+window_pad,end+window_pad # To account for pre-padding in idxs array

				pre_window = idxs[(start-window_pad):start]
				ent_tokens = idxs[start:end]
				post_window = idxs[end:(end+window_pad)]

				entity_encoding = entity_labels.transform([ent.type]).reshape(-1)

				xs.append((pre_window,ent_tokens,post_window,entity_encoding))
				ys.append(att)
				self.origins.append((ann, ent))

		self.x = xs
		self.y = labels.transform(ys)
		self.y_labels = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x,y = self.x[idx], self.y[idx]

		x,y = tuple(torch.from_numpy(i).long() for i in x[:-1]) + (torch.from_numpy(x[-1]).float(),), torch.tensor(y).long()

		return x,y

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		unpack_xs = tuple(zip(*xs))
		if self.cuda:
			return tuple(rnn.pack_sequence(i,False).cuda() for i in unpack_xs[:-1])+(torch.stack(unpack_xs[-1],0).cuda(),), torch.stack(ys,0).cuda()
		else:
			return tuple(rnn.pack_sequence(i,False) for i in unpack_xs[:-1])+(torch.stack(unpack_xs[-1],0),), torch.stack(ys,0)


class AttributeSplitSpansIndexesEncodingsDataset(AnnotationDataset):
	def __init__(self, ann_list, token_indexes, fallback_index, labels, entity_tag_labels, allowed_subject_entities, window_pad, cuda=False):
		super(AttributeSplitSpansIndexesEncodingsDataset,self).__init__(ann_list, cuda)

		xs = []
		ys = []
		self.origins = []
		for ann in self.anns:
			idxs = numpy.array(([0]*window_pad) + [token_indexes.get(t,fallback_index) for t in ann.tokens] + ([0]*window_pad))
			ent_labels = ([StandoffLabels.outside_label]*window_pad) + [l for l in ann.labels] + ([StandoffLabels.outside_label]*window_pad)
			for ent in (e for e in ann.entities if e.type in allowed_subject_entities):
				att = next((att.type for att in ann.attributes if att.subject==ent and att.type in labels.classes_),'none')

				start,end = ann.get_token_idxs(ent)

				start,end = start+window_pad,end+window_pad # To account for pre-padding in idxs array

				pre_window = idxs[(start-window_pad):start]
				pre_window_labels = entity_tag_labels.transform(ent_labels[(start-window_pad):start])
				ent_tokens = idxs[start:end]
				ent_token_labels = entity_tag_labels.transform(ent_labels[start:end])
				post_window = idxs[end:(end+window_pad)]
				post_window_labels = entity_tag_labels.transform(ent_labels[end:(end+window_pad)])

				xs.append(((pre_window, ent_tokens, post_window), (pre_window_labels, ent_token_labels, post_window_labels)))
				ys.append(att)
				self.origins.append((ann, ent))

		self.x = xs
		self.y = labels.transform(ys) if len(ys)>0 else numpy.array([])
		self.y_labels = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x,y = self.x[idx], self.y[idx]

		ts,ls = x
		ts = tuple(torch.from_numpy(i).long() for i in ts)
		ls = tuple(torch.from_numpy(j).float() for j in ls)

		y = torch.tensor(y).long()

		return (ts,ls),y

	def collate_fn(self, sequences):
		# len(sequences) == batch_size
		# Sequences: List of tuples [(((ts),(ls)),y),...] where ts (num_tokens,) and ls (nun_tokens,ent_encoding)
		xs,ys = zip(*sequences)
		# xs: [((ts),(ls)),...], ys: [y1...yn]
		ts,ls = zip(*xs)
		unpack_ts = tuple(zip(*ts))
		unpack_ls = tuple(zip(*ls))
		# ts: []
		ts = tuple(rnn.pack_sequence(i,False) for i in unpack_ts)
		ls = tuple(rnn.pack_sequence(j,False) for j in unpack_ls)
		ys = torch.stack(ys,0)
		if self.cuda:
			ts = tuple(i.cuda() for i in ts)
			ls = tuple(j.cuda() for j in ls)
			ys = ys.cuda()
		return (ts,ls),ys
		# End with: ((torch.PackedSequence(),...),(torch.PackedSequence()...)), torch.tensor(batch_size, y_size)


class CompositeDataset(AnnotationDataset):
	def __init__(self, ann_list, datasets, cuda=False):
		super(CompositeDataset,self).__init__(ann_list, cuda)
		self.datasets = list(datasets)

		assert len(self.datasets) > 0
		assert len(set([len(d) for d in self.datasets])) == 1
		assert all(d.cuda==self.cuda for d in datasets)

		ys = []

		for i in range(len(self.datasets[0])):
			yi = [d[i][1] for d in self.datasets] # Get all y values for this index from datasets
			assert all(torch.all(torch.eq(y,yi[0])) for y in yi)
			ys.append(next(iter(yi)))
			#ys.append(self.datasets[0][i][1])

		self.y = ys

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		''' Get the ith datapoint from this dataset, and return as (x,y) tuple of tensors.'''
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = tuple(d[idx][0] for d in self.datasets), self.y[idx]
		# x # tuple(num_datasets)
		# y # torch.tensor((n_classes,))
		return x,y

	def collate_fn(self, sequences):
		''' Convert a list of (x,y) datapoint tuples into a single (x,y) tuple of tensors. '''
		xs,ys = zip(*sequences) # List of tuples, list of tensors
		# xs: [(x0,x1,...),...], ys: [y1...yn]
		xs = tuple(self.datasets[i].collate_fn([(xis[i],yi) for xis,yi in sequences])[0] for i in range(len(self.datasets)))
		ys = rnn.pack_sequence(ys,False)
		if self.cuda:
			return xs, ys.cuda() # xs already dealt with by sub-datasets
		else:
			return xs, ys
