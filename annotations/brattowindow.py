import numpy
import re

from collections import defaultdict

from gaml.annotations.bratutils import Standoff

def token_iter(text):
	current_index = 0
	for match in re.finditer(r'\s+|$', text):
		start,end = current_index,match.start()
		token = text[start:end]
		if token:
			yield token,start,end
		current_index = match.end()

class StandoffLabels:

	outside_label = 'outside'

	def get_labels(entity_types, include_bio=True):
		labels = [StandoffLabels.outside_label]
		for tag in entity_types:
			labels.append(tag+('_inside' if include_bio else ''))
			labels.append(tag+('_begin' if include_bio else ''))
		return sorted(set(labels))

	def retrieve_types(labels):
		types = []
		for tag in labels:
			if tag.endswith('_inside'):
				types.append(tag[:-7])
			elif tag.endswith('_begin'):
				types.append(tag[:-6])
			else:
				types.append(tag)
		return sorted(set(types))

	def __init__(self,standoff,types=None,include_bio=True):
		self.standoff = standoff

		self.entities = sorted((i for i in self.standoff.entities if (not types or i.type in types)),key=lambda e: e.start)
		self.relations = sorted((i for i in self.standoff.relations if (not types or i.type in types) and i.arg1 in self.entities and i.arg2 in self.entities),key=lambda r: (r.arg1.start,r.arg2.start))
		self.attributes = sorted((i for i in self.standoff.attributes if (not types or i.type in types) and i.subject in self.entities),key=lambda a: a.subject.start)
		self.notes = sorted((i for i in self.standoff.notes if i.subject in self.entities),key=lambda n: n.subject.start)

		iter_ents = iter(self.entities)

		try:
			current_ent = next(iter_ents)
		except StopIteration:
			current_ent = None
		inside_ent = False

		#embeddings = []
		labels = []
		tokens = []
		self.token_idxs = []
		self.token_entities = []
		self.token_entity_map = defaultdict(list)

		for token,start,end in token_iter(standoff.text):
			#print('Token:',token)
			tokens.append(token)
			self.token_idxs.append((start,end))

			if current_ent is not None and current_ent.start < end: # This token overlaps current entity
				tag = current_ent.type
				#print('Tag:',tag)

				if inside_ent:
					labels.append(tag+('_inside' if include_bio else ''))
				else:
					labels.append(tag+('_begin' if include_bio else ''))
					inside_ent = True

				self.token_entity_map[current_ent].append(len(self.token_entities))
				#print(f'Append to map: {len(self.token_entities)} ({self.token_entity_map[current_ent]}, "{" ".join(tokens[i] for i in self.token_entity_map[current_ent])}")')
				self.token_entities.append(current_ent)

				at_end = end >= current_ent.end # This token is at the end of the current entity

				if at_end:
					inside_ent = False
					try:
						current_ent = next(iter_ents)
					except StopIteration:
						current_ent = None
			else: # Token does not overlap entity (or no entities left)
				#print('No tag.')
				labels.append(StandoffLabels.outside_label)
				self.token_entities.append(None)

		#self.embeddings = numpy.vstack(embeddings)
		self.labels = numpy.array(labels)
		self.tokens = numpy.array(tokens)

		self.token_entity_map = dict(self.token_entity_map)

		#for e in self.entities:
		#	print(f'Entity: "{e.text}", {self.token_entity_map[e]}, "{" ".join(tokens[i] for i in self.token_entity_map[e])}")')


	def open(filepath,types=None,include_bio=True):
		return StandoffLabels(Standoff.open(filepath),types=types,include_bio=include_bio)

	def __getitem__(self, x):
		#return self.tokens[x],self.labels[x],self.embeddings[x]
		return self.tokens[x],self.labels[x]

	def __iter__(self):
		return zip(self.tokens,self.labels)

	def entity_mask(self, entity):
		return numpy.array([(1 if e is entity else 0) for e in self.token_entities])

	def get_token_idxs(self,entity):
		idxs = self.token_entity_map[entity]
		if len(idxs) > 1:
			start,end = idxs[0],idxs[-1]+1
		else:
			start = idxs[0]
			end = idxs[0]+1
		return start,end

	@property
	def text(self):
		return self.standoff.text
	#@property
	#def entities(self):
	#	return self.standoff.entities
	#@property
	#def relations(self):
	#	return self.standoff.relations
	#@property
	#def attributes(self):
	#	return self.standoff.attributes
	#@property
	#def notes(self):
	#	return self.standoff.notes


if __name__ == '__main__':

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	#from gaml.annotations.word2vec import WordEmbeddings

	parser = ArgumentParser(description='Test standoff embedding code.')
	parser.add_argument('ann',action=FileAction,mustexist=True,help='Annotation file.')
	#parser.add_argument('dictionary',action=FileAction,mustexist=True,help='Word embeddings file.')
	args = parser.parse_args()

	#ann = Standoff.open(args.ann)
	#dictionary = WordEmbeddings(args.dictionary)

	labelled = StandoffLabels.open(args.ann,types=['MeasuredValue','Constraint','ParameterSymbol','ParameterName'])

	for token,label in labelled:
		print(f'{token:20} {label}')

	tokens,labels = zip(*labelled)
	print(tokens)
	print(labels)
