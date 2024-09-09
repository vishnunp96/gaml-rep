import os
import re
import regex
from copy import copy,deepcopy
import itertools
from collections import defaultdict,Counter

from gaml.annotations.bratutils import Standoff,overlap,is_overlapping,find_overlap_span
from gaml.parsing import parse_measurement

## Load custom stopwords
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'stopwords.txt'),'r') as f:
	stopwords = set(f.read().splitlines())
	stopwords_tup = tuple(stopwords)

stopwordre = re.compile('|'.join([f'(?:^|\s+){w}(?:$|\s+)' for w in sorted(stopwords,reverse=True)]),flags=re.IGNORECASE)

def remove_stopwords(ann,inplace=False):
	if not inplace:
		ann = deepcopy(ann)

	toremove = []
	for entity in ann.entities:
		start,end = entity.start,entity.end

		for m in stopwordre.finditer(entity.text):
			if m.start() == 0:
				start += m.end()
			elif m.end() == len(entity.text):
				end -= len(m.group())

		if end>start:
			entity.start,entity.end = start,end
			entity.text = ann.text[start:end]
		elif start==end:
			toremove.append(id(entity))

	ann.entities[:] = [e for e in ann.entities if id(e) not in toremove]

	return ann

def find_repititions(ann,inplace=False,types=('ParameterName','ParameterSymbol','ObjectName')):
	if not inplace:
		ann = deepcopy(ann)
	# Make a copy so we can add entities as we iterate
	entities = ann.entities.copy()

	checked = set()

	for entity in sorted(entities,key=lambda e: len(e.text),reverse=True):
		if entity.type in types and entity.text not in checked and entity.text != 'a':
			entity_text = entity.text.strip()
			for m in re.finditer(f'(?:^|\s+)(?P<entity>{re.escape(entity_text)})(?:$|\s+)',ann.text):
				if not ann.is_annotated(m.start('entity'),m.end('entity')):
					ann.entity(entity.type,m.start('entity'),m.end('entity'))

			checked.add(entity_text)

	return ann

constraint_separators = ['<','>','\gt','\lt','\leq','\geq']
sep_patterns = [(s,f'\s+{re.escape(s)}\s+') for s in constraint_separators]
def NOTseparate_constraints(ann,inplace=False):
	if not inplace:
		ann = deepcopy(ann)
	# Make copy so we can alter entities as we iterate
	entities = ann.entities.copy()

	for entity in [e for e in entities if e.type == 'Constraint']:
		for sep,pattern in sep_patterns:
			if sep in entity.text:
				#(entity.text)
				pass

sep_pattern = regex.compile('\s+(' + '|'.join([regex.escape(s) for s in constraint_separators]) + ')\s+')
def separate_constraints(ann,inplace=False):
	if not inplace:
		ann = deepcopy(ann)
	# Make copy so we can alter entities as we iterate
	entities = ann.entities.copy()

	for entity in [e for e in entities if e.type == 'Constraint']:
		matches = [m for m in sep_pattern.finditer(entity.text)]
		if matches:
			divisions = list()
			for i,match in enumerate(matches):
				span = match.span()
				start_i = matches[i-1].span()[1] if i>0 else 0
				end_i = matches[i+1].span()[0] if i<len(matches)-1 else len(entity.text)
				before = (start_i,span[0])
				after = (span[1],end_i)
				divisions.append((before,match,after))

			def print_division(div):
				before = div[0]
				match = div[1]
				after = div[2]
				before_text = entity.text[before[0]:before[1]]
				after_text = entity.text[after[0]:after[1]]
				divider = match.group()

				m_before = parse_measurement(before_text)
				m_after = parse_measurement(after_text)
				if m_before and not m_after:
					left = 'Constraint'
					right = 'Parameter'
				elif not m_before and m_after:
					left = 'Parameter'
					right = 'Constraint'
				else:
					left,right = 'UNKNOWN','UNKNOWN'

				print(f'{left}: "{before_text}", {right}: "{after_text}", Divider: "{divider}"')

			print(f'ANNOTATION: "{entity.text}"')
			if len(divisions) == 1:
				div = divisions[0]
				print_division(div)
			elif len(divisions) == 2:
				print_division(divisions[0])
				print_division(divisions[1])

	return ann

def sort(ann,inplace=False):
	if not inplace:
		ann = deepcopy(ann)
	ann.entities.sort(key=lambda e: e.start)
	ann.relations.sort(key=lambda r: (r.arg1.start,r.arg2.start))
	ann.attributes.sort(key=lambda a: a.subject.start)
	return ann

def self_overlap(ann):
	for i in ann.entities:
		for j in ann.entities:
			if i is not j:
				if is_overlapping(i.start,i.end,j.start,j.end):
					return True
	return False

def compare_annotations(anns,ignore,verbose=0):
	if len(set(a.text for a in anns)) > 1:
		raise ValueError('Cannot compare: Standoff objects based on different text files.')
	#if any(self_overlap(a) for a in anns):
	#	#raise ValueError('Cannot compare: Standoff object contains overlapping annotations.')
	#	print('SKIPPING DUE TO OVERLAPPING ANNOTATIONS')
	#	return 0

	#for a in anns:
	#	print(f'---------------------------------------\n{a.annotator}')
	#	print(a)
	#print('---------------------------------')

	class Grouping:
		def __init__(self,first):
			self.entities = [first]
			self.start = first.start
			self.end = first.end

		def __iter__(self):
			return iter(self.entities)

		def is_overlapping(self,ent):
			return is_overlapping(self.start,self.end,ent.start,ent.end)

		def add(self,ent):
			self.entities.append(ent)
			self.start = min(self.start,ent.start)
			self.end = max(self.end,ent.end)

		def get_primary(self):
			potentials = []
			for e in self.entities:
				if all(is_overlapping(e.start,e.end,other.start,other.end) for other in self.entities):
					potentials.append(e)
			if potentials:
				return max(potentials,key=len)
			else:
				## Need a way of finding entity with highest "concentration"
				return max(self.entities,key=len)

		def split(self):
			counter = defaultdict(list)
			for e in self.entities:
				counter[(e.start,e.end)].append(e)
			if max(len(v) for v in counter.values())>(len(anns)/2):
				primary_key = max(counter.keys(), key=lambda k: len(counter[k]))
				return counter[primary_key], list(itertools.chain.from_iterable(v for k,v in counter.items() if k is not primary_key))
			if all(len(v)==1 for v in counter.values()):
				return self.split_contested()

		def split_multiple_primaries(self):
			counter = defaultdict(list)
			for e in self.entities:
				counter[(e.start,e.end)].append(e)
			if all(len(v)==1 for v in counter.values()):
				p,r = self.split_contested()
				return [p],r
			primary_keys = [k for k in counter.keys() if len(counter[k])>len(anns)/2]
			return [counter[pk] for pk in primary_keys], list(itertools.chain.from_iterable(v for k,v in counter.items() if k not in primary_keys))

		def split_contested(self):
			primary = self.get_primary()
			remaining = copy(self.entities)
			remaining.remove(primary)
			return [primary],remaining

		def __str__(self):
			return 'Group([' + '\n'.join('\t'+str(e) for e in self.entities) + f'], {self.start}, {self.end})'

	annotators = {}
	for standoff in anns:
		for annotation in standoff:
			annotators[id(annotation)] = standoff.annotator

	groupings = []
	for e in sorted(itertools.chain.from_iterable(a.entities for a in anns),key=lambda e: e.start):
		found_overlap = False
		for g in groupings:
			if g.is_overlapping(e):
				found_overlap = True
				g.add(e)
				#print(f'{annotators[id(e)]} ({e.text}): Added to group')
		if not found_overlap:
			groupings.append(Grouping(e))
			#print(f'{annotators[id(e)]} ({e.text}): New group made')

	groupings = [g for g in groupings if any(e.type not in ignore for e in g)]

	consensus = Standoff.create(anns[0].text)
	consensus_listings = {}

	for group in groupings:
		#primary_list,remaining = group.split()
		all_primary_lists,remaining = group.split_multiple_primaries()
		#print(f'all_primary_lists = {all_primary_lists}')
		#print(f'remaining = {remaining}')
		for primary_list in all_primary_lists:
			primary = sorted(primary_list,key=lambda p: Counter(e.type for e in group)[p.type], reverse=True)[0]
			overlapping = list(o for o in itertools.chain.from_iterable(a.get_annotations(primary.start,primary.end) for a in anns) if id(o)!=id(primary))
			disagree = []
			agree = [(primary,1.0)]
			for o in overlapping:
				overlap_val = overlap(primary.start,primary.end,o.start,o.end)
				if overlap_val == 1 and o.type == primary.type:
					agree.append((o,overlap_val))
				else:
					disagree.append((o,overlap_val))
			if verbose>0:
				print('-----------')
				#for e in group: print(f'\t{e}')
				#print(f'\tprimary_list = ' + repr(primary_list))
				#print(f'\tremaining = ' + repr(remaining))
				#print(f'\tprimary = ' + repr(primary))
				#print(f'\toverlapping = ' + repr(overlapping))
				#print(f'\tagree = ' + repr(agree))
				#print(f'\tdisagree = ' + repr(disagree))
				#multiple_primaries,multiple_remaining = group.split_multiple_primaries()
				#print(f'\tmultiple_primaries =\n\t\t' + '\n\t\t'.join(repr(p) for p in multiple_primaries))
				#print(f'\tmultiple_remaining = ' + repr(multiple_remaining))

				print(f'{primary} ({"/".join(str(annotators[id(p)]) for p in primary_list if p==primary)})')
				for d,v in disagree: print(f'!! {v:.2f}: {d} ({annotators[id(d)]})')
			if len(agree)==len(anns):
				if verbose>0: print('AGREEMENT')
				if primary.type not in ignore:
					ent = consensus.entity(primary.type,primary.start,primary.end)
					for e,v in agree: consensus_listings[id(e)] = ent
			elif len(agree) > (len(anns)/2):
				if verbose>0: print(f'CONSENSUS ({len(agree)} against {len(disagree)})' + (' INCLUDED' if primary.type not in ignore else ''))
				if primary.type not in ignore:
					ent = consensus.entity(primary.type,primary.start,primary.end)
					for e,v in agree: consensus_listings[id(e)] = ent
			elif len(agree)==1 and len(disagree)==0:
				if verbose>0: print('SINGLETON')
			else:
				if verbose>0: print(f'Overlap count => agree: {len(agree)-1}, disagree: {len(disagree)}')
				type_matches = [primary] + [o for o in overlapping if o.type==primary.type]
				if len(type_matches) > (len(anns)/2):
					if verbose>1: print(f'Type matches: {type_matches}')
					## Get span overlap
					s,e = type_matches[0].start,type_matches[0].end
					accepted = [type_matches[0]]
					for ent in sorted(type_matches[1:], key=lambda i:len(i), reverse=True):
						new_se = find_overlap_span(s,e,ent.start,ent.end)
						if new_se is not None:
							s,e = new_se
							accepted.append(ent)
					ent = consensus.entity(primary.type,s,e)
					for e in accepted: consensus_listings[id(e)] = ent
					if verbose>0: print(f'Found compromise: {ent}')
				#if overlapping:
				#	s,e = primary.start,primary.end
				#	for a in overlapping:
				#		s,e = find_overlap_span(s,e,a.start,a.end)
				#	print(s,e)
				#	print(f'Agreement on: {anns[0].text[s:e]}')
				## Cases:
				# 2 partial agreement, nothing from 3rd
				# 2 (or more) different partial agreements (one long span with two other non-overlapping subspans)


	class AttributeGroup:
		def __init__(self,first):
			self.attributes = [first]
			self.subject = consensus_listings[id(first.subject)]

		def __iter__(self):
			return iter(self.attributes)

		def add(self, attr):
			self.attributes.append(attr)

	attributes = []
	for attr in sorted(itertools.chain.from_iterable(a.attributes for a in anns),key=lambda e: e.subject.start):
		#print(f'Check: {attr.type} {attr.subject} ({annotators[id(attr)]})')
		if id(attr.subject) not in consensus_listings:
			#print(f'\tNO CONSENSUS FOR ATTRIBUTE')
			continue

		found_overlap = False
		for attr_group in attributes:
			if attr_group.subject == consensus_listings[id(attr.subject)]:
				found_overlap = True
				attr_group.add(attr)
		if not found_overlap:
			attributes.append(AttributeGroup(attr))

	attributes = [attrs for attrs in attributes if any(a.type not in ignore for a in attrs)]

	for attr_group in attributes:
		if verbose>1:
			print('Group:')
			for attr in attr_group: print(f'\t{attr.type} {attr.subject} ({annotators[id(attr)]})')
		primary,remaining = attr_group.attributes[0],attr_group.attributes[1:]
		agree,disagree = [primary],[]
		for r in remaining:
			if r.type == primary.type:
				agree.append(r)
			else:
				disagree.append(r)
		if len(agree)==len(anns):
			attr = consensus.attribute(primary.type,attr_group.subject)
			for a in agree: consensus_listings[id(a)] = attr
			if verbose>1: print('ATTRIBUTE AGREEMENT')
		elif len(agree)==1 and len(disagree)==0:
			if verbose>1: print('ATTRIBUTE SINGLETON')
		elif len(agree) > (len(anns)/2):
			attr = consensus.attribute(primary.type,attr_group.subject)
			for a in agree: consensus_listings[id(a)] = attr
			if verbose>1: print('ATTRIBUTE CONSENSUS')
		else:
			if verbose>1: print('ATTRIBUTE DISAGREEMENT')


	class RelationGroup:
		def __init__(self,first):
			self.relations = [first]
			self.arg1 = consensus_listings[id(first.arg1)]
			self.arg2 = consensus_listings[id(first.arg2)]

		def __iter__(self):
			return iter(self.relations)

		def add(self, rel):
			self.relations.append(rel)

	relations = []
	for rel in sorted(itertools.chain.from_iterable(a.relations for a in anns),key=lambda r: (r.arg1.start,r.arg2.start)):
		#print(f'Check: {rel.type} {rel.arg1.type} {rel.arg2.type} ({annotators[id(rel)]})')
		if id(rel.arg1) not in consensus_listings or id(rel.arg2) not in consensus_listings:
			#print(f'\tNO CONSENSUS FOR RELATION')
			continue

		found_overlap = False
		for rel_group in relations:
			if rel_group.arg1 == consensus_listings[id(rel.arg1)] and rel_group.arg2 == consensus_listings[id(rel.arg2)]:
				found_overlap = True
				rel_group.add(rel)
		if not found_overlap:
			relations.append(RelationGroup(rel))

	relations = [rels for rels in relations if any(r.type not in ignore for r in rels)]

	for rel_group in relations:
		if verbose>2:
			print('Group:')
			for rel in rel_group: print(f'\t{rel.type} {rel.arg1.type} {rel.arg2.type} ({annotators[id(rel)]})')
		primary,remaining = rel_group.relations[0],rel_group.relations[1:]
		agree,disagree = [primary],[]
		for r in remaining:
			if r.type == primary.type:
				agree.append(r)
			else:
				disagree.append(r)
		if len(agree)==len(anns):
			rel = consensus.relation(primary.type,rel_group.arg1,rel_group.arg2)
			for a in agree: consensus_listings[id(a)] = rel
			if verbose>2: print('RELATION AGREEMENT')
		elif len(agree)==1 and len(disagree)==0:
			if verbose>2: print('RELATION SINGLETON')
		elif len(agree) > (len(anns)/2):
			rel = consensus.relation(primary.type,rel_group.arg1,rel_group.arg2)
			for a in agree: consensus_listings[id(a)] = rel
			if verbose>2: print('RELATION CONSENSUS')
		else:
			if verbose>2: print('RELATION DISAGREEMENT')


	if verbose>2: print(f'\n{consensus}\n')

	#sim = similarity(interested, anns)
	#print(f'######### {sim}')
	if verbose>0: print('######### ' + ' | '.join(f'{a.annotator}: {similarity(ignore,[a,consensus]):4.2f}' for a in anns))
	return consensus

def jaccart(iterables):
	sets = [set(i) for i in iterables]
	union = len(set.union(*sets))
	if union>0:
		return len(set.intersection(*sets))/union
	else:
		return 0

def similarity(ignore,anns):
	return jaccart([e for e in a if e.type not in ignore] for a in anns)
def similarity_include(interested,anns):
	return jaccart([i for i in a if i.type in interested] for a in anns)

relation_norms = {'Name':('ParameterName','ParameterSymbol')}
def normalise_name_relation(ann):
	for rel in ann.relations:
		if rel.type in relation_norms:
			if rel.arg1.type == relation_norms[rel.type][1] and rel.arg2.type == relation_norms[rel.type][0]:
				arg1,arg2 = rel.arg1,rel.arg2
				rel.arg1,rel.arg2 = arg2,arg1

def add_implied_relations(ann,inplace=False):
	if not inplace:
		ann = deepcopy(ann)

	def found_relation(rel_type,arg1,arg2):
		if not ann.get_relation(arg1,arg2):
			ann.relation(rel_type,arg1,arg2)

	# Implied Measurement relations
	## ParameterName -> (ParameterSymbol -> Value)
	for m in (i for i in ann.relations if i.type=='Measurement' and i.arg1.type=='ParameterSymbol'):
		# Symbol-Value measurements might be missing a connection to a parent ParameterName
		for n in (j.arg1 for j in ann.relations if j.type=='Name' and j.arg2==m.arg1):
			# Found a parent ParameterName
			found_relation('Measurement',n,m.arg2)
	## (ParameterName -> ParameterSymbol) -> Value
	for nrel in (i for i in ann.relations if i.type=='Name'):
		# ParameterName-Symbol Name relations might be missing Measurement relations from name to symbol children
		for v in (r.arg2 for r in ann.relations if r.type=='Measurement' and r.arg1==nrel.arg2):
			# Found "grandparent" ParameterName
			found_relation('Measurement',nrel.arg1,v)

	# Implied Name relations
	## (ParameterName -> Value <- ParameterSymbol)
	## (ParameterSymbol -> Value <- ParameterName)
	for m in (i for i in ann.relations if i.type=='Measurement' and i.arg1.type=='ParameterSymbol'):
		# Symbol-Value measurements might be missing a Name rel from a ParameterName also linked by a Measurement rel
		for n in (r.arg1 for r in ann.relations if r.type=='Measurement' and r.arg1.type=='ParameterName' and r.arg2==m.arg2):
			# Found a parent ParameterName
			found_relation('Name',n,m.arg1)
	for m in (i for i in ann.relations if i.type=='Measurement' and i.arg1.type=='ParameterName'):
		# Name-Value measurements might be missing a Name rel to a ParameterSymbol also linked by a Measurement rel
		for s in (r.arg1 for r in ann.relations if r.type=='Measurement' and r.arg1.type=='ParameterSymbol' and r.arg2==m.arg2):
			# Found a child ParameterSymbol
			found_relation('Name',m.arg1,s)

	## Helper method for property relations
	def implied_value_properties(ob,ent):
		for v in (r.arg2 for r in ann.relations if ent==r.arg1 and r.type=='Measurement'):
			found_relation('Property',ob,v)

	def implied_name_properties(ob,name):
		for s in (r.arg2 for r in ann.relations if name==r.arg1 and r.type=='Name'):
			found_relation('Property',ob,s)
			implied_value_properties(ob,s)

	# Implied Property relations
	## (ObjectName -> ParameterName -> ParameterSymbol -> Value)
	## (ObjectName -> ParameterSymbol -> Value)
	## (ObjectName -> Value <- ParameterSymbol [<- ParameterName]) # Do not need to worry about tail ParameterName, as it will be caught by implied name relations above
	## (ObjectName -> Value <- ParameterName)
	for ob in (i for i in ann.entities if i.type=='ObjectName'):
		for e in (r.arg2 for r in ann.relations if r.arg1==ob and r.type=='Property'):
			if e.type=='ParameterName':
				implied_name_properties(ob,e)
				implied_value_properties(ob,e)
			elif e.type=='ParameterSymbol':
				implied_value_properties(ob,e)
			elif e.type=='MeasuredValue' or e.type=='Constraint':
				for param in (r.arg1 for r in ann.relations if e==r.arg2 and r.type=='Measurement'):
					found_relation('Property',ob,param)

	return ann

def open_clean(path, check_repetitions=('ParameterName','ParameterSymbol','ObjectName')):
	try:
		ann = Standoff.open(path)
		ann = remove_stopwords(ann,inplace=True)
		ann = find_repititions(ann,inplace=True,types=check_repetitions)
		normalise_name_relation(ann)
		#ann = separate_constraints(ann,inplace=True)
	except KeyError:
		txtfile = os.path.splitext(path)[0]+'.txt'
		with open(txtfile,'r',encoding='utf8') as f:
			text = f.read()
		ann = Standoff.create(text)
	ann.annotator = os.path.basename(os.path.dirname(os.path.dirname(path)))
	ann.path = path
	return ann

def combine(ann1,ann2):
	entities = set()
	ent_dict = {}
	for ent in ann1.entities+ann2.entities:
		if ent in entities:
			match = next((e for e in entities if e==ent), None)
			ent_dict[id(ent)] = match
		else:
			entities.add(ent)
			ent_dict[id(ent)] = ent

	## Finish?
	pass

if __name__ == '__main__':

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction

	parser = ArgumentParser(description='Test brat normalisation.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('name',help='Name of sample to be examined.')
	parser.add_argument('-p','--paper',help='arXiv identifier for specific paper to examined. Otherwise all papers in sample will be processed.')
	args = parser.parse_args()

	listings = defaultdict(list)

	#for path in (p for p in args.ann if args.name == os.path.basename(os.path.dirname(p))):
	for path in (p for p in args.ann if args.name == os.path.basename(os.path.dirname(p)) and (not args.paper or args.paper in p)):
		arXiv = os.path.basename(os.path.splitext(path)[0])
		listings[arXiv].append(path)

	#interested = ('MeasuredValue','Constraint','ParameterSymbol','ParameterName','ObjectName','ConfidenceLimit','SeparatedUncertainty',
	#		'Measurement','Uncertainty','Confidence','Name','UncertaintyName','Property','Equivalence','Contains',
	#		'UpperBound','LowerBound','FromLiterature','AcceptedValue','Incorrect')
	#interested = ('MeasuredValue',)
	ignore = ('Reference','Details', # Entities
			'Source') # Relations

	standoffs = [(arXiv,[open_clean(path) for path in paths]) for arXiv,paths in listings.items()]
	standoffs = sorted(((arXiv,anns) for arXiv,anns in standoffs if all(a is not None for a in anns) and any(anns)), key=lambda s: similarity(ignore,s[1]),reverse=False)
	#standoffs = [(arXiv,anns) for arXiv,anns in standoffs if any(a.annotator=='tcrossland' for a in anns)]
	#standoffs = [(arXiv,anns) for arXiv,anns in standoffs if any('astro-ph9506090' in a.path for a in anns)]

	count = 0
	total_similarity = 0
	similarities = defaultdict(list)
	#for arXiv,paths in sorted(listings.items(),key=lambda l: l[0]):
	for i,(arXiv,anns) in enumerate(standoffs):
		#anns = [Standoff.open(path) for path in paths]
		#anns = [(path,open_clean(path)) for path in paths]

		#if all(len(a)>0 for p,a in anns):
		paths_str = '\n'.join([str(a.path) + (' EMPTY' if len(a)==0 else '') for a in anns])
		print(f'\n=============== ({i+1}/{len(standoffs)})\nARXIV: {arXiv}\nPaths:\n{paths_str}\n')

		consensus = compare_annotations(anns, ignore, verbose=1)
		sim = similarity(ignore, anns)
		total_similarity += sim

		count += 1

		annotators = tuple(sorted([a.annotator for a in anns]))
		similarities[annotators].append(sim)

	print(f'\nTotal papers checked: {count}')
	print(f'Average similarity: {total_similarity/count:.4f}')

	#import itertools
	#import numpy

	#annotators = sorted(set(itertools.chain.from_iterable(similarities.keys())))
	#sim_matrix = numpy.zeros((len(annotators),len(annotators)))

	for annotators,sim in similarities.items():
		print(f'{", ".join(str(a) for a in annotators)}: {sum(sim)/len(sim):.3f}')
		#sim_matrix[annotators.index(a1),annotators.index(a2)] = sum(sim)/len(sim)
	#	print(annotators.index(a1),annotators.index(a2), sum(sim)/len(sim))
	#print(sim_matrix)
	#print(sum(sim_matrix))

	#import matplotlib.pyplot as plt
	#fig, ax = plt.subplots()
	#im = ax.imshow(sim_matrix)
	## We want to show all ticks...
	#ax.set_xticks(numpy.arange(len(annotators)))
	#ax.set_yticks(numpy.arange(len(annotators)))
	## ... and label them with the respective list entries
	#ax.set_xticklabels(annotators)
	#ax.set_yticklabels(annotators)
	## Rotate the tick labels and set their alignment.
	#plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	#fig.tight_layout()
	#plt.show()

if __name__ == '!__main__':

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	parser = ArgumentParser(description='Test brat normalisation.')
	parser.add_argument('ann',action=FileAction,mustexist=True,help='Annotation file.')
	args = parser.parse_args()

	ann = Standoff.open(args.ann)

	print(ann)

	print('--------------')

	ann = remove_stopwords(ann)

	print(ann)

	print('--------------')

	ann = find_repititions(ann)

	print(ann)

	print('--------------')

	ann = separate_constraints(ann)

	print(ann)

	print('--------------')

	ann = sort(ann)

	print(ann)

