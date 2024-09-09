import os
import re
from copy import copy,deepcopy
import itertools
import argparse
from collections import defaultdict

# T# TYPE # # TEXT
# R# TYPE Arg1:T# Arg2:T#
# A# TYPE T#
annre = re.compile(r'''
	(?P<tag>[TRA\#][0-9]+)
	\s+
	(?P<type>[A-Za-z]+)
	\s+
	(?:
		(?P<annotation>[TRA][0-9]+)\s+(?P<note>.+)
		|
		(?P<start>[0-9]+)\s+(?P<end>[0-9]+)\s+(?P<text>.+)
		|
		Arg1:(?P<arg1>T[0-9]+)\s+Arg2:(?P<arg2>T[0-9]+)
		|
		(?P<subject>T[0-9]+)
	)
	''',flags=re.VERBOSE)

def parse_standoff(text):
	lines = text.splitlines()

	entities = {}
	relations = {}
	attributes = {}
	notes = {}

	for line in lines:
		match = annre.match(line.strip())
		if match:
			tag = match.group('tag')
			if tag[0] == 'T':
				entities[tag] = Entity(
						match.group('type'),
						int(match.group('start')),
						int(match.group('end')),
						match.group('text'))

	###### THIS IS AN UGLY HACK. MAKE PRETTIER.
	for line in lines:
		match = annre.match(line.strip())
		if match:
			tag = match.group('tag')
			if tag[0] =='R':
				relations[tag] = Relation(
						match.group('type'),
						entities[match.group('arg1')],
						entities[match.group('arg2')])
			elif tag[0] == 'A':
				attributes[tag] = Attribute(
						match.group('type'),
						entities[match.group('subject')])
			elif tag[0] == '#':
				assert match.group('type') == 'AnnotatorNotes'
				annotation = match.group('annotation')
				if annotation[0] == 'T':
					subject = entities[annotation]
				elif annotation[0] == 'R':
					subject = relations[annotation]
				elif annotation[0] == 'A':
					subject = attributes[annotation]
				notes[tag] = Note(
						subject,
						match.group('note'))

	return list(entities.values()), list(relations.values()), list(attributes.values()), list(notes.values())

def shortest_distance(s1, e1, s2, e2):
	if s1 < s2:
		return abs(s2-e1)
	else:
		return abs(s1-e2)

def get_text_span(s1, e1, s2, e2,offset=0):
	if s1 < s2:
		return (max(0,s1-offset),e2+offset)
	else:
		return (max(0,s2-offset),e1+offset)

def is_overlapping(s1,e1,s2,e2):
	if s1 <= s2:
		return s2 < e1
	else:
		return s1 < e2

def overlap(s1,e1,s2,e2):
	if s2 < s1:
		return overlap(s2,e2,s1,e1)
	else: # s1 <= s2
		if s2 >= e1:
			return 0.0
		else: # s2 < e1
			if e2 > e1:
				return (e1-s2)/(e2-s1)
			else: # e2 <= e1
				return (e2-s2)/(e1-s1)

def find_overlap_span(s1,e1,s2,e2):
	if s2 < s1:
		return find_overlap_span(s2,e2,s1,e1)
	else: # s1 <= s2
		if s2 >= e1:
			return None
		else: # s2 < e1
			return max(s1,s2),min(e1,e2)

class Standoff:
	def __init__(self,text,entities,relations,attributes,notes):
		self.text = text
		self.entities = entities
		self.relations = relations
		self.attributes = attributes
		self.notes = notes

	def open(filename):
		annfile = os.path.splitext(filename)[0]+'.ann'
		txtfile = os.path.splitext(filename)[0]+'.txt'

		with open(txtfile,'r',encoding='utf8') as f:
			text = f.read()

		with open(annfile,'r',encoding='utf8') as f:
			entities,relations,attributes,notes = parse_standoff(f.read())

		if not all(ent.text == text[ent.start:ent.end] for ent in entities):
			raise ValueError(f'{filename} is badly structured. Text samples do not match txt file present.')

		return Standoff(text,entities,relations,attributes,notes)

	def save(self, filename):
		annfile = os.path.splitext(filename)[0]+'.ann'
		txtfile = os.path.splitext(filename)[0]+'.txt'

		with open(txtfile,'w',encoding='utf8') as f:
			f.write(self.text)
		with open(annfile,'w',encoding='utf8') as f:
			if len(self) > 0:
				f.write(str(self))

	def create(text):
		return Standoff(text,[],[],[],[])

	def __copy__(self):
		return Standoff(copy(self.text),copy(self.entities),copy(self.relations),copy(self.attributes),copy(self.notes))

	def __deepcopy__(self,memo):
		return Standoff(deepcopy(self.text,memo),deepcopy(self.entities,memo),deepcopy(self.relations,memo),deepcopy(self.attributes,memo),deepcopy(self.notes,memo))

	def __str__(self):
		return '\n'.join(
			[f'T{i+1}\t{ent.type} {ent.start} {ent.end}\t{ent.text}' for i,ent in enumerate(self.entities)] +
			[f'R{i+1}\t{rel.type} Arg1:T{self.entities.index(rel.arg1)+1} Arg2:T{self.entities.index(rel.arg2)+1}' for i,rel in enumerate(self.relations)] +
			[f'A{i+1}\t{att.type} T{self.entities.index(att.subject)+1}' for i,att in enumerate(self.attributes)] +
			[f'#{i+1}\tAnnotatorNotes T{self.entities.index(note.subject)+1}\t{note.note}' for i,note in enumerate(self.notes) if note.subject in self.entities] +
			[f'#{i+1}\tAnnotatorNotes R{self.relations.index(note.subject)+1}\t{note.note}' for i,note in enumerate(self.notes) if note.subject in self.relations]) + '\n'

	def __len__(self):
		return len(self.entities)+len(self.relations)+len(self.attributes)+len(self.notes)

	def entity(self,type,start,end):
		ent = Entity(type,start,end,self.text[start:end])
		self.entities.append(ent)
		return ent

	def relation(self,type,arg1,arg2):
		if arg1 not in self.entities or arg2 not in self.entities:
			raise ValueError('Provided arguments not in entity list.')
		rel = Relation(type,arg1,arg2)
		self.relations.append(rel)
		return rel

	def attribute(self,type,subject):
		if subject not in self.entities:
			raise ValueError('Provided subject not in entity list.')
		attr = Attribute(type,subject)
		self.attributes.append(attr)
		return attr

	def note(self,subject,note):
		if subject not in self.entities and subject not in self.relations:
			raise ValueError('Provided subject not in entity/relation lists.')
		nt = Note(subject,note)
		self.notes.append(nt)
		return nt

	def get_annotations(self,start,end):
		overlapping = []
		for ent in self.entities:
			if is_overlapping(start,end,ent.start,ent.end):
				overlapping.append(ent)
		return overlapping

	def get_relation(self,arg1,arg2):
		## What about case of multiple relations...?
		return next((r for r in self.relations if r.arg1==arg1 and r.arg2==arg2),None)

	def get_attributes(self, subject):
		return tuple(att for att in self.attributes if att.subject == subject)

	def is_annotated(self,start,end):
		return any(is_overlapping(start,end,ent.start,ent.end) for ent in self.entities)

	def __iter__(self):
		return iter(self.entities+self.relations+self.attributes+self.notes)

	def annotations(self):
		return self.entities+self.relations+self.attributes

class Entity:
	def __init__(self,type,start,end,text):
		if start >= end:
			raise ValueError('Impossible entity annotation specified.')
		self.type = type
		self.start, self.end = start, end
		self.text = text

	def __len__(self):
		return self.end-self.start

	def __str__(self):
		return f'{self.type} {self.start} {self.end}\t{self.text}'

	def __repr__(self):
		return f'Entity({self.type}, {self.start}, {self.end}, "{self.text}")'

	def __eq__(self,other):
		if isinstance(other,Entity):
			return self.type == other.type and self.start == other.start and self.end == other.end and self.text == other.text
		else:
			return NotImplemented

	def __hash__(self):
		return hash((self.type,self.start,self.end,self.text))

class Relation:
	def __init__(self,type,arg1,arg2):
		if arg1 is None or arg2 is None:
			raise ValueError('Null argument detected.')
		self.type = type
		self.arg1 = arg1
		self.arg2 = arg2

	def __eq__(self,other):
		if isinstance(other,Relation):
			return self.type == other.type and self.arg1 == other.arg1 and self.arg2 == other.arg2
		else:
			return NotImplemented

	def __repr__(self):
		return f'Relation({self.type}, {repr(self.arg1)}, {repr(self.arg2)})'

	def __hash__(self):
		return hash((self.type,self.arg1,self.arg2))

class Attribute:
	def __init__(self,type,subject):
		if subject is None:
			raise ValueError('Null subject detected.')
		self.type = type
		self.subject = subject

	def __repr__(self):
		return f'Attribute({self.type}, {repr(self.subject)})'

	def __eq__(self,other):
		if isinstance(other,Attribute):
			return self.type == other.type and self.subject == other.subject
		else:
			return NotImplemented

	def __hash__(self):
		return hash((self.type,self.subject))

class Note:
	def __init__(self,subject,note):
		if subject is None:
			raise ValueError('Null subject detected.')
		self.subject = subject
		self.note = note

	def __eq__(self,other):
		if isinstance(other,Note):
			return self.subject == other.subject and self.note == other.note
		else:
			return NotImplemented

	def __repr__(self):
		return f'Note({repr(self.arg1)}, "{self.note}")'

	def __hash__(self):
		return hash((self.subject,self.note))

# [section] # Section header
# <name>=list # Substitution
# !name # Ignore entity
# \s+name # Entity
# name Arg1:name Arg2:name # Relation
# name Arg:name # Attribute
configre  = re.compile(r'''
	\[(?P<section>entities|relations|events|attributes)\]
	|
	(?P<macroname>\<[A-Z\-_]+\>)=(?P<macro>[A-Za-z\|]+)
	|
	\s*(?P<ignore>\![A-Za-z]+)
	|
	\s*(?P<entity>[A-Za-z]+)
	|
	(?P<relation>[A-Za-z]+)\s+Arg1:(?P<arg1>[A-Za-z\<\>\-_\|]+),\s+Arg2:(?P<arg2>[A-Za-z\<\>\-_\|]+)
	|
	(?P<attribute>[A-Za-z]+)\s+Arg:(?P<arg>[A-Za-z\<\>\-_\|]+)
	''',flags=re.VERBOSE)
def parse_config(text):
	lines = text.splitlines()

	entities = []
	macros = {}
	relations = defaultdict(list)

	def add_macros(t):
		for m,v in macros.items():
			t = t.replace(m,v)
		return t

	for line in lines:
		match = configre.fullmatch(line.strip())
		if line and match:
			groups = match.groupdict()
			if groups['section']:
				section = groups['section']
				macros = {}
			elif groups['macroname']:
				macros[groups['macroname']] = groups['macro']
			elif groups['ignore']:
				if section == 'entities':
					pass
				else:
					raise ValueError(f'Entity found outside of [entities] section: "{line.strip()}"')
			elif groups['entity']:
				if section == 'entities':
					entities.append(groups['entity'])
				else:
					raise ValueError(f'Entity found outside of [entities] section: "{line.strip()}"')
			elif groups['relation']:
				if section == 'relations':
					start = add_macros(groups['arg1']).split('|')
					end = add_macros(groups['arg2']).split('|')
					for i in itertools.product(start,end):
						relations[i].append(groups['relation']) # There could be several valid relationship types
				else:
					raise ValueError(f'Relation found outside of [relations] section: "{line.strip()}"')
			elif groups['attribute']:
				if section == 'attributes':
					#args = add_macros(groups['arg']).split('|')
					## Do something
					pass
				else:
					raise ValueError(f'Attribute found outside of [attributes] section: "{line.strip()}"')
		elif line:
			raise ValueError(f'Could not parse line: "{line.strip()}"')

	return entities,dict(relations)


class StandoffConfig:
	def __init__(self, entities, relations):
		self.entities = entities
		self.relations = relations # dict of arg tuples and lists of types

	def open(filename):
		with open(filename,'r',encoding='utf8') as f:
			entities,relations = parse_config(f.read())

		return StandoffConfig(entities,relations)

	def allowed_relation(self,start,end):
		if isinstance(start,Entity):
			start = start.type
		if isinstance(end,Entity):
			end = end.type
		return (start,end) in self.relations

class StandoffConfigAction(argparse.Action):
	"""docstring for StandoffConfigAction"""
	def __call__(self, parser, namespace, values, option_string=None):

		path = os.path.abspath(values)

		if os.path.isfile(path):
			metadata = StandoffConfig.open(path)
		else:
			parser.error('Supplied standoff config file ('+path+') does not exist.')

		setattr(namespace, self.dest, metadata)


if __name__ == '__main__':

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	parser = ArgumentParser(description='Test brat utilities.')
	parser.add_argument('ann',action=FileAction,mustexist=True,help='Annotation file.')
	args = parser.parse_args()

	annfile = Standoff.open(args.ann)

	print(annfile)

	print('--------------')

	with open(args.ann,'r',encoding='utf8') as f:
		print(f.read())
