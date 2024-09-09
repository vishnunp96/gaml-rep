import torch
#import torch.nn as nn

import itertools
from collections import defaultdict
import re
from copy import deepcopy

from contextlib import contextmanager

from sklearn.preprocessing import LabelEncoder,LabelBinarizer

from annotations.bratutils import Standoff
from annotations.bratnormalisation import add_implied_relations

from annotations.models.base import BaseANNRelationModule
from annotations.datasets import AnnotationDataset

def pairwise(iterable):
	''' s -> (s0,s1), (s1,s2), (s2, s3), ... '''
	a, b = itertools.tee(iterable)
	next(b, None)
	return zip(a, b)

# Define model
class RulesRelationModel(BaseANNRelationModule):

	def __init__(self, output_labels, allowed_relations):

		self.allowed_relations = allowed_relations
		self.labels = LabelEncoder().fit(output_labels)

		# (RelationType, Ent1, Ent2, (Spans), Reversed)
		self.simple_relations = [
				('Measurement', 'ParameterSymbol', 'MeasuredValue', ('=','>','<','\\sim','\\simeq','\\approx','\\leq','\\geq','of',''), False),
				('Measurement', 'MeasuredValue', 'ParameterSymbol', ('>','<','\\leq','\\geq','for'), True),
				('Measurement', 'ParameterName', 'MeasuredValue', ('is', 'of','(',''), False),
				('Measurement', 'MeasuredValue', 'ParameterName', ('for','for the'), True),
				('Confidence', 'MeasuredValue', 'ConfidenceLimit', ('(','at','at the'), False),
				('Name', 'ParameterName', 'ParameterSymbol', ('is', 'of', ',','(',''), False),
				('Name', 'ParameterSymbol', 'ParameterName', ('is the', 'is'), True),
				('Property', 'ParameterName', 'ObjectName', ('of', 'of the'), True),
				('Property', 'ObjectName', 'ParameterName', ('',), False),
				('Defined', 'ParameterSymbol', 'Definition', ('is', '=', '\\equiv'), False),
				('Defined', 'ParameterName', 'Definition', ('='), False),
			]

		self.entity_map = {
				'MeasuredValue': 'V',
				'Constraint': 'C',
				'ParameterSymbol': 'S',
				'ParameterName': 'N',
				'ConfidenceLimit': 'L',
				'ObjectName': 'O',
				'Definition': 'D'
			}

		self.simplenamere = re.compile(r'(?<!S)NS')
		self.multiplemeasurementsre = re.compile('(S|N)(V|C)+')
		self.confidencelimitre = re.compile('(V|C)+L')
		self.definitionmeasurementre = re.compile(r'SDV')
		self.standardmeasurementre = re.compile(r'NS(V|C)')
		self.nameobjectpropertyre = re.compile(r'NOS(V|C)')
		self.simplepropertyre = re.compile(r'O(NS|N|S)(V|C)')
		self.tuplemeasurementre = re.compile(r'(?P<names>(N|S)+)(?P<values>(V|C)+)')
		## Pattern "X which is name1 and Y which is name2"?

		self.cuda_flag = False

	def cuda(self, device=None):
		self.cuda_flag = True
		return self

	def cpu(self):
		self.cuda_flag = False
		return self

	def forward(self, x):
		#raise NotImplementedError(f'{self.__class__.__name__} has no \'forward\' method implementation. Use \'predict\' instead.')
		return x

	def make_dataset(self, ann_list):
		# Create custom dataset that performs prediction on creation, and then just directly returns
		# encoded predictions as input data, so that the forward pass can just return the inputs
		return RulesRelationDirectedPredictionDataset(ann_list, self, self.labels, self.allowed_relations, cuda=self.cuda_flag)

	def predict(self, anns, batch_size=1, inplace=True):
		if not inplace:
			anns = deepcopy(anns)

		disallowed = defaultdict(list)
		def disallow_relation(ann,rel_type,arg1,arg2):
			disallowed[(arg1,arg2)].append(rel_type)
			existing = ann.get_relation(arg1,arg2)
			if existing and existing.type==rel_type:
				ann.relations.remove(existing)
		def found_relation(ann,rel_type,arg1,arg2):
			if rel_type in self.labels.classes_ and not ann.get_relation(arg1,arg2) and not disallowed[(arg1,arg2)]:
				ann.relation(rel_type,arg1,arg2)

		for p,i,ann in anns:

			ents = sorted(ann.entities,key=lambda e: e.start)
			line_ents = [[]]
			for i,e in enumerate(ents):
				line_ents[-1].append(e)
				if i!=(len(ents)-1) and '\n' in ann.text[e.end:ents[i+1].start]:
					line_ents.append([])

			#print(line_ents)

			for line in line_ents:

				## First pass: Obvious relations (use connecting span)
				for e1,e2 in pairwise(line):
					span = ann.text[e1.end:e2.start].strip().lower()
					for rel_type,start_t,end_t,connections,reverse in self.simple_relations:
						if e1.type==start_t and e2.type==end_t:
							if span in connections:
								if reverse:
									found_relation(ann,rel_type,e2,e1)
								else:
									found_relation(ann,rel_type,e1,e2)

				line_accepted = [e for e in line if e.type in self.entity_map]
				pattern = ''.join(self.entity_map[e.type] for e in line_accepted)

				for match in self.simplenamere.finditer(pattern):
					found_relation(ann,'Name',line_accepted[match.start()],line_accepted[match.start()+1])

				for match in self.multiplemeasurementsre.finditer(pattern):
					for i in range(match.start()+1,match.end()):
						start = line_accepted[match.start()]
						end = line_accepted[i]
						if not ann.get_relation(start,end):
							found_relation(ann,'Measurement',start,end)

				for match in self.standardmeasurementre.finditer(pattern):
					found_relation(ann,'Name',line_accepted[match.start()],line_accepted[match.start()+1])
					found_relation(ann,'Measurement',line_accepted[match.start()+1],line_accepted[match.start()+2])

				for match in self.definitionmeasurementre.finditer(pattern):
					found_relation(ann,'Measurement',line_accepted[match.start()],line_accepted[match.start()+2])

				cl_accepted = [e for e in line_accepted if e.type in ('MeasuredValue','Constraint','ConfidenceLimit')]
				cl_pattern = ''.join(self.entity_map[e.type] for e in cl_accepted)
				for match in self.confidencelimitre.finditer(cl_pattern):
					for i in range(match.start(),match.end()-1):
						start = cl_accepted[i]
						end = cl_accepted[match.end()-1]
						if not ann.get_relation(start,end):
							found_relation(ann,'Confidence',start,end)

				for match in self.nameobjectpropertyre.finditer(pattern):
					found_relation(ann,'Property',line_accepted[match.start()+1],line_accepted[match.start()])
					found_relation(ann,'Name',line_accepted[match.start()],line_accepted[match.start()+2])
					disallow_relation(ann,'Property',line_accepted[match.start()+1],line_accepted[match.start()+2])

				for match in self.simplepropertyre.finditer(pattern):
					found_relation(ann,'Property',line_accepted[match.start()],line_accepted[match.start()+1])

				line_objects = [e for e in line_accepted if e.type=='ObjectName']
				first_line_objects = [e for e in line_ents[0] if e.type=='ObjectName']
				if len(line_objects)==1 or (len(line_objects)==0 and len(first_line_objects)==1):
					if len(line_objects)==1:
						object_name = line_objects[0]
					else:
						object_name = first_line_objects[0]
					for ent in (e for e in line_accepted if e.type in ('MeasuredValue','Constraint')):
						final_ent = ent
						while True:
							parents = [r.arg1 for r in ann.relations if r.arg2==final_ent and r.arg1.type in ('ParameterSymbol','ParameterName')]
							if parents:
								if final_ent.type in ('MeasuredValue','Constraint'):
									final_ent = next(e for e in parents if e.type in ('ParameterSymbol','ParameterName'))
								elif final_ent.type=='ParameterSymbol':
									final_ent = next(e for e in parents if e.type=='ParameterName')
								else:
									break
							else:
								break
						found_relation(ann,'Property',object_name,final_ent)

				## TODO If cannot find object on this, or first, line, then just find previously mentioned object and use that

				for match in self.tuplemeasurementre.finditer(pattern):
					if len(match.group('names')) >= len(match.group('values')):
						values = [line_accepted[i] for i in range(match.start('values'),match.end('values'))]
						names = [line_accepted[i] for i in range(match.start('names'),match.end('names'))]
						names = names[-len(values):] # Only select a complimentary number of names
						for i,n in enumerate(names):
							for j,v in enumerate(values):
								if i==j:
									found_relation(ann,'Measurement',n,v)
								else:
									disallow_relation(ann,'Measurement',n,v)

			## Add in implied relations to keep everything standard
			add_implied_relations(ann)

		return anns

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		_state_dict = dict()
		_state_dict['__model_class'] = self.__class__.__name__
		_state_dict['output_labels'] = list(self.labels.classes_)
		_state_dict['allowed_relations'] = self.allowed_relations
		return _state_dict

	def load_from_state_dict(_state_dict):
		''' Create an instance of the model based on a provided state_dict of arbitrary shape. '''
		model = RulesRelationModel(
				_state_dict.pop('output_labels'),
				_state_dict.pop('allowed_relations')
			)
		return model

	@contextmanager
	def evaluation(self):
		yield self


class RulesRelationDirectedPredictionDataset(AnnotationDataset):
	def __init__(self, ann_list, model, labels, allowed_relations, cuda=False):
		super(RulesRelationDirectedPredictionDataset,self).__init__(ann_list, cuda)

		self.onehot_relations = LabelBinarizer().fit(labels.classes_)

		entity_only_anns = [('','',Standoff(a.text,a.entities,[],[],[])) for a in anns]
		predictions = model.predict(entity_only_anns, inplace=True)

		xs = []
		ys = []
		self.origins = []
		for a,p in zip(self.anns, predictions):
			for start,end in itertools.permutations(a.entities,2):
				if (start.type,end.type) in allowed_relations:
					orig_rel = next((r.type for r in a.relations if r.arg1==start and r.arg2==end),'none')
					pred_rel = next((r.type for r in p.relations if r.arg1==start and r.arg2==end),'none')

					self.origins.append((a, start, end))

					xs.append(pred_rel)
					ys.append(orig_rel)

		self.x = self.onehot_relations.transform(xs)
		self.y = labels.transform(ys)
		self.x_labels = xs
		self.y_labels = ys

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x,y = self.x[idx], self.y[idx]

		x,y = torch.tensor(x).float(), torch.tensor(y).long()

		return x,y

	def collate_fn(self, sequences):
		xs,ys = zip(*sequences)
		if self.cuda:
			return torch.stack(xs,1).cuda(), torch.stack(ys,0).cuda()
		else:
			return torch.stack(xs,1), torch.stack(ys,0)


if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	import warnings
	from sklearn.exceptions import UndefinedMetricWarning
	warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

	import os

	from utilities.argparseactions import ArgumentParser,IterFilesAction,DirectoryAction,ListAction
	from annotations.bratutils import StandoffConfigAction
	from utilities.mlutils import split_data

	from annotations.annmlutils import open_anns
	from annotations.models.training import evaluate_relations

	def parse_tuple(s):
		return tuple(int(i) for i in s.split('-'))

	parser = ArgumentParser(description='Predict (or evaluate) using rules-based model for relation classification.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('modeldir',action=DirectoryAction,mustexist=False,mkdirs=True,help='Directory to use when saving outputs.')
	parser.add_argument('config',action=StandoffConfigAction,help='Standoff config file for these annotations.')
	parser.add_argument('-a','--all',action='store_true',help='Use all data (rather than just test set) for evaluation.')
	parser.add_argument('--seed', type=int, default=42, help='Random number seed for this evaluation run (ignored if --all flag set).')
	parser.add_argument('--split', type=parse_tuple, default=(0.6,0.2,0.2), help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')
	parser.add_argument('--types',action=ListAction, help='Annotation types to consider.')
	args = parser.parse_args()

	modelname = os.path.basename(args.modeldir)

	# Read in data
	if not args.types:
		args.types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName', 'Definition','Confidence','Measurement','Name','Property','Defined']
	#types = None
	anns = open_anns(args.ann,types=args.types,use_labelled=True)

	stopwatch.tick('Opened all files',report=True)

	output_labels = list(set(r.type for a in anns for r in a.relations))+['none']

	model = RulesRelationModel(output_labels, args.config.relations)
	model.save(os.path.join(args.modeldir,modelname+'.pt'))

	if args.all:
		class_metrics,overall_metrics = evaluate_relations(model, anns, args.config.relations)
	else:
		# Make test/train split
		ann_train,ann_dev,ann_test = split_data(anns, args.split, random_state=args.seed)

		class_metrics,overall_metrics = evaluate_relations(model, ann_test, args.config.relations)

	class_metrics.to_csv(os.path.join(args.modeldir,'class_metrics.csv'))
	overall_metrics.to_csv(os.path.join(args.modeldir,'overall_metrics.csv'))

	stopwatch.tick('Finished evaluation')

	stopwatch.report()
