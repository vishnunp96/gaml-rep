if __name__ == '__main__':

	import itertools
	import os
	import re
	#from copy import deepcopy
	from collections import defaultdict

	import warnings
	from sklearn.exceptions import UndefinedMetricWarning
	warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

	from utilities.argparseactions import ArgumentParser,DirectoryAction,IterFilesAction

	from annotations.bratnormalisation import open_clean
	from annotations.bratutils import Standoff

	from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,f1_score

	parser = ArgumentParser(description='Predict (or evaluate) using rules-based model for relation classification.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	#parser.add_argument('config',action=FileAction,help='Brat annotation.conf file for this dataset.')
	parser.add_argument('-o','--output',action=DirectoryAction, mustexist=False, mkdirs=True,help='Directory in which to output predictions.')
	parser.add_argument('--eval',action='store_true',help='Flag to indicate that incoming files should be used for evaluation rather than prediction.')
	parser.add_argument('-v','--verbose',nargs='?',type=int,default=1,const=1,help='Flag to indicate that incoming files should be used for evaluation rather than prediction.')
	args = parser.parse_args()

	# Read in data
	entity_types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Definition']
	relation_types = ['Measurement','Confidence','Name','Property','Defined']
	anns = []
	for path in args.ann:
		try:
			#anns.append(StandoffLabels.open(path,types=types,include_bio=True))
			standoff = open_clean(path,check_repetitions=('ParameterName','ParameterSymbol','ObjectName'))
			# Convert Constraints to MeasuredValues (we can recover Constraints using Attributes)
			for c in [e for e in standoff.entities if e.type=='Constraint']:
				c.type = 'MeasuredValue'
			if standoff:
				#anns.append(StandoffLabels(standoff,types=entity_types,include_bio=True))
				anns.append((os.path.basename(path),standoff))
		except KeyError:
			print(f'Error opening {path}')

	def pairwise(iterable):
		''' s -> (s0,s1), (s1,s2), (s2, s3), ... '''
		a, b = itertools.tee(iterable)
		next(b, None)
		return zip(a, b)

	if args.eval:
		ground_truth = []
		predictions = []

	disallowed = defaultdict(list)
	def disallow_relation(ann,rel_type,arg1,arg2):
		disallowed[(arg1,arg2)].append(rel_type)
		existing = ann.get_relation(arg1,arg2)
		if existing and existing.type==rel_type:
			ann.relations.remove(existing)
	def found_relation(ann,rel_type,arg1,arg2):
		if not ann.get_relation(arg1,arg2) and not disallowed[(arg1,arg2)]:
			ann.relation(rel_type,arg1,arg2)

	# (RelationType, Ent1, Ent2, (Spans), Reversed)
	simple_relations = [
			('Measurement', 'ParameterSymbol', 'MeasuredValue', ('=','>','<','\\sim','\\simeq','\\approx','\\leq','of',''), False),
			('Measurement', 'MeasuredValue', 'ParameterSymbol', ('>','<','\\leq'), True),
			('Measurement', 'ParameterName', 'MeasuredValue', ('is', 'of','(',''), False),
			('Confidence', 'MeasuredValue', 'ConfidenceLimit', ('(','at','at the'), False),
			('Name', 'ParameterName', 'ParameterSymbol', ('is', 'of', ',','(',''), False),
			('Name', 'ParameterSymbol', 'ParameterName', ('is the', 'is'), True),
			('Property', 'ParameterName', 'ObjectName', ('of', 'of the'), True),
			('Property', 'ObjectName', 'ParameterName', ('',), False),
			('Defined', 'ParameterSymbol', 'Definition', ('is', '=', '\\equiv'), False),
			('Defined', 'ParameterName', 'Definition', ('='), False),
		]

	entity_map = {
			'MeasuredValue': 'V',
			'Constraint': 'C',
			'ParameterSymbol': 'S',
			'ParameterName': 'N',
			'ConfidenceLimit': 'L',
			'ObjectName': 'O',
			'Definition': 'D'
		}

	simplenamere = re.compile(r'(?<!S)NS')
	multiplemeasurementsre = re.compile('(S|N)(V|C)+')
	confidencelimitre = re.compile('(V|C)+L')
	definitionmeasurementre = re.compile(r'SDV')
	standardmeasurementre = re.compile(r'NS(V|C)')
	nameobjectpropertyre = re.compile(r'NOS(V|C)')
	simplepropertyre = re.compile(r'O(NS|N|S)(V|C)')
	tuplemeasurementre = re.compile(r'(?P<names>(N|S)+)(?P<values>(V|C)+)')

	for filename,original_ann in anns:

		if args.eval:
			ann = Standoff(original_ann.text,original_ann.entities,[],[],[])
		else:
			ann = original_ann

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
				for rel_type,start_t,end_t,connections,reverse in simple_relations:
					if e1.type==start_t and e2.type==end_t:
						if span in connections:
							if reverse:
								found_relation(ann,rel_type,e2,e1)
							else:
								found_relation(ann,rel_type,e1,e2)

			line_accepted = [e for e in line if e.type in entity_map]
			pattern = ''.join(entity_map[e.type] for e in line_accepted)

			for match in simplenamere.finditer(pattern):
				found_relation(ann,'Name',line_accepted[match.start()],line_accepted[match.start()+1])

			for match in multiplemeasurementsre.finditer(pattern):
				for i in range(match.start()+1,match.end()):
					start = line_accepted[match.start()]
					end = line_accepted[i]
					if not ann.get_relation(start,end):
						found_relation(ann,'Measurement',start,end)

			for match in standardmeasurementre.finditer(pattern):
				found_relation(ann,'Name',line_accepted[match.start()],line_accepted[match.start()+1])
				found_relation(ann,'Measurement',line_accepted[match.start()+1],line_accepted[match.start()+2])

			for match in definitionmeasurementre.finditer(pattern):
				found_relation(ann,'Measurement',line_accepted[match.start()],line_accepted[match.start()+2])

			cl_accepted = [e for e in line_accepted if e.type in ('MeasuredValue','Constraint','ConfidenceLimit')]
			cl_pattern = ''.join(entity_map[e.type] for e in cl_accepted)
			for match in confidencelimitre.finditer(cl_pattern):
				for i in range(match.start(),match.end()-1):
					start = cl_accepted[i]
					end = cl_accepted[match.end()-1]
					if not ann.get_relation(start,end):
						found_relation(ann,'Confidence',start,end)

			for match in nameobjectpropertyre.finditer(pattern):
				found_relation(ann,'Property',line_accepted[match.start()+1],line_accepted[match.start()])
				found_relation(ann,'Name',line_accepted[match.start()],line_accepted[match.start()+2])
				disallow_relation(ann,'Property',line_accepted[match.start()+1],line_accepted[match.start()+2])

			for match in simplepropertyre.finditer(pattern):
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

			for match in tuplemeasurementre.finditer(pattern):
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

			#if len([e for e in line_ents[0] if e.type=='ObjectName'])==1 and len([e for e in line_accepted if e.type=='ObjectName'])==0:
			#	# Perhaps linked?
			#	object_name = next(e for e in line_ents[0] if e.type=='ObjectName')
			#	for ent in (e for e in line_accepted if e.type in ('ParameterName',)):
			#		found_relation(ann,'Property',object_name,ent)


		if args.verbose>1:
			print(f'{filename:>22}: {len(ann.entities):4d} entities, {len(ann.relations):4d} relations')

		if args.output:
			ann.save(os.path.join(args.output,filename))

		if args.eval:
			for start,end in itertools.permutations(original_ann.entities,2):
				original_rel = original_ann.get_relation(start,end)
				ground_truth.append(original_rel.type if original_rel and original_rel.type in relation_types else 'none')
				predicted_rel = ann.get_relation(start,end)
				predictions.append(predicted_rel.type if predicted_rel else 'none')
				#if (original_rel and original_rel.type=='Property') or (predicted_rel and predicted_rel.type=='Property'):
				#	print(f'Original rel: {repr(original_rel)}     ->      Predicted rel: {repr(predicted_rel)}')
				#	if original_rel is None:
				#		start = min(predicted_rel.arg1.start, predicted_rel.arg2.start)
				#		end = max(predicted_rel.arg1.end, predicted_rel.arg2.end)
				#		print('\t'+ann.text[start:end])


	if args.eval and args.verbose>0:
		#for t,p in zip(ground_truth,predictions):
		#	print(t,p)

		evaluation_types = sorted(relation_types + ['none'])

		confusion = confusion_matrix(ground_truth, predictions, labels=evaluation_types)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(ground_truth, predictions, labels=evaluation_types)

		print_len = max(len(c) for c in evaluation_types) + 2
		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(evaluation_types):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(ground_truth,predictions,labels=evaluation_types, average='micro')
		f1weighted = f1_score(ground_truth,predictions,labels=evaluation_types, average='weighted')
		f1macro = f1_score(ground_truth,predictions,labels=evaluation_types, average='macro')
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')
