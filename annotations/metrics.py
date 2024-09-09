from sklearn.preprocessing import LabelEncoder
import statistics

from annotations.brattowindow import StandoffLabels

def get_type(s):
	if s == 'outside':
		return 0
	elif s.endswith('_inside'):
		return 2
	else: # s.endswith(begin)
		return 1

def get_ent(s):
	if s == 'outside':
		return 'outside'
	elif s.endswith('_inside'):
		return s[:-7]
	else: # s.endswith('_begin')
		return s[:-6]

def get_state_mapping(labels):
	mapping = {l:i for i, l in enumerate(labels.classes_)}
	ent_labels = LabelEncoder().fit([get_ent(l) for l in labels.classes_])
	ent_mapping = {l:i for i, l in enumerate(ent_labels.classes_)}

	bio_dict = {i:get_type(l) for l,i in mapping.items()} # Dict mapping encodings to BIO type

	ent_dict = {i:ent_mapping[get_ent(l)] for l,i in mapping.items()} # Dict mapping encodings to Entity types

	def get_state(i):
		return ent_dict[i],bio_dict[i] # Returns integer (ent_index,type_index)

	return get_state,ent_labels

def ann_entity_overlap_score(output, target, strict, types=None):
	''' Output and target must be given as lists of Standoff objects. '''

	if len(output) != len(target):
		raise ValueError(f'Output ({len(output)}) and target ({len(target)}) list lengths must match.')

	# 6 Possible cases for entity alignment (ignoring entity type):
	## Case 1: 1-1 mapping between target and output entities.
	##		Case 1.1: Full overlap between entities. Return 1.
	##		Case 1.2: Partial overlap between entities. Return 0 if strict, otherwise Jaccart overlap betwen tokens
	## Case 2: 1-many overlap between target and output entities. Return 0.
	## Case 3: many-1 overlap between target and output entities. Return 0.
	## Case 4: 1-0 mapping between target and output entities. Return 0.
	## Case 5: 0-1 mapping between target and output entities. Return 0.
	## Case 6: many-many mapping between target and output entities. This will be treated as multiple instances of Cases 2 and 3
	# We then take an average over these events.
	# Note that this implies we do need to identify all cases, as zeros also contribute to the final score.

	#idxs = {l:i for i, l in enumerate(labels.classes_)}

	## Cannot account for True Negative
	#basic_confusion = numpy.zeros((2,2))
	#confusion = numpy.zeros(len(labels.classes_,)*2)

	def jaccart(idxs1,idxs2):
		(s1,e1),(s2,e2) = idxs1,idxs2
		s,e = min(s1,s2),max(e1,e2)
		i,j = max(s1,s2),min(e1,e2)
		return max(0,j-i)/(e-s)

	results = []

	for o_ann,t_ann in zip(output,target):

		if not strict:
			o_labelled = StandoffLabels(o_ann,types=types,include_bio=False)
			t_labelled = StandoffLabels(t_ann,types=types,include_bio=False)

		o_ents = o_ann.entities if types is None else [e for e in o_ann.entities if e.type in types]
		t_ents = t_ann.entities if types is None else [e for e in t_ann.entities if e.type in types]

		## Target Entities (ground truth)
		for e in t_ents:
			overlapping = o_ann.get_annotations(e.start,e.end)
			if types is not None: overlapping = [i for i in overlapping if i.type in types]
			if len(overlapping)==1:
				reverse_overlap = t_ann.get_annotations(overlapping[0].start,overlapping[0].end)
				if types is not None: reverse_overlap = [i for i in reverse_overlap if i.type in types]
				if len(reverse_overlap)==1:
					## Case 1
					if overlapping[0].start==e.start and overlapping[0].end==e.end: # Exact match
						## Case 1.1
						results.append(1 if overlapping[0].type==e.type else 0)
					else: # Partial overlap
						## Case 1.2
						if overlapping[0].type==e.type:
							results.append(0 if strict else jaccart(t_labelled.get_token_idxs(e),o_labelled.get_token_idxs(overlapping[0])))
						else:
							results.append(0)
				else:
					## Case 3
					pass ## Dealt with below to avoid repetition
			else: # This includes full overlaps of the wrong type
				if len(overlapping)>1:
					## Case 2
					results.append(0)
				else: # len(overlapping)==0
					## Case 4
					results.append(0)

		## Output entities (predictions)
		for e in o_ents:
			overlapping = t_ann.get_annotations(e.start,e.end)
			if types is not None: overlapping = [i for i in overlapping if i.type in types]
			if len(overlapping)==1:
				reverse_overlap = o_ann.get_annotations(overlapping[0].start,overlapping[0].end)
				if types is not None: reverse_overlap = [i for i in reverse_overlap if i.type in types]
				if len(reverse_overlap)==1:
					## Case 1
					pass # Accounted for above
				else:
					## Case 2
					pass ## Dealt with above to avoid repetition
			else: # This includes partial overlaps
				if len(overlapping)>1:
					## Case 3
					results.append(0)
				else: # len(overlapping)==0
					## Case 5
					results.append(0)

	return statistics.mean(results)
