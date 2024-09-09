if __name__ == '__main__':

	import itertools
	from collections import Counter

	from sklearn.model_selection import train_test_split

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction
	from gaml.annotations.bratutils import StandoffConfigAction

	from gaml.annotations.annmlutils import open_anns

	parser = ArgumentParser(description='Run tests on relations in training set.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('config',action=StandoffConfigAction,help='Standoff config file for these annotations.')
	args = parser.parse_args()

	# Read in data
	types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Confidence','Measurement','Name','Property']
	anns = open_anns(args.ann,types=types,use_labelled=True)

	ann_train,ann_other = train_test_split(anns, test_size=0.2, random_state=42)

	window_pad = 5
	spans = []
	for a in ann_train:
		tokens,tags = zip(*a)
		for start,end in itertools.combinations(a.entities,2):
			rel = next((r for r in a.relations if set([start,end])==set([r.arg1,r.arg2])),None)
			if rel:

				start_s,start_e = a.get_token_idxs(start)
				end_s,end_e = a.get_token_idxs(end)

				pre_window = tokens[(start_s-window_pad):start_s]
				start_tokens = tokens[start_s:start_e]
				between_tokens = tokens[start_e:end_s]
				end_tokens = tokens[end_s:end_e]
				post_window = tokens[end_e:(end_e+window_pad)]

				spans.append({'rel':rel,'pre':pre_window,'start':start,'between':between_tokens,'end':end,'post':post_window})

	min_count = int(0.05*len(ann_train))
	print(f'Min Count: {min_count}')

	for start,end in itertools.product(types,types):
		if args.config.allowed_relation(start,end) or args.config.allowed_relation(end,start):
			names = [' '.join(i['between']) for i in spans if i['start'].type==start and i['end'].type==end]
			print(f'\n{start} -> {end}')
			for span,n in Counter(names).most_common():
				if n > min_count:
					print(f'"{span}" -> {n}')

	entity_map = {
			'MeasuredValue': 'V',
			'Constraint': 'C',
			'ParameterSymbol': 'S',
			'ParameterName': 'N',
			'ConfidenceLimit': 'L',
			'ObjectName': 'O',
			'Definition': 'D'
		}

	window = 0
	for t in types:
		property_patterns = []
		patterns = []
		for a in ann_train:
			pattern = ''.join(entity_map[e.type] for e in a.entities)
			patterns.append(pattern)
			for rel in [r for r in a.relations if r.type==t]:
				start_idx,end_idx = a.entities.index(rel.arg1), a.entities.index(rel.arg2)
				start,end = min(start_idx,end_idx),max(start_idx,end_idx)+1
				for window in [0,1,2,3]:
					span = pattern[max(0,start-window):start].lower() + pattern[start:end] + pattern[end:(end+window)].lower()
					property_patterns.append(span)

		print(f'\n{t} spans:')
		for span,n in Counter(property_patterns).most_common():
			if n > min_count:
				count = sum(i.count(span.upper()) for i in patterns)
				print(f'{span} -> {n} / {count} ({n/count:.2f})')
