if __name__ == '__main__':

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	from gaml.utilities.fileutilities import changeext
	from gaml.utilities.bratutils import read_ann
	import json
	import re
	import itertools
	import lxml.etree as let

	from collections import defaultdict

	from pprint import pprint

	from gaml.metadata.oaipmh import arXivID_from_path

	from gaml.preprocessing.latexmlpy import get_local_overlap_from_multiple

	parser = ArgumentParser(description='Description.')
	parser.add_argument('source',action=IterFilesAction,mustexist=True,suffix='.ann',help='Source file or directory.')
	parser.add_argument('results',action=FileAction,mustexist=True,help='JSON results of keywordsearch.')
	args = parser.parse_args()

	with open(args.results,'r') as f:
		results = json.load(f)

	ground_truth = defaultdict(list)
	allowed_spans = defaultdict(list)

	all_entries = defaultdict(list)

	for ann in args.source:

		arXiv = arXivID_from_path(ann)

		annotations = read_ann(ann)
		with open(changeext(ann,'.spans'),'r') as f:
			spans = json.load(f)
		allowed_spans[arXiv] += [(s[2],s[3]) for s in spans]

		article = let.parse(changeext(ann,'.xml'))

		#pprint(annotations.entities)
		#pprint(annotations.relations)

		#pprint(spans)

		for measurement in annotations.get_relations(tag='Measurement'):
			overlap = get_local_overlap_from_multiple(annotations[measurement['arg2']]['span'],spans)
			#print(f'Overlap: {overlap}')
			#print(article.xpath(overlap[0][2])[0])
			#print(getattr(article.xpath(overlap[0][2])[0],overlap[0][3]))

			#print(' '.join([getattr(article.xpath(o[2])[0],o[3])[o[0]:o[1]] for o in overlap]))

			entry = {
					'arXiv': arXiv,
					'xml': changeext(ann,'.xml'),
					'number_spans': annotations[measurement['arg2']]['span'],
					'number_text': annotations[measurement['arg2']]['text'],
					'element_paths': [o[2] for o in overlap],
					'element_spans': [(o[0],o[1]) for o in overlap],
					'origins': overlap,
					annotations[measurement['arg1']]['tag']: annotations[measurement['arg1']]['text'],
					annotations[measurement['arg2']]['tag']: annotations[measurement['arg2']]['text'],
					'measurement': annotations[measurement['arg2']]['text']
				}

			if ('Parameter' in entry and re.match('H _ \{.*(?:0|o|\\\\circ).*\}|h',entry['Parameter'])) or ('ParameterName' in entry and 'hubble' in entry['ParameterName'].lower()):
				#print(f'Overlap: {overlap}')
				#print('Entry:')
				#pprint(entry,indent=4)
				#for o in overlap:
				#	print(f'{o[0]} {o[1]} {getattr(article.xpath(o[2])[0],o[3])}')
				xml_text = ' '.join([getattr(article.xpath(o[2])[0],o[3])[o[0]:o[1]] for o in overlap])
				if xml_text != entry['number_text']:
					print('ERROR==============================================================')

				#ground_truth[entry['arXiv']].append(entry)
				ground_truth[entry['arXiv']].append(entry['measurement'])

			all_entries[arXiv].append(entry)

		if not ground_truth[arXiv]:
			print(f'No entries for {arXiv}')

	#pprint(ground_truth)

	abstract_results = defaultdict(list)

	accepted_counter = 0
	total_counter = 0
	for found_value in itertools.chain.from_iterable(results.values()):
		total_counter += 1
		if all((origin[2],origin[3]) in allowed_spans[found_value['identifier']] for origin in found_value['origins']):
			#abstract_results[found_value['identifier']].append(found_value)
			abstract_results[found_value['identifier']].append(found_value['value_text'])
			accepted_counter += 1

	#pprint(abstract_results)


	true_positive = 0
	false_positive = 0
	false_negative = 0
	for arXiv in set(ground_truth.keys()) | set(abstract_results.keys()):
		y = set(abstract_results[arXiv])
		t = set(ground_truth[arXiv])
		#print(f'{str(sorted(y)):40} {str(sorted(t)):30}')
		true_positive += len(y&t)
		false_positive += len(y-t)
		false_negative += len(t-y)

		if len(t) == 0:
			print(f'No results for: {arXiv}')
			pprint(all_entries[arXiv])

	print(f'TP = {true_positive}\nFP = {false_positive}\nFN = {false_negative}')
	precision = true_positive/(true_positive + false_positive)
	recall = true_positive/(true_positive + false_negative)
	f1 = 2 * (precision * recall) / (precision + recall)
	print(f'Precision = {precision:.2}\nRecall = {recall:.2}\nF1 = {f1:.2}')
