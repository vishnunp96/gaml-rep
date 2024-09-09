from utilities.bratutils import read_ann

if __name__ == '__main__':

	from utilities.argparseactions import ArgumentParser,IterFilesAction
	from metadata import MetadataAction
	from utilities.fileutilities import changeext
	from utilities.jsonutils import load_json
	import re
	import lxml.etree as let

	from collections import defaultdict

	from pprint import pprint

	from metadata.oaipmh import arXivID_from_path

	from preprocessing.latexmlpy import get_local_overlap_from_multiple

	parser = ArgumentParser(description='Description.')
	parser.add_argument('source',action=IterFilesAction,mustexist=True,suffix='.ann',help='Source file or directory.')
	parser.add_argument('metadata',action=MetadataAction,help='Path to ArXiv metadata.')
	args = parser.parse_args()

	ground_truth = defaultdict(list)
	allowed_spans = defaultdict(list)
	all_entries = defaultdict(list)

	measurement_statistics = defaultdict(int)
	measurement_statistics['Parameter_expressions'] = defaultdict(int)
	measurement_statistics['ParameterName_expressions'] = defaultdict(int)
	measurement_statistics['parameter_and_name_expressions'] = defaultdict(int)
	paper_statistics = defaultdict(int)

	for ann in args.source:

		arXiv = arXivID_from_path(ann)

		annotations = read_ann(ann)
		spans = load_json(changeext(ann,'.spans'))
		allowed_spans[arXiv] += [(s[2],s[3]) for s in spans]

		article = let.parse(changeext(ann,'.xml'))

		hubble_count = 0
		for measurement in annotations.get_relations(tag='Measurement'):
			overlap = get_local_overlap_from_multiple(annotations[measurement['arg2']]['span'],spans)

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

				hubble_count += 1

				xml_text = ' '.join([getattr(article.xpath(o[2])[0],o[3])[o[0]:o[1]] for o in overlap])
				if xml_text != entry['number_text']:
					print('ERROR==============================================================')

				ground_truth[entry['arXiv']].append(entry)

				## Store:
				## Number of measurements
				measurement_statistics['num_measurements'] += 1
				## Parameter or ParameterName, and ways of expressing
				if 'Parameter' in entry:
					measurement_statistics['Parameter_given'] += 1
					measurement_statistics['Parameter_expressions'][entry['Parameter']] += 1
				elif 'ParameterName' in entry:
					measurement_statistics['ParameterName_given'] +=1
					measurement_statistics['ParameterName_expressions'][entry['ParameterName']] += 1
				else:
					measurement_statistics['other_given'] += 1
				## Parameter also has reference to ParameterName
				if 'Parameter' in entry:
					name = annotations.get_relations(tag='Name',arg1=measurement['arg1'])
					if name:
						measurement_statistics['parameter_and_name'] += 1
						measurement_statistics['parameter_and_name_expressions'][(annotations[name[0]['arg2']]['text'],entry['Parameter'])] += 1
				## Measurement type
				measurement_statistics['measurement_'+annotations[measurement['arg2']]['tag']] += 1
				## Uncertainty types
				for measurement_detail in annotations.get_relations(arg1=measurement['arg2']):
					measurement_statistics['info_'+annotations[measurement_detail['arg2']]['tag']] += 1

			all_entries[arXiv].append(entry)

		## Total number of papers
		paper_statistics['num_papers'] += 1
		## More than one Hubble constant value reported in abstract
		paper_statistics[str(hubble_count)+'_values_reported'] += 1
		if hubble_count > 1:
			paper_statistics['multiple_values'] += 1
		## Is parameter name in abstract?
		if any('hubble' in p['text'].lower() for p in annotations.get_entities(tag='ParameterName')):
			paper_statistics['name_in_abstract'] += 1
		## Is parameter name in title?
		m = re.search('Hubble(?:\'s)? [Cc]onstant|Hubble(?:\'s) [Pp]arameter| h |H_[0Oo]|H_\{[0Oo]\}|H[0Oo]|H_\{\\\\circ\}', re.sub('\s+',' ',args.metadata[arXiv]['title']))
		if m:
			paper_statistics['name_in_title'] += 1


		if not ground_truth[arXiv]:
			print(f'No entries for {arXiv}')


	pprint(measurement_statistics)
	pprint(paper_statistics)

	from datetime import datetime
	startdate = datetime(2016,1,1)
	astro_papers = defaultdict(int)
	all_papers = defaultdict(int)
	astro_proportion = defaultdict(int)
	for md in [m for m in args.metadata.entries() if 'astro-ph' in m['categories'] and m['date'] < startdate]:
		astro_papers[md['date'].year] += 1
	for md in [m for m in args.metadata.entries() if m['date'] < startdate]:
		all_papers[md['date'].year] += 1
	for year in all_papers.keys():
		astro_proportion[year] = astro_papers[year]/all_papers[year]
	pprint(astro_papers)
	pprint(astro_proportion)
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('Year')
	ax.set_ylabel('# Papers')
	#ax.stackplot(sorted(all_papers.keys()),[astro_papers[k] for k in sorted(all_papers.keys())],[all_papers[k]-astro_papers[k] for k in sorted(all_papers.keys())])
	ax.plot(sorted(astro_papers.keys()),[astro_papers[k] for k in sorted(astro_papers.keys())])
	plt.show()
