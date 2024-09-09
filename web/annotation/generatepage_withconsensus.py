if __name__ == '__main__':

	import os
	from collections import defaultdict
	from datetime import datetime
	from statistics import mean

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction,IterFilesAction

	from gaml.annotations.bratnormalisation import open_clean, similarity, compare_annotations
	from gaml.metadata.oaipmh import arXivID

	import dominate
	from dominate.tags import div,link,h1,h2,table,thead,tbody,tr,th,td,a

	parser = ArgumentParser(description='Generate webpage for annotation sample from annotators listings.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	parser.add_argument('output',action=FileAction,mustexist=False,help='Output location for HTML file.')
	parser.add_argument('consensus',action=DirectoryAction,mustexist=True,help='Directory in which to store consensus files. Must exist before script is called.')
	parser.add_argument('name',help='Sample name.')
	parser.add_argument('-t','--tick',action='store_true',help='Flag to print time when code is run.')
	args = parser.parse_args()

	doc = dominate.document(title='Numerical Atlas Annotation Project')

	#ignore = ('Reference','Details','Definition', # Entities
	#		'Condition','Source','Defined',) # Relations
	ignore = ('Reference','Details', # Entities
			'Source') # Relations

	arXiv_listings = defaultdict(list)
	annotator_listings = defaultdict(list)

	for path in (p for p in args.ann if args.name == os.path.basename(os.path.dirname(p))):
		ann = open_clean(path)

		arXiv = arXivID(os.path.basename(os.path.splitext(path)[0]))
		ann.arXiv = arXiv

		arXiv_listings[arXiv].append(ann)
		annotator_listings[ann.annotator].append(ann)

	annotators = defaultdict(lambda: defaultdict(list))
	for name,anns in annotator_listings.items():
		for ann in anns:
			arXiv = ann.arXiv
			abstracts = arXiv_listings[arXiv]
			annotators[name][arXiv].append(ann)
			for other in sorted([a for a in abstracts if a is not ann],key=lambda s: s.annotator):
				annotators[name][arXiv].append(other)

	consensus = {}
	similarities = {}
	consensus_similarities = defaultdict(dict)
	for arXiv,anns in arXiv_listings.items():
		consensus[arXiv] = compare_annotations(anns,ignore)
		consensus[arXiv].save(os.path.join(args.consensus,arXiv.replace('/',''))+'.ann')

		similarities[arXiv] = similarity(ignore,anns)
		for ann in anns:
			consensus_similarities[ann.annotator][arXiv] = similarity(ignore,[consensus[arXiv],ann])

	mean_similarity = mean(sim for arXiv,sim in similarities.items() if any(arXiv_listings[arXiv]))
	#mean_consensus_similarity = mean(mean(d.values()) for d in consensus_similarities.values())
	mean_consensus_similarity = mean(mean(sim for arXiv,sim in d.items() if any(arXiv_listings[arXiv])) for d in consensus_similarities.values())

	print(f'Sample: {args.name}')
	print(f'Average similarity between annotators: {mean_similarity:.2f}')
	print(f'Average similarity to consensus: {mean_consensus_similarity:.2f}')

	num_annotators = max(len(anns) for anns in arXiv_listings.values())

	with doc.head:
		link(rel="stylesheet",
				href="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css",
				integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS",
				crossorigin="anonymous")

	def arXiv_link(arXiv_id):
		return 'https://arxiv.org/abs/' + arXiv_id

	def numericalatlas_link(annotator,arXiv_id):
		return 'http://numericalatlas.cs.ucl.ac.uk/annotation/' + annotator + '/#/' + args.name + '/' + arXiv_id.replace('/','')

	def consensus_link(arXiv_id):
		return 'http://numericalatlas.cs.ucl.ac.uk/consensus/#/' + arXiv_id.replace('/','')

	with doc:
		with div(cls='container'):
			div(div(h1('Numerical Atlas'),cls='col-12'),cls='row')
			div(div('Listings for: ' + args.name,cls='col-12 pl-5 lead'),cls='row')
			div(div(f'Average similarity between annotators: {mean_similarity:.2f}',cls='col-12 pl-5 lead'),cls='row')
			div(div(f'Average similarity to consensus: {mean_consensus_similarity:.2f}',cls='col-12 pl-5 lead'),cls='row')

			for name in sorted(annotators):
				papers = annotators[name]
				with div(cls='row mt-5'):
					div(cls='row').add(h2(name,id=name))
				annotators_similarity = mean(similarities[arXiv] for arXiv in papers.keys() if any(arXiv_listings[arXiv]))
				annotator_consensus_similarity = mean(sim for arXiv,sim in consensus_similarities[name].items() if any(arXiv_listings[arXiv]))
				div(div(f'Average similarity to other annotators: {annotators_similarity:.2f}',cls='col-12 pl-5 lead'),cls='row')
				div(div(f'Average similarity to consensus: {annotator_consensus_similarity:.2f}',cls='col-12 pl-5 lead'),cls='row')
				with div(cls='row mt-3'):
					with table(cls='table table-hover'):
						with thead(cls="thead-light").add(tr()):
							th('arXiv',scope="col")
							for i in range(num_annotators):
								th(f'Annotator {i+1}',scope="col")
							th(f'Similarity',scope="col")
							th(f'Consensus',scope="col")
						with tbody():
							for arXiv in sorted(papers):
								anns = papers[arXiv]
								with tr():
									td().add(a(arXiv,
											href=arXiv_link(arXiv)))
									#td().add(a(name,
									#		href=numericalatlas_link(name,arXiv)))
									for ann in anns:
										td().add(a(f'{ann.annotator}{"" if ann else "*"}',
												href=numericalatlas_link(ann.annotator,arXiv)))
									td(f'{similarity(ignore,anns):4.2f}' if any(anns) else 'Empty')
									td().add(a(f'{similarity(ignore,[consensus[arXiv],anns[0]]):4.2f}' if consensus[arXiv] else 'Empty',
											href=consensus_link(arXiv)))

	with open(args.output,'w') as f:
		f.write(str(doc))

	if args.tick:
		print(datetime.now().strftime('Generated at %d-%m-%Y %H:%M:%S'))
