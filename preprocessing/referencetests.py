if __name__ == '__main__':

	from gaml.metadata.oaipmh import arXivID_from_path
	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction

	import gaml.preprocessing.latexmlpy as latexml

	parser = ArgumentParser(description="Find occurances of tags.")
	parser.add_argument("sources",action=IterFilesAction, recursive=True, suffix='.xml', help='Path to xml source(s).')
	parser.add_argument("-S","--summary",action='store_true',help='Provide summary rather than iterate.')
	args = parser.parse_args()

	if not args.summary:
		for path in args.sources:
			for abstract in latexml.elemiter(path,'abstract'):
				try:
					if any(e is not None for e in abstract.findall('.//cite')):
						print(latexml.format(abstract).encode('ascii',errors='replace').decode('ascii'))
						print(path)
						input()
				except UnicodeEncodeError as e:
					pass
	else:
		papers = set()
		paperssearched = 0
		errors = 0
		for path in args.sources:
			paperssearched += 1
			arXiv = arXivID_from_path(path)
			for abstract in latexml.elemiter(path,'abstract'):
				try:
					if any(e is not None for e in abstract.findall('.//cite')):
						papers.add(arXiv)
				except UnicodeEncodeError as e:
					errors += 1

	print(f'Citations in abstracts of {len(papers)} papers, from {paperssearched} searched ({100*len(papers)/paperssearched:.1f}%). {errors} errors.')
