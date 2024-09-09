if __name__ == '__main__':

	import re

	from gaml.utilities.argparseactions import ArgumentParser,FileAction
	from gaml.metadata.oaipmh import MetadataAction,arXivID_from_path

	from gaml.utilities import StopWatch
	stopwatch = StopWatch()

	parser = ArgumentParser(description='Print titles of arXiv papers from file of filepaths.')
	parser.add_argument('source',action=FileAction,mustexist=True,help='File of arXiv paths.')
	parser.add_argument('metadata',action=MetadataAction,help='ArXiv metadata.')
	args = parser.parse_args()

	with open(args.source,'r') as paths:
		for line in paths:
			arXiv = arXivID_from_path(line)
			title = re.sub("\s+"," ",args.metadata.get(arXiv,"title"))
			print(f'{arXiv}: {title}\n')

	stopwatch.report()
