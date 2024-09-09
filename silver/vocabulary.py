from utilities.gxml import fastxmliter
import preprocessing.latexmlpy as latexml
from utilities.parallel import parallel_results

import re
import operator

#stop = set(stopwords.words('english') + [',','.'])

numberre = re.compile('(?<=\s)(?:\+|\-)?[0-9]+(?:\.[0-9]+)?(?=\s)')

def read_article(path):
	for event,p in fastxmliter(path, events=("end",), tag='p'):
		text,spans = latexml.tostring(p)
		text = numberre.sub('NUMBER',text)
		## Should we be dealing with lower/upper case here? -> Pontus says no.
		yield text

def _xml_token_count(path,results):
	for text in read_article(path):
		for token in text.split():
			results[token] += 1

def find_vocab(sources, mincount, processes=1, chunksize=1000):
	results = parallel_results(_xml_token_count, sources, chunksize=chunksize, processes=processes)

	vocabulary = {'OOV': 0}
	tokencounter = 1
	for token,count in sorted(results.items(), key=operator.itemgetter(0), reverse=False):
		if count >= mincount:
			vocabulary[token] = tokencounter
			tokencounter += 1

	return vocabulary

def token_vector(path,vocabulary):
	vector_form = []
	for text in read_article(path):
		for token in text.split():
			try:
				vector_form.append(vocabulary[token])
			except KeyError:
				vector_form.append(vocabulary['OOV'])
	return vector_form

if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction

	from utilities.jsonutils import dump_json

	parser = ArgumentParser()
	parser.add_argument('source',action=IterFilesAction, recursive=True, suffix='.xml')
	parser.add_argument('results',action=FileAction,mustexist=False,help='Path at which to store vocabulary.')
	parser.add_argument('-m','--mincount',type=int,default=1)
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=1000)
	args = parser.parse_args()

	vocabulary = find_vocab(args.source, args.mincount, processes=args.processes, chunksize=args.chunksize)

	dump_json(vocabulary,args.results,indent='\t')

	stopwatch.report()

