from metadata.oaipmh import arXivID_from_path
from utilities.parallel import parallel_results
from preprocessing import latexmlpy as latexml
from utilities.gxml import fastxmliter
import re

def score_paper_abstract(path, metadata):
	arXiv = arXivID_from_path(path)
	text = ''
	for event,p in fastxmliter(path, events=("end",), tag='abstract'):
		text += latexml.tostring(p)[0] + '\n'

	if text:
		result = 0
		if re.search('measurement',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('measurement of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('=\s*[0-9]',text): result += 1
		if re.search('[0-9]\s*\\\\pm\s*[0-9]',text): result += 1
		if re.search('we\s+\w+\s+(?:a|the)\s+value',text,flags=re.IGNORECASE): result += 1
		if re.search('yields a value of',text,flags=re.IGNORECASE): result += 1
		if re.search('(?:measurement|statistical|systematic|random)\s+error',text,flags=re.IGNORECASE): result += 1
		if re.search('we\s+measure',text,flags=re.IGNORECASE): result += 1
		if re.search('(?:these|our)\s+results',text,flags=re.IGNORECASE): result += 1
		if re.search('CL|C\s*\.\s*L\s*\.',text): result += 1
		return result
	else:
		return None

def score_paper_title(path, metadata):
	arXiv = arXivID_from_path(path)
	#text = ''
	#for event,p in fastxmliter(path, events=("end",), tag='abstract'):
	#	text += latexml.tostring(p)[0] + '\n'

	#if text:
	if any(fastxmliter(path, events=("end",), tag='abstract')):
		result = 0
		if re.search('measurement',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('measurement of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		return result
	else:
		return None

def score_paper_title_improved(path, metadata):
	arXiv = arXivID_from_path(path)
	#text = ''
	#for event,p in fastxmliter(path, events=("end",), tag='abstract'):
	#	text += latexml.tostring(p)[0] + '\n'

	#if text:
	if any(fastxmliter(path, events=("end",), tag='abstract')):
		result = 0
		if re.search('measurement',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('measurement of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1

		if re.search('determination',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('determination of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('determining',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('determining the',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('value of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('the value of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		#if re.search('(?:estimation|estimate)',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('(?:estimation|estimate) of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('parameters? from',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('parameter (?:estimation|estimate|constraints?)',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('constraints? on',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('calibrations? of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('measuring',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('measuring the',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		#if re.search('determine',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		#if re.search('to determine',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('re-?determination',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		if re.search('re-?determination of',metadata[arXiv]['title'],flags=re.IGNORECASE): result += 1
		return result
	else:
		return None

from parsing import parse_measurement
from keywordsearch.rulesBasedSearch import measurementre
conclusionre = re.compile('conclusions?|summary',flags=re.IGNORECASE)
titlere = re.compile('measurement|measuring|determination|determining|estimation|value|parameter|constraint',flags=re.IGNORECASE)
def conclusion_title(s):
	return bool(conclusionre.search(s))
def score_paper_conclusion(filepath, metadata):
	arXiv = arXivID_from_path(filepath)

	tree = latexml.openxml(filepath,getroot=False)
	#if tree.find('//abstract') is not None:
	if len(''.join([latexml.tostring(a)[0] for a in tree.findall('//abstract')])) > 8:
		conclusion_paths = [
				tree.getpath(i) + '/..'
				for i in tree.findall('//title')
				if
					conclusion_title(latexml.tostring(i)[0]) and
					i.getparent().tag == 'section']

		if conclusion_paths:
			text = ''
			for path in conclusion_paths:
				for elem in tree.xpath(path):
					text += latexml.tostring(elem)[0] + '\n\n'

			measurements = [parse_measurement(m.group(0)) for m in measurementre.finditer(text)]

			if measurements:
				result = max(len(m.uncertainties) if m else 0 for m in measurements)
				if titlere.search(metadata[arXiv]['title']):
					return result

			return 0 ## Has abstract and conclusion, but no measurements

	return None

def _collate_paper_score(path, results, metadata):
	arXiv = arXivID_from_path(path)
	score = score_paper_conclusion(path,metadata)
	if score is not None: results[arXiv] = score

def get_scores(sources, metadata, processes=1, chunksize=1000):
	return parallel_results(_collate_paper_score, sources, additional_args=(metadata,), chunksize=chunksize, processes=processes)


if __name__ == '__main__':

	import pandas

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	from metadata.oaipmh import MetadataAction

	from utilities import StopWatch
	stopwatch = StopWatch()

	parser = ArgumentParser(description='Create silver data labels for papers containing new measurements.')
	parser.add_argument('source',action=IterFilesAction,recursive=True,suffix='.xml',help='ArXiv XML source files.')
	parser.add_argument('metadata',action=MetadataAction,help='ArXiv metadata.')
	#parser.add_argument('minimum_score',type=int,help='Minimum score required for inclusion in silver data list. Defaults to 3.')
	parser.add_argument('output',action=FileAction,mustexist=False,help='File in which to store output scores.')
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=100)
	#parser.add_argument('-s','--include_score',action='store_true',help='Record score of each paper in output file.')
	args = parser.parse_args()

	results = get_scores(args.source, args.metadata, chunksize=args.chunksize, processes=args.processes)

	#sorted_results = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
	#accepted_results = [arXiv,score for arXiv,score in sorted_results if score>=args.minimum_score]

	data = pandas.DataFrame.from_dict(results,orient='index',columns=['score'])
	data.index.name = "id"
	data.to_csv(args.output)

	stopwatch.report()
