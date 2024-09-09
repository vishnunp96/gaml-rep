from metadata.oaipmh import arXivID_from_path
from utilities.parallel import parallel_results
from preprocessing import latexmlpy as latexml
import re
from parsing import parse_measurement
from keywordsearch.rulesBasedSearch import measurementre
from units import unit
from units.dimensionless import DimensionlessUnit

# "Conclusion" here includes results section
conclusionre = re.compile('conclusions?|summary|results?',flags=re.IGNORECASE)
#titlere = re.compile('measurement|measuring|determination|determining|estimation|value|parameter|constraint',flags=re.IGNORECASE)

dimensionless = DimensionlessUnit(1)
percentage_unit = unit('%')

def conclusion_title(s):
	return bool(conclusionre.search(s))

# Score paper abstract and conclusions
def score_paper(filepath, metadata):

	tree = latexml.openxml(filepath,getroot=False)

	abstracttext = ''
	for a in tree.findall('//abstract'):
		abstracttext += latexml.tostring(a)[0] + '\n'

	## Must have abstract to continue
	if len(abstracttext) > 8:
		conclusion_paths = [
				tree.getpath(i) + '/..'
				for i in tree.findall('//title')
				if
					conclusion_title(latexml.tostring(i)[0]) and
					i.getparent().tag == 'section']

		conclusiontext = ''
		for path in conclusion_paths:
			for elem in tree.xpath(path):
				conclusiontext += latexml.tostring(elem)[0] + '\n'

		abstract_m = get_measurements(abstracttext)
		conc_m = get_measurements(conclusiontext)

		#result = max(len(m.uncertainties) if m else 0 for m in measurements)
		#result = len(measurements)
		return {
				'abstract': len(abstract_m),
				'conclusion': len(conc_m),
				'n_conclusion_sections': len(conclusion_paths),
				'abstract_units': len([m for m in abstract_m if m.unit is not None and m.unit.canonical()!=dimensionless and m.unit!=percentage_unit]),
				'conclusion_units': len([m for m in conc_m if m.unit is not None and m.unit.canonical()!=dimensionless and m.unit!=percentage_unit])
			}

	return None

def get_measurements(text):
	measurements = [(try_parse_measurement(m.group(0)),m) for m in measurementre.finditer(text)]
	measurements = [(p,m) for p,m in measurements if p and p.uncertainties]
	#measurements = [(p,m) for p,m in measurements if (p.unit==percentage_unit if p.unit else True)] ## This is wrong, right? Should be !=?
	measurements = measurements if not all(m.group('range') for p,m in measurements) else []
	return [p for p,m in measurements]


def try_parse_measurement(text):
	try:
		return parse_measurement(text)
	except OverflowError:
		print(f'Failed to parse: {text}')
		return None

def _collate_paper_score(path, results, metadata):
	arXiv = arXivID_from_path(path)
	scores = score_paper(path,metadata)
	if scores is not None: results[arXiv] = scores

def get_scores(sources, metadata, processes=1, chunksize=1000):
	return parallel_results(_collate_paper_score, sources, additional_args=(metadata,), results_type=resultsdict, chunksize=chunksize, processes=processes)

class resultsdict(dict):
	def append(self, other):
		for key,value in other.items():
			self[key] = value

if __name__ == '__main__':

	import pandas

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	from metadata.oaipmh import MetadataAction

	from utilities import StopWatch
	stopwatch = StopWatch()

	parser = ArgumentParser(description='Create silver data labels for papers containing new measurements.')
	parser.add_argument('source',action=IterFilesAction,recursive=True,suffix='.xml',help='ArXiv XML source files.')
	parser.add_argument('metadata',action=MetadataAction,help='ArXiv metadata.')
	parser.add_argument('output',action=FileAction,mustexist=False,help='File in which to store output scores.')
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=100)
	args = parser.parse_args()

	results = get_scores(args.source, args.metadata, chunksize=args.chunksize, processes=args.processes)

	data = pandas.DataFrame.from_dict(results,orient='index')
	data.index.name = "id"

	data['total'] = data['abstract'] + data['conclusion']

	data.to_csv(args.output)

	stopwatch.report()
