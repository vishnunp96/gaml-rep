import gaml.metadata.oaipmh as oaipmh
#import gaml.preprocessing.latexmlpy as latexml
from gaml.utilities.parallel import parallel_results
#from gaml.utilities.gxml import fastxmliter

def search_file(filepath, results, keywords, metadata):

	identifier = oaipmh.arXivID_from_path(filepath)
	date = str(metadata.get(identifier,field='date'))

	#for event,elem in fastxmliter(filepath, events=("end",), tag='abstract'):
		#text,span = latexml.tostring(elem)
		#title = metadata[identifier]['title']

	results[filepath] = date

if __name__ == '__main__':

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction
	from gaml.utilities import StopWatch

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Use keyword search to find values of provided keywords in a corpus of LateXML documents.")
	parser.add_argument("sources",action=IterFilesAction, recursive=True, suffix='.xml', help='Path to xml source(s).')
	parser.add_argument("keywords", help='Comma-separated list of keywords to search for.')
	parser.add_argument('metadata',action=oaipmh.MetadataAction,help='Path to metadata.')
	parser.add_argument('-p','--processes',type=int,default=1,help='Number of processes to use. Default to one.')
	parser.add_argument('-c','--chunksize',type=int,default=1000,help='Size of chunks to send to processes. Default to 1000.')
	args = parser.parse_args()

	results = parallel_results(search_file, args.sources, additional_args=(args.keywords,args.metadata), chunksize=args.chunksize, processes=args.processes)

	print(results)

	stopwatch.report()
