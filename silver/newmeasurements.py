from gaml.metadata.oaipmh import arXivID_from_path

def process_paper(path, results, metadata):
	text = ''
	for event,p in fastxmliter(path, events=("end",), tag='abstract'):
		text += latexml.tostring(p)[0] + '\n'

	if text:
		results['Papers examined'] += 1

		contains_pm = bool(re.search('[0-9]\s*\\\\pm\s*[0-9]',text))

		if re.search('measurement of',args.metadata[arXivID_from_path(path)]['title'],flags=re.IGNORECASE):
			results['measurement_of_title'].append(path)
			if contains_pm:
				results['pm_and_title_measurement_of'].append(path)
		if re.search('measurement',args.metadata[arXivID_from_path(path)]['title'],flags=re.IGNORECASE):
			results['measurement_title'].append(path)
			if contains_pm:
				results['pm_and_title_measurement'].append(path)
		if re.search('=\s*[0-9]',text):
			results['equals_sign'].append(path)
			if contains_pm:
				results['pm_and_equals'].append(path)
		if contains_pm: results['pm'].append(path)
		if re.search('we\s+\w+\s+(?:a|the)\s+value',text,flags=re.IGNORECASE): results['we_X_a_value'].append(path)
		if re.search('yields a value of',text,flags=re.IGNORECASE): results['yield_value'].append(path)
		if re.search('(?:measurement|statistical|systematic|random)\s+error',text,flags=re.IGNORECASE):
			results['error'].append(path)
			if contains_pm:
				results['pm_and_error'].append(path)
		if re.search('we\s+measure',text,flags=re.IGNORECASE): results['we_measure'].append(path)
		if re.search('(?:these|our)\s+results',text,flags=re.IGNORECASE): results['our_results'].append(path)
		if re.search('CL|C\.L\.',text): results['CL'].append(path)


if __name__ == '__main__':

	import re
	from pprint import pprint

	from gaml.utilities.parallel import parallel_results
	from gaml.utilities.iterutilities import randomly

	from gaml.preprocessing import latexmlpy as latexml
	from gaml.utilities.gxml import fastxmliter

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	from gaml.metadata.oaipmh import MetadataAction
	from gaml.utilities.fileutilities import addsuffix

	from gaml.utilities import StopWatch
	stopwatch = StopWatch()

	parser = ArgumentParser(description='Create silver data labels for papers containing new measurements.')
	parser.add_argument('source',action=IterFilesAction,mustexist=True,recursive=True,suffix='.xml',help='ArXiv XML source files.')
	parser.add_argument('metadata',action=MetadataAction,help='ArXiv metadata.')
	parser.add_argument('-f','--file',action=FileAction,mustexist=False,default='newmeasurements.txt')
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=100)
	args = parser.parse_args()

	results = parallel_results(process_paper, args.source, additional_args=(args.metadata,), chunksize=args.chunksize, processes=args.processes)

	pprint(results)

	counts = {key: (len(value) if isinstance(value,list) else value) for key,value in results.items()}
	pprint(counts)
	pprint({key: value/counts['Papers examined'] for key,value in counts.items()})

	with open(args.file,'w') as f:
		for p in results['pm_and_title_measurement_of']:
			f.write(p+'\n')
	with open(addsuffix(args.file,'_sample'),'w') as f:
		for i,p in zip(range(20),randomly(results['pm_and_title_measurement_of'])):
			f.write(p+'\n')

	stopwatch.report()
