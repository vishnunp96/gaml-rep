from utilities.bratutils import read_ann

if __name__ == '__main__':

	from utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction
	from utilities.fileutilities import addsuffix

	from pprint import pprint

	from metadata.oaipmh import arXivID_from_path
	import pandas

	parser = ArgumentParser(description='Description.')
	parser.add_argument('source',action=IterFilesAction,suffix='.ann',help='Source file or directory.')
	parser.add_argument('-d','--data',action=FileAction,mustexist=False,help='Path at which to save measurements.')
	args = parser.parse_args()

	measurements = pandas.DataFrame()

	for ann in args.source:

		arXiv = arXivID_from_path(ann)

		annotations = read_ann(ann)

		pprint(annotations.entities)
		pprint(annotations.relations)

		for measurement in annotations.get_relations(tag='Measurement'):
			entry = {
					'arXiv': arXiv,
					annotations[measurement['arg1']]['tag']: annotations[measurement['arg1']]['text'],
					annotations[measurement['arg2']]['tag']: annotations[measurement['arg2']]['text']
				}

			if 'Parameter' in entry:
				for parametername in annotations.get_relations(tag='Name',arg1=measurement['arg1']):
					entry[annotations[parametername['arg2']]['tag']] = annotations[parametername['arg2']]['text']
					break

			for measurement_detail in annotations.get_relations(arg1=measurement['arg2']):
				entry[annotations[measurement_detail['arg2']]['tag']] = annotations[measurement_detail['arg2']]['text']

			pprint(entry)

			measurements = measurements.append(entry, ignore_index=True)








	import shutil
	with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', shutil.get_terminal_size().columns, 'display.height', shutil.get_terminal_size().lines*3):
		print(measurements)
		print(measurements.columns)


	if args.data:
		measurements.to_csv(args.data, index=False)

		#for p in sorted(measurements['Parameter'].dropna().unique()):
		#	print(p)

		hubble = measurements[(measurements['Parameter'].str.match('H _ \{.*(?:0|o|\\\\circ).*\}|h',na=False)) | (measurements['ParameterName'].fillna('').str.lower().str.contains('hubble'))]
		#print(hubble)

		hubble = hubble.dropna(axis=1,how='all')

		hubble.to_csv(addsuffix(args.data,'hubble'), index=False)
