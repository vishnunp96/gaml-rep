if __name__ == "__main__":

	import os
	import re
	import pandas
	from utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction,IterFilesAction
	from utilities import StopWatch

	#import sqlite3
	from annotations.database import query,loadconnect
	from annotations.database import get as get_ann
	from parsing import parse_measurement,parse_unit

	from units.compatibility import compatible
	from utilities.jsonutils import load_json
	from metadata.oaipmh import arXivID_from_path

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Collect Hubble constant annotation files from neural and keyword model outputs, and save CSV files of values from both neural and keyword sources.")
	parser.add_argument('output', action=DirectoryAction,mkdirs=True,mustexist=False, help='Directory in which to store .ann files for comparison.')
	parser.add_argument('rulebased',action=FileAction, mustexist=True, help='JSON file containing results of keyword measurement search.')
	parser.add_argument('neural',action=FileAction, mustexist=True, help='Database containing results of neural models.')
	parser.add_argument('--gold',action=IterFilesAction, recursive=True, suffix='.ann', help='Source directory for training dataset .ann files, searched recursively.')
	args = parser.parse_args()

	#connection = sqlite3.connect(f'file:{args.neural}?mode=ro',uri=True)
	connection = loadconnect(args.neural)
	cursor = connection.cursor()

	names = ('Hubble constant','Hubble Constant','Hubble parameter')
	symbols = ('H _ { 0 }','H _ { o }',r'H _ { \circ }')
	values = query(cursor,f"""
		SELECT value_id,name_id,symbol_id,value,bound,name,symbol_norm AS symbol,confidence,P.arxiv_id,date
		FROM
			all_measurements_confidences M LEFT OUTER JOIN papers P
			ON M.arxiv_id = P.arxiv_id
		WHERE
			(
				({' OR '.join('name LIKE ?' for i in names)})
			OR
				({' OR '.join('symbol_norm LIKE ?' for i in symbols)})
			)
		""", tuple(f'%{i}%' for i in names)+tuple(symbols))
	values['parsed'] = values['value'].apply(parse_measurement)
	values['unit'] = values['parsed'].apply(lambda p: p.unit/p.unit.str_multiplier() if p is not None else None)
	values['canonical'] = values['unit'].apply(lambda u: u.canonical() if u is not None else None)

	values.to_csv('hubbleNeural.csv',index=False)

	valuesDict = load_json(args.rulebased)
	hubbleUnit = parse_unit('km/s/Mpc')

	keywordValues = {'keyword':[],'match':[],'arxiv_id':[],'mention':[], 'text':[], 'keywordspan':[], 'valuespan':[], 'date':[]}
	for keyword, entrylist in valuesDict.items():
		for entry in entrylist:
			if not any('abstract' in s for s in [o[2] for o in entry['origins']]):
				continue

			keywordValues['keyword'].append(keyword)
			keywordValues['match'].append(entry['match'])
			keywordValues['arxiv_id'].append(entry['identifier'])
			keywordValues['mention'].append(entry['mention'])
			keywordValues['text'].append(entry['text'])
			keywordValues['keywordspan'].append(entry['keywordtextspan'])
			keywordValues['valuespan'].append(entry['valuetextspan'])
			keywordValues['date'].append(entry['date'])

	keywordValues = pandas.DataFrame(keywordValues).assign(length=lambda i: i['mention'].str.len()).sort_values('length').groupby(['match','arxiv_id'], as_index=False).first().drop(['length','mention'],1)
	keywordValues['parsed'] = keywordValues['match'].apply(parse_measurement)
	keywordValues['unit'] = keywordValues['parsed'].apply(lambda p: p.unit/p.unit.str_multiplier() if p is not None else None)
	keywordValues['canonical'] = keywordValues['unit'].apply(lambda u: u.canonical() if u is not None else None)

	keywordValues.to_csv('hubbleKeyword.csv',index=False)

	hubbleKeyword = keywordValues[keywordValues['unit'].apply(lambda u: compatible(u,hubbleUnit))]

	hubbleNeural = values[values['unit'].apply(lambda u: compatible(u,hubbleUnit))]
	hubbleNeural['parsed'] = hubbleNeural['parsed'].apply(lambda p: hubbleUnit(p))

	hubbleKeyword['x'] = hubbleKeyword['parsed'].apply(lambda p: p.value)
	hubbleNeural['x'] = hubbleNeural['parsed'].apply(lambda p: p.value)

	hubbleKeywordUnc = hubbleKeyword[hubbleKeyword['parsed'].apply(lambda p: bool(p.uncertainties))]
	hubbleNeuralUnc = hubbleNeural[hubbleNeural['parsed'].apply(lambda p: bool(p.uncertainties))]
	hubbleOverlap = pandas.merge(hubbleKeywordUnc, hubbleNeuralUnc, on=['arxiv_id','x'])

	hubbleKeywordOnly = hubbleKeywordUnc.merge(hubbleOverlap, on=['arxiv_id','x'], how='left', indicator=True)
	hubbleKeywordOnly = hubbleKeywordOnly[hubbleKeywordOnly['_merge'] == 'left_only']
	hubbleNeuralOnly = hubbleNeuralUnc.merge(hubbleOverlap, on=['arxiv_id','x'], how='left', indicator=True)
	hubbleNeuralOnly = hubbleNeuralOnly[hubbleNeuralOnly['_merge'] == 'left_only']

	print('Loaded all datasets.')

	ann_dict = dict()

	for i,row in hubbleKeywordUnc.iterrows():
		#print(f'Process row: {row["arxiv_id"]}')
		if row['arxiv_id'] not in ann_dict:
			ann_dict[row['arxiv_id']] = get_ann(row['arxiv_id'], cursor, parse_notes=True)
			#print('\tStandoff file added')

		continue
		ann = ann_dict[row['arxiv_id']]

		rowtext =  row['text']
		abstracttext = re.sub(r'\u002D|\u2013|\u2014|\u2212','-', ann.text) # Replace alternate hyphens with single ascii hyphen
		m = re.search(re.escape(rowtext), abstracttext)
		if m:
			#print('\tLine match found')
			kent = ann.entity('keyword',m.start()+row['keywordspan'][0], m.start()+row['keywordspan'][1])
			vent = ann.entity('value',m.start()+row['valuespan'][0], m.start()+row['valuespan'][1])
			ann.relation('keywordrel',kent,vent)
			ann.note(vent, str(parse_measurement(vent.text)))
			#print(f'\tAnnotations added: {repr(kent)} -> {repr(vent)}')
		else:
			print(f'Could not find text for {row["arxiv_id"]}.')
			print(repr(rowtext))
			print(abstracttext)


	for i,row in hubbleNeuralUnc.iterrows():
		#print(f'Process row: {row["arxiv_id"]}')
		if row['arxiv_id'] not in ann_dict:
			ann_dict[row['arxiv_id']] = get_ann(row['arxiv_id'], cursor, parse_notes=True)
			#print('\tStandoff file added')


	for i,row in keywordValues.iterrows():
		if row['arxiv_id'] in ann_dict:
			ann = ann_dict[row['arxiv_id']]

			rowtext =  row['text']
			abstracttext = re.sub(r'\u002D|\u2013|\u2014|\u2212','-', ann.text) # Replace alternate hyphens with single ascii hyphen
			m = re.search(re.escape(rowtext), abstracttext)
			if m:
				#print('\tLine match found')
				kent = ann.entity('keyword',m.start()+row['keywordspan'][0], m.start()+row['keywordspan'][1])
				vent = ann.entity('value',m.start()+row['valuespan'][0], m.start()+row['valuespan'][1])
				ann.relation('keywordrel',kent,vent)
				ann.note(vent, str(parse_measurement(vent.text)))
				#print(f'\tAnnotations added: {repr(kent)} -> {repr(vent)}')
			else:
				print(f'Could not find text for {row["arxiv_id"]}.')
				print(repr(rowtext))
				print(abstracttext)


	if args.gold:
		gold_ids = set(arXivID_from_path(p) for p in args.gold)
	else:
		gold_ids = None

	# Make directories for .ann divisions
	neural_path = os.path.join(args.output,'neural')
	keyword_path = os.path.join(args.output,'keyword')
	overlap_path = os.path.join(args.output,'overlap')
	for dirpath in [neural_path,keyword_path,overlap_path]:
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		if gold_ids:
			for state in ['training','other']:
				statepath = os.path.join(dirpath,state)
				if not os.path.exists(statepath):
					os.mkdir(statepath)

	## Save .ann files
	for arXiv,ann in ann_dict.items():
		filename = arXiv.replace('/','')+'.ann'
		in_neural = arXiv in hubbleNeuralUnc['arxiv_id'].values
		in_keyword = arXiv in hubbleKeywordUnc['arxiv_id'].values

		gold_state = ''
		if gold_ids and arXiv in gold_ids:
			gold_state = 'training'
		elif gold_ids:
			gold_state = 'other'

		if in_neural and in_keyword:
			ann.save(os.path.join(overlap_path,gold_state,filename))
		elif in_neural and not in_keyword:
			ann.save(os.path.join(neural_path,gold_state,filename))
		elif in_keyword and not in_neural:
			ann.save(os.path.join(keyword_path,gold_state,filename))
		else:
			print(f'Confused location for {arXiv}')

	cursor.close()
	connection.close()
