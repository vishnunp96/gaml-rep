if __name__ == "__main__":

	import numpy
	import pandas
	import re
	from gaml.preprocessing.manifest import ManifestAction
	from gaml.utilities.argparseactions import ArgumentParser,FileAction
	from gaml.utilities import StopWatch
	from gaml.utilities.terminalutils import printacross,printwrapped,color_text,ColorCode

	import sqlite3
	from gaml.annotations.database import query
	from gaml.parsing import parse_measurement,parse_unit

	from gaml.units.compatibility import compatible
	from gaml.utilities.jsonutils import load_json,dump_json

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Manually annotate results of keyword search (from json) and neural model (from database).")
	parser.add_argument('source',action=ManifestAction, help='Directory containing articles and manifest.')
	parser.add_argument('output',action=FileAction, mustexist=False, help='File in which output will be stored or read from (to resume classification).')
	parser.add_argument('rulebased',action=FileAction, mustexist=True, help='JSON file containing results of keyword measurement search.')
	parser.add_argument('neural',action=FileAction, mustexist=True, help='Database containing results of neural models.')
	parser.add_argument('-n','--number', type=int,default=100, help='Number of samples to annotate. Defaults to 100.')
	args = parser.parse_args()

	# Select n entries from score list
	# Loop over entries
	# Print abstract of each entry
	# Request classification from user
	# Save classification and id (write to file as soon as recieved, to prevent losing work)
	# Try and implement a function for resuming a lost session?
	##### Create table of ids and labels, with -1s in label column
	##### Upon reloading, any entry which does not contain -1 has already been classified
	##### This would mean that we have to re-save whole table every time, but that won't be costly

	manifest = args.source

	if args.output_exists:
		samples = load_json(args.output)
	else:
		connection = sqlite3.connect(f'file:{args.neural}?mode=ro',uri=True)
		cursor = connection.cursor()

		names = ('Hubble constant','Hubble Constant','Hubble parameter')
		symbols = ('H _ { 0 }','H _ { o }',r'H _ { \\circ }')
		values = query(cursor,f"""
			SELECT value_id,name_id,symbol_id,value,bound,name,symbol_norm AS symbol,arxiv_id FROM all_measurements
			WHERE
				(
					({' OR '.join('name LIKE ?' for i in names)})
				OR
					({' OR '.join('symbol_norm LIKE ?' for i in symbols)})
				)
			""", tuple(f'%{i}%' for i in names)+symbols)
		values['parsed'] = values['value'].apply(parse_measurement)
		values['unit'] = values['parsed'].apply(lambda p: p.unit/p.unit.str_multiplier() if p is not None else None)
		values['canonical'] = values['unit'].apply(lambda u: u.canonical() if u is not None else None)


		valuesDict = load_json(args.rulebased)
		hubbleUnit = parse_unit('km/s/Mpc')
		hubbleKeyword = {'keyword':[],'match':[],'parsed':[],'arxiv_id':[],'mention':[], 'text':[], 'keywordspan':[], 'valuespan':[]}
		for keyword, entrylist in valuesDict.items():
			for entry in entrylist:
				if not any('abstract' in s for s in [o[2] for o in entry['origins']]):
					continue

				parsed = parse_measurement(entry['match'])

				if parsed and compatible(hubbleUnit,parsed.unit):
					parsed = hubbleUnit(parsed)

					hubbleKeyword['keyword'].append(keyword)
					hubbleKeyword['match'].append(entry['match'])
					hubbleKeyword['parsed'].append(parsed)
					hubbleKeyword['arxiv_id'].append(entry['identifier'])
					hubbleKeyword['mention'].append(entry['mention'])
					hubbleKeyword['text'].append(entry['text'])
					hubbleKeyword['keywordspan'].append(entry['keywordtextspan'])
					hubbleKeyword['valuespan'].append(entry['valuetextspan'])

		hubbleKeyword = pandas.DataFrame(hubbleKeyword).assign(length=lambda i: i['mention'].str.len()).sort_values('length').groupby(['match','arxiv_id'], as_index=False).first().drop(['length','mention'],1)

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


		## Get stratified sample
		hubbleKeywordOnlyCount = hubbleKeywordOnly.shape[0]
		hubbleNeuralOnlyCount = hubbleNeuralOnly.shape[0]
		hubbleOverlapCount = hubbleOverlap.shape[0]
		totalCount = hubbleKeywordOnlyCount + hubbleNeuralOnlyCount + hubbleOverlapCount

		keywordOnlySampleCount = round(args.number*(hubbleKeywordOnlyCount/totalCount))
		neuralOnlySampleCount = round(args.number*(hubbleNeuralOnlyCount/totalCount))
		overlapSampleCount = round(args.number*(hubbleOverlapCount/totalCount) / 2) # Divided as each entry will be examined twice: from the perspective of the keyword and neural results

		print(f'Keyword samples: {keywordOnlySampleCount}, Neural samples: {neuralOnlySampleCount}, Overlap samples: {overlapSampleCount}')

		keywordSamples = hubbleKeywordOnly.sample(n=keywordOnlySampleCount)
		neuralSamples = hubbleNeuralOnly.sample(n=neuralOnlySampleCount)
		overlapSamples = hubbleOverlap.sample(n=overlapSampleCount)


		samples = []

		for i,row in keywordSamples.iterrows():
			if keyword in names:
				namespan = row['keywordspan_x']
				symbolspan = None
			else:
				symbolspan = row['keywordspan_x']
				namespan = None
			entry = {
					'origin': 'keyword',
					'arxiv': row['arxiv_id'],
					'text': row['text_x'],
					'namespan': namespan,
					'symbolspan': symbolspan,
					'valuespan': row['valuespan_x']
				}
			samples.append(entry)

		for i,row in neuralSamples.iterrows():
			namespan = None
			symbolspan = None
			if not numpy.isnan(row['name_id_x']):
				namerow = query(cursor,'SELECT start,end FROM parameter_name_occurences WHERE entity_id = ?', (int(row['name_id_x']),)).iloc[0]
				namespan = (int(namerow['start']), int(namerow['end']))
			if not numpy.isnan(row['symbol_id_x']):
				symbolrow = query(cursor,'SELECT start,end FROM parameter_symbol_occurences WHERE entity_id = ?', (int(row['symbol_id_x']),)).iloc[0]
				symbolspan = (int(symbolrow['start']), int(symbolrow['end']))
			valrow = query(cursor,'SELECT start,end FROM measurements WHERE value_id = ?', (int(row['value_id_x']),)).iloc[0]
			valuespan = (int(valrow['start']), int(valrow['end']))
			entry = {
					'origin': 'neural',
					'arxiv': row['arxiv_id'],
					'text': query(cursor,'SELECT abstract FROM papers WHERE arxiv_id = ?', (row['arxiv_id'],)).iloc[0,0],
					'namespan': namespan,
					'symbolspan': symbolspan,
					'valuespan': valuespan
				}
			samples.append(entry)

		## Repeat above steps for those in overlap
		## All overlap rows must have both keyword and neural results

		for i,row in overlapSamples.iterrows():
			if keyword in names:
				namespan = row['keywordspan']
				symbolspan = None
			else:
				symbolspan = row['keywordspan']
				namespan = None
			entry = {
					'origin': 'keyword',
					'arxiv': row['arxiv_id'],
					'text': row['text'],
					'namespan': namespan,
					'symbolspan': symbolspan,
					'valuespan': row['valuespan']
				}
			samples.append(entry)

		for i,row in overlapSamples.iterrows():
			namespan = None
			symbolspan = None
			if not numpy.isnan(row['name_id']):
				namerow = query(cursor,'SELECT start,end FROM parameter_name_occurences WHERE entity_id = ?', (int(row['name_id']),)).iloc[0]
				namespan = (int(namerow['start']), int(namerow['end']))
			if not numpy.isnan(row['symbol_id']):
				symbolrow = query(cursor,'SELECT start,end FROM parameter_symbol_occurences WHERE entity_id = ?', (int(row['symbol_id']),)).iloc[0]
				symbolspan = (int(symbolrow['start']), int(symbolrow['end']))
			valrow = query(cursor,'SELECT start,end FROM measurements WHERE value_id = ?', (int(row['value_id']),)).iloc[0]
			valuespan = (int(valrow['start']), int(valrow['end']))
			entry = {
					'origin': 'neural',
					'arxiv': row['arxiv_id'],
					'text': query(cursor,'SELECT abstract FROM papers WHERE arxiv_id = ?', (row['arxiv_id'],)).iloc[0,0],
					'namespan': namespan,
					'symbolspan': symbolspan,
					'valuespan': valuespan
				}
			samples.append(entry)

		cursor.close()
		connection.close()

		dump_json(samples, args.output)



	counter = 0
	for sample in samples:
		counter += 1

		if 'note' in sample:
			continue

		print('\033[2J\033[H',end='')
		printacross('=',maxwidth=70)
		printacross('=', begin=f'{sample["arxiv"]:24}({counter}/{len(samples)}) ', maxwidth=70)
		print()

		text = sample['text']

		#if sample['namespan']: text = color_text(text,ColorCode.red,spans=[sample['namespan']])
		#if sample['symbolspan']: text = color_text(text,ColorCode.green,spans=[sample['symbolspan']])
		#text = color_text(text,ColorCode.blue,spans=[sample['valuespan']])
		text = color_text(text,ColorCode.blue,spans=[i for i in [sample['namespan'], sample['symbolspan'], sample['valuespan']] if i is not None])

		text = text.strip()
		text = re.sub('\ +\.','.',text)
		text = re.sub('\ +\,',',',text)
		text = re.sub('\ +\:',':',text)
		text = re.sub('\ +\)',')',text)
		text = re.sub('\(\ +','(',text)
		text = re.sub('(?<=[A-Za-z])\s*\-\s*(?=[A-Za-z])','-',text)
		text = re.sub('\xe2','-',text)
		text = text.encode('utf8').decode('ascii',errors='ignore')

		#text = re.sub(measurementre,'\x1b[0;31;40m\g<0>\x1b[0m',text)

		printwrapped(text, width=60)
		print()

		printacross('=',maxwidth=70)

		while True:
			note = input('Note for this sample (y/n): ').strip()
			if note: break

		result = note.lower()

		if result == 'exit':
			break
		elif result == 's':
			continue
		else:
			sample['note'] = note

		dump_json(samples, args.output)


	print('\033[2J\033[H',end='') # Clear console

	#print(samples)
