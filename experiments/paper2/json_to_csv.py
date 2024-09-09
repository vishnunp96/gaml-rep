if __name__ == "__main__":

	import pandas
	from gaml.utilities.argparseactions import ArgumentParser,FileAction
	from gaml.utilities import StopWatch

	from gaml.parsing import parse_measurement

	from gaml.utilities.jsonutils import load_json

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Collect measurements from JSON search results (keyword search), and save as CSV.")
	parser.add_argument('rulebased',action=FileAction, mustexist=True, help='JSON file containing results of keyword measurement search.')
	parser.add_argument('output', action=FileAction, mustexist=False, help='File in which to store CSV.')
	args = parser.parse_args()

	valuesDict = load_json(args.rulebased)

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

	keywordValues.to_csv(args.output,index=False)

