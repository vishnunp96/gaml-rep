import sqlite3
import pandas
from annotations.database import query
from parsing import parse_measurement,parse_unit,parse_symbol

from units.compatibility import compatible
#from units.dimensionless import DimensionlessUnit

#conn = sqlite3.connect('file:/mnt/databases/database4simpleentrulerel.db?mode=ro',uri=True)
#conn = sqlite3.connect('file:/mnt/databases/database5charentrulerelwindowattr.db?mode=ro',uri=True)
conn = sqlite3.connect('file:/mnt/databases/database6bestfromsearch.db?mode=ro',uri=True)
c = conn.cursor()

#symbols = query(c,'SELECT symbol,COUNT(arxiv_id) AS count FROM parameter_symbol_occurences GROUP BY symbol ORDER BY count')
symbols = query(c,'''
		SELECT symbol_norm AS symbol,SUM(partialcount) AS count
		FROM
			(SELECT symbol,COUNT(arxiv_id) AS partialcount FROM parameter_symbol_occurences GROUP BY symbol) C
		LEFT JOIN
			parameter_symbols S
		ON C.symbol = S.symbol
		GROUP BY symbol_norm''')
symbols['parsed'] = symbols['symbol'].apply(lambda s: parse_symbol(s))

#data = query(c,'SELECT * FROM all_measurements_values')
data = query(c,'SELECT value_id,name_id,symbol_id,value,bound,name,symbol_norm AS symbol,arxiv_id FROM all_measurements')
data['parsed'] = data['value'].apply(parse_measurement)
data['unit'] = data['parsed'].apply(lambda p: p.unit/p.unit.str_multiplier() if p is not None else None)
data['canonical'] = data['unit'].apply(lambda u: u.canonical() if u is not None else None)

#names = query(c,'SELECT names.name_id,name,symbol,COUNT(arxiv_id) AS count FROM names LEFT OUTER JOIN name_occurences ON names.name_id=name_occurences.name_id GROUP BY names.name_id ORDER BY count')
names = query(c,'''
		SELECT N.name_id,N.name,N.symbol,SUM(C.partialcount) AS count
		FROM
			(SELECT name_id,COUNT(arxiv_id) AS partialcount FROM name_occurences GROUP BY name_id) C
		LEFT JOIN
			names N
		ON C.name_id = N.name_id
		GROUP BY N.name_id,N.name,N.symbol''')

values = data[data['parsed'].apply(lambda p: p is not None)]

value_names_canonical = values.groupby(['name','symbol','canonical']).size().to_frame('size').reset_index().sort_values('size').groupby(['name','symbol'],group_keys=False).apply(lambda i: i.assign(frac=i['size']/i['size'].sum()))
value_names_units = values.assign(canonical=lambda i: i['unit'].apply(lambda u: u.canonical())).groupby(['name','symbol','unit','canonical']).size().to_frame('size').reset_index().sort_values('size').groupby(['name','symbol','canonical'],group_keys=False).apply(lambda i: i.assign(frac=i['size']/i['size'].sum()))
value_names = pandas.merge(value_names_canonical, value_names_units, on=['name','symbol','canonical'], suffixes=('_canonical','_units'))

collected_names = pandas.merge(names,value_names,on=['name','symbol'])[['name','symbol','canonical','unit','count','size_canonical','frac_canonical','size_units','frac_units']]



from utilities.jsonutils import load_json

valuesDict = load_json('/mnt/searches/arXivNeat_latest_withspans.json')
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

hubbleNeuralAll = values[(values['name'].isin(['Hubble constant','Hubble Constant','Hubble parameter']) | values['symbol'].isin(['H _ { 0 }','H _ { o }',r'H _ { \\circ }']))]
hubbleNeural = hubbleNeuralAll[hubbleNeuralAll['unit'].apply(lambda u: compatible(u,hubbleUnit))]
hubbleNeural['parsed'] = hubbleNeural['parsed'].apply(lambda p: hubbleUnit(p))

hubbleKeyword['x'] = hubbleKeyword['parsed'].apply(lambda p: p.value)
hubbleNeural['x'] = hubbleNeural['parsed'].apply(lambda p: p.value)

hubbleKeywordUnc = hubbleKeyword[hubbleKeyword['parsed'].apply(lambda p: bool(p.uncertainties))]
hubbleNeuralUnc = hubbleNeural[hubbleNeural['parsed'].apply(lambda p: bool(p.uncertainties))]
hubbleOverlap = pandas.merge(hubbleKeywordUnc[['keyword','parsed','arxiv_id','x']], hubbleNeuralUnc[['parsed','arxiv_id','x']], on=['arxiv_id','x'])

hubbleKeywordOnly = hubbleKeywordUnc.merge(hubbleOverlap, on=['arxiv_id','x'], how='left', indicator=True)
hubbleKeywordOnly = hubbleKeywordOnly[hubbleKeywordOnly['_merge'] == 'left_only']
hubbleNeuralOnly = hubbleNeuralUnc.merge(hubbleOverlap, on=['arxiv_id','x'], how='left', indicator=True)
hubbleNeuralOnly = hubbleNeuralOnly[hubbleNeuralOnly['_merge'] == 'left_only']


allKeywordCount = sum([len(i) for i in valuesDict.values()])
allKeywordParsedCount = len([e for elist in valuesDict.values() for e in elist if bool(parse_measurement(e['match']))])
