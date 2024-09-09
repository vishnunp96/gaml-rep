if __name__ == "__main__":

	import os
	import matplotlib
	matplotlib.use('Agg')
	import pandas

	from units.abstract import AbstractUnit
	from units.measurement import Measurement
	from units.compatibility import compatible
	from units.dimensionless import DimensionlessUnit
	from parsing import parse_measurement,parse_unit,parse_symbol

	from utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from utilities.dateutilities import getdatetime
	from utilities.jsonutils import dump_json
	from json import JSONEncoder
	import datetime

	from annotations.database import query,loadconnect

	parser = ArgumentParser(description="Plot histogram of values in file.")
	parser.add_argument('database',action=FileAction, mustexist=True, help='Database to take values from.')
	parser.add_argument('hubbleconstant',action=FileAction, mustexist=True, help='Hubble constant values to normalise.')
	parser.add_argument('omegam',action=FileAction, mustexist=True, help='Omega_m values to normalise.')
	parser.add_argument('curvature',action=FileAction, mustexist=True, help='Curvature values to normalise.')
	parser.add_argument('outdir',action=DirectoryAction, mustexist=False, mkdirs=True, help='Directory in which to place .csv files.')
	args = parser.parse_args()

	fileext = '.pdf'

	def makefile(filename):
		if args.outdir:
			return os.path.join(args.outdir, filename)
		else:
			return filename

	def query_name_symbol(cursor, names, symbols, unit=None, requireunc=False):
		if symbols:
			symbols = [str(parse_symbol(s)) for s in symbols]
		if names and symbols:
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
		elif names:
			values = query(cursor,f"""
				SELECT value_id,name_id,symbol_id,value,bound,name,symbol_norm AS symbol,confidence,P.arxiv_id,date
				FROM
					all_measurements_confidences M LEFT OUTER JOIN papers P
					ON M.arxiv_id = P.arxiv_id
				WHERE ({' OR '.join('name LIKE ?' for i in names)})
				""", tuple(f'%{i}%' for i in names))
		elif symbols:
			values = query(cursor,f"""
				SELECT value_id,name_id,symbol_id,value,bound,name,symbol_norm AS symbol,confidence,P.arxiv_id,date
				FROM
					all_measurements_confidences M LEFT OUTER JOIN papers P
					ON M.arxiv_id = P.arxiv_id
				WHERE ({' OR '.join('symbol_norm LIKE ?' for i in symbols)})
				""", tuple(symbols))
		else:
			values = pandas.DataFrame(columns=['value_id','name_id','symbol_id','value','bound','name','symbol','confidence','arxiv_id','date'])

		values['parsed'] = values['value'].apply(parse_measurement)
		values['date'] = values['date'].apply(getdatetime)

		values = values[values['parsed'].apply(lambda p: p is not None)]

		if unit is not None:
			values = values[values['parsed'].apply(lambda p: compatible(p.unit,unit) if p else False)]

		if requireunc:
			#values = values[values['parsed'].apply(lambda p: bool(p.uncertainties) if p else False)]
			values = values[values.apply(lambda row: bool(row['parsed'].uncertainties) or row['bound'] in 'UL', axis=1)]

		return values

	## Listings of quantities to plot
	listings = [
			[
				'hubbleconstant',
				'H_0',
				['Hubble constant','Hubble Constant','Hubble parameter'],
				['H _ { 0 }','H _ { o }',r'H _ { \circ }'],
				(35,105),
				('1994-01-01', '2020-12-31'),
				parse_unit('km/s/Mpc')
			],
			[
				'omegam',
				'\\Omega_M',
				['matter density', 'mass density'],
				['\Omega _ { m }', '\Omega _ { M }', '\Omega _ { 0 }'],
				(0,1),
				('1994-01-01', '2020-12-31'),
				DimensionlessUnit(1)
			],
			[
				'omegalambda',
				'\\Omega_\\Lambda',
				[],
				['\Omega _ { \Lambda }'],
				(0,1),
				None,
				DimensionlessUnit(1)
			],
			[
				'sigma8',
				'\\sigma_8',
				[],
				['\sigma _ { 8 }'],
				(0,2),
				None,
				DimensionlessUnit(1)
			],
			[
				'omegab',
				'\\Omega_b',
				['baryon density'],
				['\Omega _ { b }', '\Omega _ { B }'],
				(0,0.1),
				None,
				DimensionlessUnit(1)
			],
			[
				'omegabh2',
				'\\Omega_b h^2',
				['baryon density'],
				['\Omega _ { b } h ^ { 2 }', '\Omega _ { B } h ^ { 2 }'],
				(0,0.04),
				None,
				DimensionlessUnit(1)
			],
			[
				'spectralindex',
				'n_s',
				['spectral index'],
				['n _ { s }'],
				(0.9,1.05),
				None,
				DimensionlessUnit(1)
			],
			[
				'mnu',
				'm_{\\nu}',
				['neutrino mass'],
				['m _ { \\nu }', 'M _ { \\nu }'],
				(0,1.5),
				None,
				parse_unit('eV')
			],
			[
				'totalmnu',
				'\\sum m_{\\nu}',
				['sum of neutrino masses', 'total neutrino mass'],
				['\sum m _ { \\nu }', '\sum M _ { \\nu }', '\Sigma m _ { \\nu }', '\Sigma M _ { \\nu }'],
				(0,1.5),
				None,
				parse_unit('eV')
			],
			[
				'omegak',
				'\\Omega_k',
				[],
				['\Omega _ { k }', '\Omega _ { K }'],
				(-0.1,0.1),
				None,
				DimensionlessUnit(1)
			],
			[
				'w0',
				'w_0',
				[],
				['w _ { 0 }'],
				(-2,-0.5),
				None,
				DimensionlessUnit(1)
			]
		]

	## Open database
	connection = loadconnect(args.database)
	cursor = connection.cursor()

	print('Database loaded.')

	cached_values = dict()

	class MyEncoder(JSONEncoder):
		def default(self, o):
			if isinstance(o, (datetime.datetime, pandas.Timestamp, AbstractUnit, Measurement)):
				return str(o)
			return JSONEncoder.default(self, o)

	def save_values(name, fancy_name, values, bounds, datebounds, unit):
		#values_dict = values[['date','parsed','confidence','bound']].to_dict(orient='records')
		values_dict = values.to_dict(orient='records')
		data = {'name': name, 'latex': fancy_name, 'bounds': bounds, 'datebounds': datebounds, 'unit': unit, 'data': values_dict}
		dump_json(data, makefile(name+'.json'), cls=MyEncoder)
		print(f'{name} values: {values.shape[0]}')
		cached_values[name] = values

	## Get data and make plots
	for name,fancy_name,names,symbols,bounds,datebounds,unit in listings:
		values = query_name_symbol(cursor, names, symbols, unit=unit, requireunc=True)
		save_values(name, fancy_name, values, bounds, datebounds, unit)

	def process_rulesbased(filename, name, fancy_name, bounds, datebounds, unit, requireunc=False):
		keywordValues = pandas.read_csv(filename)
		keywordValues['confidence'] = 1
		keywordValues['bound'] = 'C'

		keywordValues['parsed'] = keywordValues['match'].apply(parse_measurement)
		keywordValues['date'] = keywordValues['date'].apply(getdatetime)

		keywordValues = keywordValues[keywordValues['parsed'].apply(lambda p: bool(p))]

		if unit is not None:
			keywordValues = keywordValues[keywordValues['parsed'].apply(lambda p: compatible(p.unit,unit) if p else False)]
		if requireunc:
			keywordValues = keywordValues[keywordValues['parsed'].apply(lambda p: bool(p.uncertainties))]

		save_values(name, fancy_name, keywordValues, bounds, datebounds, unit)

	process_rulesbased(args.hubbleconstant, 'hubbleconstant_rulesbased', 'H_0', (35,105), (getdatetime('1994-01-01'), getdatetime('2020-12-31')), parse_unit('km/s/Mpc'), requireunc=True)
	process_rulesbased(args.omegam, 'omegam_rulesbased', '\\Omega_M', (0,1), (getdatetime('1994-01-01'), getdatetime('2020-12-31')), DimensionlessUnit(1), requireunc=True)
	process_rulesbased(args.curvature, 'curvature_rulesbased', '\\Omega_k', (-1,1), (getdatetime('1994-01-01'), getdatetime('2020-12-31')), DimensionlessUnit(1), requireunc=True)
