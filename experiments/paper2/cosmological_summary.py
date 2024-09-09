if __name__ == "__main__":

	import os
	import numpy
	import math
	from localstats import mean
	from datetime import datetime,timedelta
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.mlab as mlab
	from matplotlib.ticker import NullFormatter
	import pandas

	from units.compatibility import compatible
	from units.dimensionless import DimensionlessUnit
	from parsing import parse_measurement,parse_unit,parse_symbol

	from utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from utilities.dateutilities import getdatetime

	from annotations.database import query,loadconnect

	parser = ArgumentParser(description="Plot histogram of values in file.")
	parser.add_argument('database',action=FileAction, mustexist=True, help='Values to plot.')
	parser.add_argument('omegam',action=FileAction, mustexist=True, help='Omega_m values to plot.')
	parser.add_argument('curvature',action=FileAction, mustexist=True, help='Curvature values to plot.')
	parser.add_argument('-o','--outdir',action=DirectoryAction, mustexist=False, mkdirs=True, help='Directory in which to place plots.')
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

	def get_uncertainty(measurement, confidence, bound):
		if bound == 'U':
			return 1,0
		elif bound == 'L':
			return 0,1
		elif measurement.uncertainties:
			negerr = sum([u.lower**2 for u in measurement.uncertainties])**0.5
			poserr = sum([u.upper**2 for u in measurement.uncertainties])**0.5
			return (negerr / confidence), (poserr / confidence)
		else:
			return 0,0

	def daterange(year_start,month_start,year_end,month_end):
		for year in range(year_start,year_end+1):
			#for month in range(1,13):
			for month in [1,4,7,10]:
				if (year==year_start and month<month_start) or (year==year_end and month>month_end):
					continue
				yield datetime(year,month,1)

	def get_trendline(data, period):
		## Period is timedelta object
		mindate = data['date'].min() + period
		maxdate = data['date'].max()

		dates = []
		values = []
		uncs = []

		for date in daterange(mindate.year, mindate.month, maxdate.year, maxdate.month):
			startdate = date - period
			enddate = date

			period_data = data[(data['date'] >= startdate) & (data['date'] <= enddate)]

			if len(period_data) > 0:

				period_data = period_data.loc[period_data['parsed'].apply(float).sort_values().index]
				crop = math.floor(0.05*len(period_data))
				if crop > 0: period_data = period_data[crop:len(period_data-crop)]

				average = mean(float(p) for p in period_data['parsed'])
				unc = period_data.apply(lambda row: mean(get_uncertainty(row['parsed'],row['confidence'],row['bound']))).mean()

				dates.append(date)
				values.append(average)
				uncs.append(unc)
			elif len(values) > 0:
				dates.append(date)
				values.append(values[-1])
				uncs.append(uncs[-1])

		return numpy.array(dates),numpy.array(values),numpy.array(uncs)

	def make_plots(name, data, bounds=None, datebounds=None, unit=None):

		bottom,left = 0.15,0.15
		width,height = 0.65,0.65
		bottom_xh = bottom + height + 0.02
		left_yh = left + width + 0.02
		height_h = 0.15

		rect_scatter = [left, bottom, width, height]
		rect_histx = [left, bottom_xh, width, height_h]
		rect_histy = [left_yh, bottom, height_h, height]

		#fig = plt.figure(figsize=(7.225,7))
		fig = plt.figure(figsize=(5.16,5))

		axScatter = fig.add_axes(rect_scatter)
		axHistx = fig.add_axes(rect_histx)
		axHisty = fig.add_axes(rect_histy)

		if bounds:
			data = data[data['parsed'].apply(lambda p: bounds[0] <= float(p) <= bounds[1])]

		upperlimits = numpy.array([b == 'U' for b in data['bound']])
		lowerlimits = numpy.array([b == 'L' for b in data['bound']])
		constraints = upperlimits | lowerlimits

		uncertainties = numpy.array([get_uncertainty(p,c,b) for p,c,b in zip(data['parsed'],data['confidence'],data['bound'])]).T

		measurements = numpy.array([float(p) for p in data['parsed']])
		dates = numpy.array(list(data['date'].values)) # numpy.array([getdatetime(d) for d in data['date']])

		# Plot central values
		axScatter.errorbar(dates[~constraints],measurements[~constraints],yerr=uncertainties.T[~constraints].T,lw=0.5,alpha=0.5,marker=None,fmt="none",zorder=0)
		axScatter.scatter(dates[~constraints],measurements[~constraints],s=10,marker='o',alpha=0.5,zorder=100)

		# Plot constraint values
		#ymin,ymax = axScatter.get_ylim()
		#linelength = (ymax - ymin) * 0.05
		#axScatter.errorbar(dates[constraints],measurements[constraints],yerr=uncertainties.T[constraints].T*linelength,uplims=upperlimits[constraints],lolims=lowerlimits[constraints],lw=0.5,marker=None,fmt="none",capsize=2,zorder=0)
		#axScatter.scatter(dates[constraints],measurements[constraints],s=40,marker='_',alpha=0.5,zorder=100)

		# Plot constraint values with arrow markers
		axScatter.scatter(dates[upperlimits],measurements[upperlimits],s=30,marker='v',alpha=0.5,zorder=100)
		axScatter.scatter(dates[lowerlimits],measurements[lowerlimits],s=30,marker='^',alpha=0.5,zorder=100)

		years = 5
		trend_x, trend_y, trend_u = get_trendline(data,timedelta(weeks=years*52))
		axScatter.plot(trend_x, trend_y, zorder=101, alpha=0.7)
		axScatter.fill_between(trend_x, trend_y-trend_u, trend_y+trend_u, zorder=50, alpha=0.2)

		if bounds:
			#ymin,ymax = axScatter.get_ylim()
			#ylim = max(ymin,bounds[0]),min(ymax,bounds[1])
			#axScatter.set_ylim(ylim)
			axScatter.set_ylim(bounds)

		if datebounds:
			#xmin,xmax = axScatter.get_xlim()
			#print(xmin,xmax,type(xmin),type(xmax))
			#xlim = max(xmin,float(datebounds[0])),min(xmax,float(datebounds[1]))
			#axScatter.set_xlim(xlim)
			axScatter.set_xlim(datebounds)

		#date_bins = numpy.array(list(months_range(dates.min(),dates.max(),inclusive=True,follow_on=True)))
		#date_bins = mpl.dates.date2num(date_bins)
		#dates = mpl.dates.date2num(dates)

		axHistx.hist(dates, bins='auto')
		axHisty.hist(measurements, bins='auto', orientation='horizontal')

		axHistx.set_xlim(axScatter.get_xlim())
		axHisty.set_ylim(axScatter.get_ylim())

		nullfmt = NullFormatter()
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)

		ticksize = 10
		axHistx.yaxis.set_tick_params(labelsize=ticksize)
		axHisty.xaxis.set_tick_params(labelsize=ticksize)
		axScatter.xaxis.set_tick_params(labelsize=ticksize)
		axScatter.yaxis.set_tick_params(labelsize=ticksize)

		labelsize = 10
		if isinstance(unit,DimensionlessUnit):
			unittext = ' / Dimensionless'
		else:
			unittext = f' / {unit.latex()}' if unit else ''
		axScatter.set_xlabel('Date of Publication / Year',fontsize=labelsize,labelpad=labelsize)
		axScatter.set_ylabel('Extracted Value' + unittext,fontsize=labelsize)
		#axScatter.set_ylabel('Measurement Value')

		hist_filename = makefile(name+fileext)
		fig.savefig(hist_filename)


		## Plot error distribution figure
		fig = plt.figure(figsize=(5,3.87))
		ax = fig.add_subplot('111')

		mean_value = mean(measurements[~constraints])

		asym_unc = numpy.array([u[0] if mean_value<m else u[1] for m,u in zip(measurements[~constraints],uncertainties.T[~constraints])]) # u[0] is lower,u[1] is upper

		n_sigma = (measurements[~constraints]-mean_value)/numpy.abs(asym_unc)
		n_sigma = n_sigma[(-5<n_sigma) & (n_sigma<5)] # Limit range for visual readability

		# Plot error histogram
		ax.hist(n_sigma,bins='auto',density=True)

		# Set x limits
		ax.set_xlim((-5,5))
		#ymin,ymax = ax.get_xlim()
		#ax.set_xlim((max(ymin,-5),min(ymax,5)))

		# Add Gaussian normal overlay
		x = numpy.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
		ax.plot(x,mlab.normpdf(x, 0, 1))
		ax.axvline(0,linestyle='--')

		labelsize = 10
		ax.set_xlabel('$n_{\sigma}$',fontsize=labelsize,labelpad=labelsize)
		ax.set_ylabel('Frequency',fontsize=labelsize,labelpad=labelsize)

		ticksize = 10
		ax.xaxis.set_tick_params(labelsize=ticksize)
		ax.yaxis.set_tick_params(labelsize=ticksize)

		fig.tight_layout()

		error_filename = makefile(name+'_error'+fileext)
		fig.savefig(error_filename)


	## Listings of quantities to plot
	listings = [
			[
				'hubbleconstant',
				['Hubble constant','Hubble Constant','Hubble parameter'],
				['H _ { 0 }','H _ { o }',r'H _ { \circ }'],
				(-5,105),
				(getdatetime('1994-01-01'), getdatetime('2020-12-31')),
				parse_unit('km/s/Mpc')
			],
			[
				'omegam',
				['matter density', 'mass density'],
				['\Omega _ { m }', '\Omega _ { M }', '\Omega _ { 0 }'],
				(0,1),
				(getdatetime('1994-01-01'), getdatetime('2020-12-31')),
				DimensionlessUnit(1)
			],
			[
				'omegalambda',
				[],
				['\Omega _ { \Lambda }'],
				(0,1),
				None,
				DimensionlessUnit(1)
			],
			[
				'sigma8',
				[],
				['\sigma _ { 8 }'],
				(0,2),
				None,
				DimensionlessUnit(1)
			],
			[
				'omegab',
				['baryon density'],
				['\Omega _ { b }', '\Omega _ { B }'],
				(0,0.1),
				None,
				DimensionlessUnit(1)
			],
			[
				'omegabh2',
				['baryon density'],
				['\Omega _ { b } h ^ { 2 }', '\Omega _ { B } h ^ { 2 }'],
				(0,0.04),
				None,
				DimensionlessUnit(1)
			],
			[
				'spectralindex',
				['spectral index'],
				['n _ { s }'],
				(0.9,1.05),
				None,
				DimensionlessUnit(1)
			],
			[
				'mnu',
				['neutrino mass'],
				['m _ { \\nu }', 'M _ { \\nu }'],
				(0,1.5),
				None,
				parse_unit('eV')
			],
			[
				'totalmnu',
				['sum of neutrino masses', 'total neutrino mass'],
				['\sum m _ { \\nu }', '\sum M _ { \\nu }', '\Sigma m _ { \\nu }', '\Sigma M _ { \\nu }'],
				(0,1.5),
				None,
				parse_unit('eV')
			],
			[
				'omegak',
				[],
				['\Omega _ { k }', '\Omega _ { K }'],
				(-0.1,0.1),
				None,
				DimensionlessUnit(1)
			],
			[
				'w0',
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

	## Get data and make plots
	for name,names,symbols,bounds,datebounds,unit in listings:
		values = query_name_symbol(cursor, names, symbols, unit=unit, requireunc=True)
		print(f'{name} values: {values.shape[0]}')
		make_plots(name, values, bounds=bounds, datebounds=datebounds, unit=unit)
		make_plots(name+'_unc', values[values['bound'].apply(lambda b: b not in 'UL')], bounds=bounds, datebounds=datebounds, unit=unit)
		cached_values[name] = values


	## Plot Omega_m keyword search values
	keywordValues = pandas.read_csv(args.omegam)
	keywordValues['confidence'] = 1
	keywordValues['bound'] = 'C'
	keywordValues['parsed'] = keywordValues['match'].apply(parse_measurement)
	keywordValues = keywordValues[keywordValues['parsed'].apply(lambda p: bool(p))]
	keywordValues['unit'] = keywordValues['parsed'].apply(lambda p: p.unit if p else None)
	keywordValues['date'] = keywordValues['date'].apply(getdatetime)
	keywordValues['x'] = keywordValues['parsed'].apply(lambda p: p.value)

	keywordUnc = keywordValues[keywordValues['parsed'].apply(lambda p: bool(p.uncertainties))]
	keywordUnc = keywordUnc[keywordUnc['x'].apply(lambda x: 0 <= x <= 1)]

	make_plots('omegam_rulesbased',keywordUnc,bounds=(0,1), datebounds=(getdatetime('1994-01-01'), getdatetime('2020-12-31')), unit=DimensionlessUnit(1))
	print(f'Omega_M keyword values: {keywordUnc.shape[0]}')

	omegamneural = cached_values['omegam']
	#omegamneural['datetime'] = omegamneural['date'].apply(getdatetime)
	omegamcount = omegamneural[omegamneural['date'] < getdatetime('2017-10-01')].shape[0]
	print(f'Omega_M neural pre-Sep17 count: {omegamcount}')


	## Plot curvature keyword search values
	keywordValues = pandas.read_csv(args.curvature)
	keywordValues['confidence'] = 1
	keywordValues['bound'] = 'C'
	keywordValues['parsed'] = keywordValues['match'].apply(parse_measurement)
	keywordValues = keywordValues[keywordValues['parsed'].apply(lambda p: bool(p))]
	keywordValues['unit'] = keywordValues['parsed'].apply(lambda p: p.unit if p else None)
	keywordValues['date'] = keywordValues['date'].apply(getdatetime)
	keywordValues['x'] = keywordValues['parsed'].apply(lambda p: p.value)

	keywordUnc = keywordValues[keywordValues['parsed'].apply(lambda p: bool(p.uncertainties))]
	keywordUnc = keywordUnc[keywordUnc['x'].apply(lambda x: -1 <= x <= 1)]

	#make_plots('curvature_rulesbased', keywordUnc,bounds=(0,1), datebounds=(getdatetime('1994-01-01'), getdatetime('2020-12-31')), unit=DimensionlessUnit(1))
	make_plots('curvature_rulesbased', keywordUnc, unit=DimensionlessUnit(1))
	print(f'Curvature keyword values: {keywordUnc.shape[0]}')
