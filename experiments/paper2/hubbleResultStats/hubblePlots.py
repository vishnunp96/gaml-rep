if __name__ == "__main__":

	import os
	import pandas
	pandas.options.mode.chained_assignment = None  # default='warn'

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from gaml.utilities import StopWatch

	#import sqlite3
	from gaml.parsing import parse_measurement,parse_unit

	from gaml.units.compatibility import compatible

	import numpy
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.ticker import NullFormatter

	from gaml.units.dimensionless import DimensionlessUnit
	from gaml.units.measurement import Measurement,Uncertainty

	from gaml.utilities.dateutilities import getdatetime

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Collect Hubble constant annotation files from neural and keyword model outputs.")
	parser.add_argument('output', action=DirectoryAction,mkdirs=True,mustexist=False, help='Directory in which to store .ann files for comparison.')
	parser.add_argument('keyword',action=FileAction, mustexist=True, help='CSV file containing keyword measurements.')
	parser.add_argument('neural',action=FileAction, mustexist=True, help='CSV file containing neural measurements.')
	parser.add_argument('croft',action=FileAction, mustexist=True, help='Croft data for comparison.')
	args = parser.parse_args()

	fileext = '.pdf'

	hubbleUnit = parse_unit('km/s/Mpc')

	### Load croft values
	def get_croft_measurement(row):
		value = float(row['value'])
		if row['lower error'] == 0:
			unc = Uncertainty(value=float(row['upper error']))
		else:
			unc = Uncertainty(upper=float(row['upper error']), lower=float(row['lower error']))
		return Measurement(value,hubbleUnit,[unc])
	croft = pandas.read_csv(args.croft)
	hubbleCroft = croft[croft['parameter'].fillna('').str.match('^H_0$')]
	hubbleCroft['date'] = hubbleCroft.apply(lambda row: f"{row['year']:04d}-{row['month']:02d}-01 00:00:00", axis=1)
	hubbleCroft['parsed'] = hubbleCroft.apply(get_croft_measurement, axis=1)
	hubbleCroft['unit'] = hubbleCroft['parsed'].apply(lambda p: p.unit if p else None)
	hubbleCroft['date'] = hubbleCroft['date'].apply(getdatetime)

	### Load keyword values
	keywordValues = pandas.read_csv(args.keyword)
	keywordValues['parsed'] = keywordValues['match'].apply(parse_measurement)
	keywordValues = keywordValues[keywordValues['parsed'].apply(lambda p: bool(p))]
	keywordValues['unit'] = keywordValues['parsed'].apply(lambda p: p.unit if p else None)
	keywordValues['date'] = keywordValues['date'].apply(getdatetime)

	### Load neural values
	neuralValues = pandas.read_csv(args.neural)
	neuralValues['parsed'] = neuralValues['value'].apply(parse_measurement)
	neuralValues = neuralValues[neuralValues['parsed'].apply(lambda p: bool(p))]
	neuralValues['unit'] = neuralValues['parsed'].apply(lambda p: p.unit if p else None)
	neuralValues['date'] = neuralValues['date'].apply(getdatetime)


	### Perform various filterings
	hubbleNeural = neuralValues[neuralValues['unit'].apply(lambda u: compatible(u,hubbleUnit))]
	hubbleNeural['parsed'] = hubbleNeural['parsed'].apply(lambda p: hubbleUnit(p))

	hubbleKeyword = keywordValues[keywordValues['unit'].apply(lambda u: compatible(u,hubbleUnit))]
	hubbleKeyword['parsed'] = hubbleKeyword['parsed'].apply(lambda p: hubbleUnit(p))

	hubbleKeyword['x'] = hubbleKeyword['parsed'].apply(lambda p: p.value)
	hubbleNeural['x'] = hubbleNeural['parsed'].apply(lambda p: p.value)

	hubbleKeywordUnc = hubbleKeyword[hubbleKeyword['parsed'].apply(lambda p: bool(p.uncertainties))]
	hubbleNeuralUnc = hubbleNeural[hubbleNeural['parsed'].apply(lambda p: bool(p.uncertainties))]

	def get_overlap(keyword,neural):

		keyword['x'] = keyword['parsed'].apply(lambda p: p.value)
		neural['x'] = neural['parsed'].apply(lambda p: p.value)

		overlap = pandas.merge(keyword, neural, on=['arxiv_id','x'], suffixes=('','_y'))

		keywordOnly = keyword.merge(overlap, on=['arxiv_id','x'], how='left', indicator=True, suffixes=('','_y'))
		keywordOnly = keywordOnly[keywordOnly['_merge'] == 'left_only']

		neuralOnly = neural.merge(overlap, on=['arxiv_id','x'], how='left', indicator=True, suffixes=('','_y'))
		neuralOnly = neuralOnly[neuralOnly['_merge'] == 'left_only']

		return keywordOnly,neuralOnly,overlap

	print('Loaded all datasets.')

	####
	#### Make pretty plots
	####
	def makefile(filename):
		if args.output:
			return os.path.join(args.output, filename)
		else:
			return filename
	def make_plots(name, data, labels, xbounds=None, ybounds=None, unit=None, markers=None,colors=None):

		print(f'{name} values: ' + ', '.join(str(d.shape[0]) for d in data))

		plot_width = 1.0

		bottom,left = 0.12,0.12
		width,height = 0.65,0.65
		bottom_xh = bottom + height + 0.02
		left_yh = left + width + 0.02
		height_h = 0.18

		rect_scatter = [plot_width*left, bottom, plot_width*width, height]
		rect_histx = [plot_width*left, bottom_xh, plot_width*width, height_h]
		rect_histy = [plot_width*left_yh, bottom, plot_width*height_h, height]

		fig = plt.figure(figsize=(5.16,5))

		axScatter = fig.add_axes(rect_scatter)
		axHistx = fig.add_axes(rect_histx)
		axHisty = fig.add_axes(rect_histy)

		if ybounds:
			data = [d[d['parsed'].apply(lambda p: ybounds[0] <= float(p) <= ybounds[1])] for d in data]

		def get_uncertainty(measurement):
			if measurement.uncertainties:
				poserr = sum([u.upper**2 for u in measurement.uncertainties])**0.5
				negerr = sum([u.lower**2 for u in measurement.uncertainties])**0.5
				return poserr,negerr
			else:
				return 0,0
		uncs = [[get_uncertainty(p) for p in d['parsed']] for d in data]

		measurements = [numpy.array([float(p) for p in d['parsed']]) for d in data]
		dates = [numpy.array([dt for dt in d['date']]) for d in data]

		#uncertainties = [numpy.array([[u[1] for ui in u],[u[0] for ui in u]])/d['confidence'].values for d,u in zip(data,uncs)]
		uncertainties = [numpy.array([[ui[1] for ui in u],[ui[0] for ui in u]]) for d,u in zip(data,uncs)]

		markers = markers if markers is not None else ['o' for d in data]

		plotlabels = labels if labels is not None else [None for m in markers]

		for dts,ms,u,mk,c,l in zip(dates,measurements,uncertainties,markers,colors,plotlabels):
			axScatter.errorbar(dts, ms, yerr=u, lw=0.5, alpha=0.5, marker=None, fmt="none", zorder=0, color=c)
			axScatter.scatter(dts, ms, s=10, marker=mk, alpha=0.5, zorder=100, color=c, label=l)

		if labels is not None: axScatter.legend(loc='lower left')

		if xbounds:
			#xmin,xmax = axScatter.get_xlim()
			#xlim = max(xmin,xbounds[0]),min(xmax,xbounds[1])
			#axScatter.set_xlim(xlim)
			axScatter.set_xlim(xbounds)
		if ybounds:
			#ymin,ymax = axScatter.get_ylim()
			#ylim = max(ymin,ybounds[0]),min(ymax,ybounds[1])
			#axScatter.set_ylim(ylim)
			axScatter.set_ylim(ybounds)

		#date_bins = numpy.array(list(months_range(dates.min(),dates.max(),inclusive=True,follow_on=True)))
		#date_bins = mpl.dates.date2num(date_bins)
		#dates = mpl.dates.date2num(dates)

		axHistx.hist(dates, bins='auto', stacked=True, color=colors) # bins=30
		axHisty.hist(measurements, bins='auto', orientation='horizontal', stacked=True, color=colors) # bins=50

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

	xbounds = getdatetime('1989-01-01 00:00:00'),getdatetime('2021-01-01 00:00:00')
	ybounds = (-5,105)

	cmap = matplotlib.cm.get_cmap('Set1')
	# Colors: Keyword, Neural, Both, Croft
	#cs=[cmap(0.0),cmap(1.0),cmap(0.5),cmap(0.25)]
	#cs = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#1f77b4']
	cs = ['darkorange','blue','darkorchid','orangered']

	#cs = [(57/255,106/255,177/255),(218/255,124/255,48/255),(62/255,150/255,81/255),(83/255,81/255,84/255)]

	colors = [cs[0],cs[1],cs[2]]

	make_plots(
			'parsed',
			get_overlap(keywordValues,neuralValues),
			['Keyword','Neural','Both'],
			xbounds=xbounds,
			ybounds=ybounds,
			unit=hubbleUnit,
			markers=['^','v','o'],
			colors=colors)
	make_plots(
			'parsedunc',
			get_overlap(
				keywordValues[keywordValues['parsed'].apply(lambda p: bool(p.uncertainties))],
				neuralValues[neuralValues['parsed'].apply(lambda p: bool(p.uncertainties))]
				),
			['Keyword','Neural','Both'],
			xbounds=xbounds,
			ybounds=ybounds,
			unit=hubbleUnit,
			markers=['^','v','o'],
			colors=colors)
	make_plots(
			'parsedunit',
			get_overlap(hubbleKeyword,hubbleNeural),
			['Keyword','Neural','Both'],
			xbounds=xbounds,
			ybounds=ybounds,
			unit=hubbleUnit,
			markers=['^','v','o'],
			colors=colors)
	make_plots(
			'parseunitunc',
			get_overlap(hubbleKeywordUnc,hubbleNeuralUnc),
			['Keyword','Neural','Both'],
			xbounds=xbounds,
			ybounds=ybounds,
			unit=hubbleUnit,
			markers=['^','v','o'],
			colors=colors)
	make_plots(
			'croftandneural',
			[hubbleCroft,hubbleNeuralUnc],
			['Croft&Dailey','Neural'],
			xbounds=xbounds,
			ybounds=ybounds,
			unit=hubbleUnit,
			markers=['D','v'],
			colors=[cs[3],cs[1]])

	make_plots(
			'keyword',
			[hubbleKeywordUnc],
			None,
			xbounds=(getdatetime('1994-01-01 00:00:00'),getdatetime('2020-12-31 00:00:00')),
			ybounds=ybounds,
			unit=hubbleUnit,
			markers=None,
			colors=['#1f77b4'])
