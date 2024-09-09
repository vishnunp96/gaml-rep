if __name__ == '__main__':

	import sqlite3
	import numpy
	import pandas

	import matplotlib.pyplot as plt

	from gaml.utilities.argparseactions import ArgumentParser,FileAction
	from gaml.metadata.oaipmh import MetadataAction

	from gaml.annotations.database import query
	from gaml.parsing import parse_measurement,parse_unit
	from gaml.units.compatibility import compatible
	from gaml.utilities.dateutilities import getdatetime

	parser = ArgumentParser(description='Access data in database and plot, based on name and symbol queries.')
	parser.add_argument('database', action=FileAction, mustexist=True, help='Path to database.')
	parser.add_argument('metadata',action=MetadataAction, help='arXiv metadata file (pickled).')
	parser.add_argument('output', action=FileAction, mustexist=False, help='Path at which to save plot.')
	parser.add_argument('-s','--strict', action='store_true', help='Flag to indicate value selection should be strict (i.e. include both name and symbol).')
	parser.add_argument('-d','--dates',action=FileAction, mustexist=True, help='Important dates to include in plots.')
	parser.add_argument('-m','--markers',action='store_false',help='Toggle markers on pretty plot. Default true.')
	parser.add_argument('-n','--nbins',nargs='+',type=int,default=[100,100],help='Number of bins used in plotting histograms. If two numbers are provided, first is used for value axis, second for time axis. Defaults to [100,100].')
	parser.add_argument('--title',dest='titlesize',type=int,default=12,help='Font size for plot titles.')
	parser.add_argument('--label',dest='labelsize',type=int,default=10,help='Font size for plot axis titles.')
	parser.add_argument('--tick',dest='ticksize',type=int,default=10,help='Font size for plot tick labels.')
	boundsgroup = parser.add_mutually_exclusive_group()
	boundsgroup.add_argument('-b','--bounds',type=float,nargs=2,default=(float("-inf"),float("inf")),help='Lower and upper bounds on the values to consider from the source file.')
	boundsgroup.add_argument('-u','--upper',type=float,default=float("inf"),help='Upper bound of values to consider from source file.')
	boundsgroup.add_argument('-l','--lower',type=float,default=float("-inf"),help='Lower bound of values to consider from source file.')
	args = parser.parse_args()

	connection = sqlite3.connect(f'file:{args.database}?mode=ro',uri=True)
	cursor = connection.cursor()

	### GETTING DATA
	data = query(cursor,'SELECT * FROM all_measurements_values')

	print('Data:')
	print(data)

	if args.strict:
		hubble = data[data['name'].fillna('').str.match(r'^Hubble [Cc]onstant$') & data['symbol'].fillna('').str.match(r'^(H _ { 0 }|H _ { o }|H _ { \\circ })$')]
	else:
		hubble = data[data['name'].fillna('').str.match(r'^Hubble [Cc]onstant$') | data['symbol'].fillna('').str.match(r'^(H _ { 0 }|H _ { o }|H _ { \\circ })$')]

	print('Hubble:')
	print(hubble)

	hubble['parsed'] = hubble['value'].apply(parse_measurement)

	print('Hubble parsed:')
	print(hubble)

	unit = parse_unit('km/s/Mpc')

	print(f'Unit: {unit}')

	correct_units = hubble[hubble['parsed'].apply(lambda p: compatible(p.unit,unit) and bool(p.uncertainties) if p is not None else False)]

	print('Correct units:')
	print(correct_units)

	correct_units['parsed'] = correct_units['parsed'].apply(lambda i: unit(i))

	print('Correct units adjusted:')
	print(correct_units)

	print(correct_units.shape)

	##### PLOTTING CODE
	from matplotlib.ticker import NullFormatter
	nullfmt = NullFormatter()

	bounds = max(args.lower,args.bounds[0]),min(args.upper,args.bounds[1])

	in_bounds = correct_units[correct_units['parsed'].apply(lambda i: bounds[0] <= i.value <= bounds[1])]
	print(f'In bounds length: {in_bounds.shape}')

	def get_uncertainty(measurement):
		if measurement.uncertainties:
			poserr = sum([u.upper**2 for u in measurement.uncertainties])**0.5
			negerr = sum([u.lower**2 for u in measurement.uncertainties])**0.5
			return poserr,negerr
		else:
			return 0,0
	def get_value(row):
		parsed = row['parsed']
		return (
				args.metadata.get(row['arxiv_id'],field='date'),
				float(parsed.value),
				get_uncertainty(parsed)
			)
	values = [get_value(row) for i,row in in_bounds.iterrows()]
	dates = numpy.array([d for d,f,u in values])
	measurements = numpy.array([f for d,f,u in values])
	uncertainties = numpy.array([[u[1] for d,f,u in values],[u[0] for d,f,u in values]])

	plot_width = 1.0
	cbar_width = 1-plot_width

	bottom,left = 0.12,0.12
	width,height = 0.65,0.65
	bottom_xh = bottom + height + 0.02
	left_yh = left + width + 0.02
	height_h = 0.18

	left_c = 0.85
	width_c = 0.025

	rect_scatter = [plot_width*left, bottom, plot_width*width, height]
	rect_histx = [plot_width*left, bottom_xh, plot_width*width, height_h]
	rect_histy = [plot_width*left_yh, bottom, plot_width*height_h, height]
	rect_cbar = [left_c, bottom, width_c, height]

	fig = plt.figure(figsize=(7.225,7))
	cmap = None

	axScatter = fig.add_axes(rect_scatter)
	axHistx = fig.add_axes(rect_histx)
	axHisty = fig.add_axes(rect_histy)

	markers = numpy.array(['v' if u[0]==0 and u[1]==0 else 'o' for d,f,u in values])

	axScatter.errorbar(dates,measurements,yerr=uncertainties,lw=0.5,marker=None,fmt="none",zorder=0)

	if args.markers:
		#scatters = []
		for m in set(markers):
			mask = (markers == m)
			s = axScatter.scatter(dates[mask],measurements[mask],s=10,marker=m,alpha=0.5,zorder=100)
			#scatters.append(s)
		#axScatter.legend(scatters,('Has Uncertainty','No Uncertainty'),loc='lower left',bbox_to_anchor=(left_yh,bottom_xh))
	else:
		s = axScatter.scatter(dates,measurements,s=10,marker='o',alpha=0.5,zorder=100)

	ymin,ymax = axScatter.get_ylim()
	ylim = max(ymin,bounds[0]),min(ymax,bounds[1])
	axScatter.set_ylim(ylim)

	axHistx.hist(dates, bins=args.nbins[1])
	axHisty.hist(measurements, bins=args.nbins[0], orientation='horizontal')

	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())

	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)

	axHistx.yaxis.set_tick_params(labelsize=args.ticksize)
	axHisty.xaxis.set_tick_params(labelsize=args.ticksize)
	axScatter.xaxis.set_tick_params(labelsize=args.ticksize)
	axScatter.yaxis.set_tick_params(labelsize=args.ticksize)

	axScatter.set_xlabel('Date of Publication / Year',fontsize=args.labelsize,labelpad=args.labelsize)
	axScatter.set_ylabel('Extracted Value' + (f' / {unit.latex()}'),fontsize=args.labelsize)

	if args.dates:
		keydates = pandas.read_csv(args.dates, parse_dates=['date'], date_parser=getdatetime)
		minval,maxval = axScatter.get_ylim()
		y = 0.1*(maxval-minval)+minval
		for ax in (axHistx,axScatter):
			for index,row in keydates.iterrows():
				name,date = row
				axHistx.axvline(date,linestyle='--') ## Could 'annotate' this
				axScatter.axvline(date,linestyle='--') ## Could 'annotate' this
				axScatter.annotate(name,xy=(date,y),rotation='90',horizontalalignment='right',verticalalignment='bottom',fontsize=args.labelsize)

	fig.savefig(args.output)

	cursor.close()
	connection.close()
