import numpy
from matplotlib.ticker import Locator

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
	Author: https://stackoverflow.com/a/20495928
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = numpy.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(numpy.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

def symlogbins(minval,maxval,nbins,threshold):

	if minval < -threshold:

		negrange = numpy.log10(-minval) - numpy.log10(threshold)
		posrange = numpy.log10(maxval) - numpy.log10(threshold)
		linrange = 2
		density = nbins / (negrange + posrange + linrange)

		negnum = int(round(negrange * density))
		posnum = int(round(posrange * density))
		linnum = int(round(linrange * density))

		negetive = -numpy.logspace(numpy.log10(-minval),numpy.log10(threshold),negnum)
		linear = numpy.linspace(-threshold,threshold,linnum)
		positive = numpy.logspace(numpy.log10(threshold),numpy.log10(maxval),posnum)
	elif minval <= 0:

		posrange = numpy.log10(maxval) - numpy.log10(threshold)
		linrange =  1 + (-minval/threshold)
		density = nbins / (posrange + linrange)

		posnum = int(round(posrange * density))
		linnum = int(round(linrange * density))

		negetive = numpy.array([])
		linear = numpy.linspace(minval,threshold,linnum)
		positive = numpy.logspace(numpy.log10(threshold),numpy.log10(maxval),posnum)
	else:
		negetive = numpy.array([])
		linear = numpy.array([])
		positive = 	numpy.logspace(numpy.log10(minval),numpy.log10(maxval),nbins)

	return numpy.concatenate((negetive,linear,positive))


if __name__ == "__main__":

	from scipy import stats
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import pandas

	from parsing import parse_measurement,parse_unit

	from utilities.argparseactions import ArgumentParser,FileAction
	from utilities.dateutilities import getdatetime
	from utilities.jsonutils import load_json,dump_json

	from collections import defaultdict

	from units.exception import IncompatibleUnitsError

	parser = ArgumentParser(description="Plot histogram of values in file.")
	parser.add_argument('sourcepath',action=FileAction, mustexist=True, help='Values to plot.')
	boundsgroup = parser.add_mutually_exclusive_group()
	boundsgroup.add_argument('-b','--bounds',type=float,nargs=2,default=(float("-inf"),float("inf")),help='Lower and upper bounds on the values to consider from the source file.')
	boundsgroup.add_argument('-u','--upper',type=float,default=float("inf"),help='Upper bound of values to consider from source file.')
	boundsgroup.add_argument('-l','--lower',type=float,default=float("-inf"),help='Lower bound of values to consider from source file.')
	parser.add_argument('-i','--interesting',type=float,nargs=2,help='Lower and upper bounds on interesting values to save to file.')
	parser.add_argument('-n','--nbins',nargs='+',type=int,default=[100,100],help='Number of bins used in plotting histograms. If two numbers are provided, first is used for value axis, second for time axis. Defaults to [100,100].')
	parser.add_argument('-t','--threshold',type=float,default=1e-4,help='Linear region threshold for symlog histogram plotting. Defaults to 1e-4.')
	parser.add_argument('-H','--histogram',action=FileAction,nargs='?',help='Plot histogram of values.')
	parser.add_argument('-T','--timeseries',action='store_true',help='Plot time series of values.')
	parser.add_argument('-L','--symlog',action='store_true',help='Plot log-histogram of values.')
	parser.add_argument('-P','--pretty',action=FileAction,nargs='?',help='Plot scatter/histogram of values.')
	parser.add_argument('-E','--errorbars',action='store_true',help='Plot plot error bars on scatter/histogram plot.')
	parser.add_argument('-D','--unit',help='Unit which measurements must conform to.')
	parser.add_argument('-U','--uncertainty',action='store_true',help='Flag to require uncertainty in measurements.')
	parser.add_argument('-s','--section',help='LateXML element tag which measurement must fall under.')
	parser.add_argument('-x','--vlim',type=float,nargs=2,help='Limits on the measurement axes of the plots.')
	parser.add_argument('-y','--tlim',type=getdatetime,nargs=2,help='Limits on the time axes of the plots.')
	parser.add_argument('-p','--predictions',action=FileAction, mustexist=True, help='New measurement predictions for the scatter plot.')
	parser.add_argument('-d','--dates',action=FileAction, mustexist=True, help='Important dates to include in plots.')
	parser.add_argument('-S','--splithist',action=FileAction,nargs='?',help='Plot split histograms using dates provided.')
	parser.add_argument('-o','--outofbounds',action='store_true',help='Print out of bounds entries.')
	parser.add_argument('--display',action='store_true',help='Print accepted entries.')
	parser.add_argument('-m','--markers',action='store_false',help='Toggle markers on pretty plot. Default true.')
	parser.add_argument('--table',action=FileAction,mustexist=False,help='Write out table of values matching criteria to this path.')
	parser.add_argument('--web',action=FileAction,mustexist=False,help='Write out table of values matching criteria to this path, formatted for web usage.')
	parser.add_argument('--true-value',nargs='+',help='True value for use constructing uncertainty plot. --uncertainty must also be specified. If a second argument is provided to this flag, it is interpreted as a path in which to store plot.')
	parser.add_argument('--hide',dest='show',action='store_false',help='Flag to suppress showing pyplot output.')
	parser.add_argument('--title',dest='titlesize',type=int,default=12,help='Font size for plot titles.')
	parser.add_argument('--label',dest='labelsize',type=int,default=10,help='Font size for plot axis titles.')
	parser.add_argument('--tick',dest='ticksize',type=int,default=10,help='Font size for plot tick labels.')
	args = parser.parse_args()

	def make_pair(l,type=None):
		if type:
			l = [type(i) for i in l]
		if len(l) == 2:
			return tuple(l)
		elif len(l) == 1:
			return (l[0],l[0])
		raise ValueError(f'Must take 1 or 2 length argument, not {len(l)}.')
	args.nbins = make_pair(args.nbins,type=int)

	valuesDict = load_json(args.sourcepath)

	unit = parse_unit(args.unit) if args.unit else None

	predictions = None
	if args.predictions:
		predictions = pandas.read_csv(args.predictions,index_col='arXiv')
		print(f'Loaded predictions (length {predictions.size}).')

	def get_uncertainty(measurement):
		if measurement.uncertainties:
			poserr = sum([u.upper**2 for u in measurement.uncertainties])**0.5
			negerr = sum([u.lower**2 for u in measurement.uncertainties])**0.5
			return poserr,negerr
		else:
			return 0,0

	def get_prediction(identifier,default=0.5):
		if args.predictions:
			return predictions.prediction.get(identifier,default)
		else:
			return default

	bounds = max(args.lower,args.bounds[0]),min(args.upper,args.bounds[1])

	keywords = []
	all_outofbounds = []
	parsefail = 0
	parseattempt = 0
	all_accepted = []
	all_dimensions = defaultdict(int)
	for keyword, entrylist in valuesDict.items():
		keywords.append(keyword)
		for entry in entrylist:
			if args.section and not any(args.section in s for s in [o[2] for o in entry['origins']]):
				continue

			parseattempt += 1
			parsed = parse_measurement(entry['match'])
			if not parsed:
				parsefail += 1
				#print(f'Parse failed: {entry["match"]}\n\t{entry["mention"].encode("ascii",errors="replace").decode("ascii")}')
				#print(f'Parse failed: {entry["match"]}')
				continue
			elif parsed.unit:
				all_dimensions[str(parsed.unit.canonical())] += 1
			else:
				all_dimensions[None] += 1
				#print(entry['match'],'->',parsed)

			if unit is not None and parsed:
				try:
					parsed = unit(parsed)
				except IncompatibleUnitsError:
					parsed = None

			if args.uncertainty and parsed and not parsed.uncertainties:
				parsed = None

			if parsed and bounds[0] <= float(parsed.value) <= bounds[1]:
				all_accepted.append((entry,parsed))
			elif parsed:
				all_outofbounds.append((entry,parsed))

	#from pprint import pprint
	#pprint(dict(all_dimensions))

	def remove_duplicates(all_vals):
		value_dict = defaultdict(list)
		for e,p in all_vals:
			#value_dict[(e['identifier'],p.value,get_uncertainty(p))].append((e,p))
			value_dict[(e['identifier'],e['match'])].append((e,p))
		reduced_vals = [min(l,key=lambda i: len(i[0]['mention'])) for l in value_dict.values()]
		return reduced_vals

	def get_value(entry,parsed):
		return (
				getdatetime(entry['date']),
				float(parsed.value),
				get_prediction(entry['identifier'],default=0.5),
				get_uncertainty(parsed)
			)

	outofbounds = remove_duplicates(all_outofbounds)
	final_accepted = remove_duplicates(all_accepted)
	values = [get_value(e,p) for e,p in final_accepted]
	entries = [e for e,p in final_accepted]
	articles = set(e['identifier'] for e in entries)

	print(f'{len(all_accepted)} values found and {len(entries)} accepted from {len(articles)} papers.')
	print(f'{len(outofbounds)} out-of-bounds values found (without duplicates) ({len(all_outofbounds)} total).')

	#for e,p in final_accepted: print(f'{p} ({e["identifier"]})')

	valuecount = 0
	rangecount = 0
	for entry in entries:
		if 'value_text' in entry and entry['value_text']:
			valuecount += 1
		elif 'range_text' in entry and entry['range_text']:
			rangecount += 1

	if args.interesting:
		interesting = [(e,p) for e,p in final_accepted if args.interesting[0] <= float(p.value) <= args.interesting[1]]

	print(f'Attempted to parse {parseattempt} entries, {parsefail} failures.' + (f' Using unit {unit}.' if args.unit else ''))
	print(f'Central values: {valuecount}, Range count: {rangecount}')

	mean = numpy.mean([f for d,f,c,u in values])
	std = numpy.std([f for d,f,c,u in values])
	mode = stats.mode([f for d,f,c,u in values])
	median = numpy.median([f for d,f,c,u in values])

	print('Mean = ' + str(mean) + ', std.dev. = ' + str(std) + ', ' + str(len(values)) + ' values.')
	print('Median = ' + str(median) + ', Mode = ' + str(mode))
	print(f'From {len(articles)} articles.')

	#x = [(d,f,c,u) for d,f,c,u in values if bounds[0] <= f <= bounds[1]]

	print(f'{len(outofbounds)} value{"s" if len(outofbounds)>1 else ""} lie{"" if len(outofbounds)>1 else "s"} outside bounds ({len(values)+len(outofbounds)}-{len(values)}, {100*len(outofbounds)/(len(values)+len(outofbounds)):.2f}%).')
	if args.outofbounds:
		for entry,parsed in outofbounds:
				print('{:<20} {:<14.2f} {}'.format(entry['identifier'],parsed.value,entry['mention']).encode('ascii',errors='ignore').decode())

	if args.display:
		for entry,parsed in final_accepted:
				print('{:<20} {:<14.2f} {}'.format(entry['identifier'],parsed.value,entry['mention']).encode('ascii',errors='ignore').decode())

	dates = numpy.array([d for d,f,c,u in values])
	measurements = numpy.array([f for d,f,c,u in values])
	uncertainties = numpy.array([[u[1] for d,f,c,u in values],[u[0] for d,f,c,u in values]])
	colors = numpy.array([c for d,f,c,u in values])

	if args.interesting:
		print(f'Interesting range: {args.interesting}')
		with open('interestingvalues.txt','wb') as f:
			for entry,parsed in interesting:
				#pprint(entry)
				#print('Interesting: ',entry['value'])
				#f.write('{:<25} {:<12.10s} {}\n'.format(entry['identifier'],str(entry['value']),entry['mention']).encode('ascii',errors='ignore'))
				f.write('{:<25} {:<12.10s} {}\n'.format(entry['identifier'],str(parsed.value),entry['mention']).encode('ascii',errors='ignore'))

	if args.table:
		def row_dict(entry,parsed):
			row = {}
			row['Normalised'] = str(parsed)
			row['Value'] = parsed.value
			for i,u in enumerate(parsed.uncertainties):
				row['Uncertainty_'+str(i)+'_upper'] = u.upper
				row['Uncertainty_'+str(i)+'_lower'] = u.lower
			row['arXiv'] = entry['identifier']
			row['Date'] = getdatetime(entry['date'])
			row['match'] = entry['match']
			return row

		pandas.DataFrame([row_dict(e,p) for e,p in final_accepted]).to_csv(args.table,index=False)

	if args.web:
		def datapoint_dict(entry,parsed):
			row = {}
			row['normalised'] = str(parsed)
			row['value'] = parsed.value
			uncertainty = get_uncertainty(parsed)
			row['posunc'] = uncertainty[0]
			row['negunc'] = uncertainty[1]
			row['arxiv'] = entry['identifier']
			row['date'] = str(getdatetime(entry['date']).date())
			row['match'] = entry['match']
			return row
		dump_json([datapoint_dict(e,p) for e,p in final_accepted], args.web, indent=1)

	if args.histogram:

		fig = plt.figure()
		ax = fig.add_subplot('111')
		ax.hist(measurements, args.nbins[0])
		#plt.title(','.join(keywords))
		ax.set_xlabel('Measurement Value')
		ax.set_ylabel('Counts')
		if args.vlim:
			ax.set_xlim(args.vlim)
		fig.tight_layout()

		fig = plt.figure()
		ax = fig.add_subplot('111')
		ax.hist(dates, args.nbins[1])
		#plt.title(','.join(keywords))
		ax.set_xlabel('Date')
		ax.set_ylabel('Counts')
		if args.tlim:
			ax.set_xlim(args.tlim)
		fig.tight_layout()

		if args.histogram_given:
			fig.savefig(args.histogram)

	if args.timeseries:
		fig = plt.figure()
		ax = fig.add_subplot('111')
		if args.predictions:
			cmap = plt.get_cmap('plasma_r')
			norm = mpl.colors.Normalize(vmin=colors.min(),vmax=colors.max())
			sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
			sm.set_array([])

			s = ax.scatter(dates,measurements,alpha=0.5,c=colors,cmap=cmap)

			fig.colorbar(sm, ax=ax)
		else:
			ax.scatter(dates,measurements,alpha=0.5)

		#plt.gca().set_yscale('symlog', linthreshx=args.threshold)
		#plt.gca().yaxis.set_minor_locator(MinorSymLogLocator(args.threshold))
		#plt.locator_params(axis='y', numticks=10)

		#plt.title(','.join(keywords))
		ax.set_xlabel('Date of Publication')
		ax.set_ylabel('Measurement Value')
		if args.vlim:
			ax.set_ylim(args.vlim)
		if args.tlim:
			ax.set_xlim(args.tlim)

		fig.tight_layout()

	if args.symlog:
		fig = plt.figure()
		ax = fig.add_subplot('111')

		bins = symlogbins(measurements.min(), measurements.max(), nbins=args.nbins[0], threshold=args.threshold)
		#for b in numpy.nditer(bins): print(b)
		ax.hist(measurements, bins=bins)

		ax.set_xscale('symlog', linthreshx=args.threshold)
		ax.xaxis.set_minor_locator(MinorSymLogLocator(args.threshold))
		ax.locator_params(axis='x', numticks=10)
		ax.set_yscale("log", nonposy='clip')

		#plt.title(','.join(keywords))
		ax.set_xlabel('Measurement Value')
		ax.set_ylabel('Counts')

		if args.vlim:
			ax.set_xlim(args.vlim)

		fig.tight_layout()

	if args.uncertainty and args.true_value:

		import matplotlib.mlab as mlab

		fig = plt.figure()
		ax = fig.add_subplot('111')

		true_value = float(args.true_value[0])

		unc = numpy.array([u[0] if true_value<m else u[1] for m,u in zip(measurements,uncertainties.T)]) # u[0] is lower,u[1] is upper
		m = measurements

		if args.tlim:
			unc = unc[(args.tlim[0] < dates) & (dates < args.tlim[1])]
			m = m[(args.tlim[0] < dates) & (dates < args.tlim[1])]

		n_sigma = (m-true_value)/numpy.abs(unc)
		n_sigma = n_sigma[(-5<n_sigma) & (n_sigma<5)]

		binwidth = 0.45
		#bins = numpy.arange(min(n_sigma),max(n_sigma)+binwidth,binwidth)
		bins = numpy.arange(-5,5+binwidth,binwidth)

		ax.hist(n_sigma,bins=bins,density=True)
		ax.set_xlim((-5,5))
		x = numpy.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
		ax.plot(x,mlab.normpdf(x, 0, 1))
		ax.axvline(0,linestyle='--')

		ax.set_xlabel('$n_{\sigma}$',fontsize=args.labelsize,labelpad=args.labelsize)
		ax.set_ylabel('Frequency',fontsize=args.labelsize,labelpad=args.labelsize)

		ax.xaxis.set_tick_params(labelsize=args.ticksize)
		ax.yaxis.set_tick_params(labelsize=args.ticksize)

		fig.tight_layout()

		if len(args.true_value) > 1:
			fig.savefig(args.true_value[1])
	elif args.true_value and not args.uncertainty:
		print('Cannot construct uncertainty plot without requiring uncertainty.')

	if args.pretty:

		from matplotlib.ticker import NullFormatter

		nullfmt = NullFormatter()

		plot_width = 0.85 if args.predictions else 1.0
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

		fig = plt.figure(figsize=(8.5 if args.predictions else 7.225,7))
		cmap = plt.get_cmap('plasma_r') if args.predictions else None

		axScatter = fig.add_axes(rect_scatter)
		axHistx = fig.add_axes(rect_histx)
		axHisty = fig.add_axes(rect_histy)
		if args.predictions:
			axCbar = fig.add_axes(rect_cbar)

		markers = numpy.array(['v' if u[0]==0 and u[1]==0 else 'o' for d,f,c,u in values])

		if args.errorbars:
			axScatter.errorbar(dates,measurements,yerr=uncertainties,lw=0.5,marker=None,fmt="none",zorder=0)

		if args.markers:
			#scatters = []
			for m in set(markers):
				mask = (markers == m)
				if args.predictions:
					s = axScatter.scatter(dates[mask],measurements[mask],c=colors[mask],s=10,marker=m,cmap=cmap,alpha=0.5,zorder=100)
				else:
					s = axScatter.scatter(dates[mask],measurements[mask],s=10,marker=m,alpha=0.5,zorder=100)
				#scatters.append(s)
			#axScatter.legend(scatters,('Has Uncertainty','No Uncertainty'),loc='lower left',bbox_to_anchor=(left_yh,bottom_xh))
		else:
			if args.predictions:
				s = axScatter.scatter(dates,measurements,c=colors,s=10,marker='o',cmap=cmap,alpha=0.5,zorder=100)
			else:
				s = axScatter.scatter(dates,measurements,s=10,marker='o',alpha=0.5,zorder=100)

		ymin,ymax = axScatter.get_ylim()
		ylim = max(ymin,bounds[0]),min(ymax,bounds[1])
		axScatter.set_ylim(ylim)

		#date_bins = numpy.array(list(months_range(dates.min(),dates.max(),inclusive=True,follow_on=True)))
		#date_bins = mpl.dates.date2num(date_bins)
		#dates = mpl.dates.date2num(dates)

		if args.predictions:
			norm = mpl.colors.Normalize(vmin=colors.min(),vmax=colors.max())
			sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
			sm.set_array([])

			nm_mask = colors.round()==1
			colors = [cmap(0.9),cmap(0.1)]

			axHistx.hist([dates[nm_mask],dates[~nm_mask]], bins=args.nbins[1],stacked=True,color=colors)
			axHisty.hist([measurements[nm_mask],measurements[~nm_mask]], bins=args.nbins[0],stacked=True,color=colors, orientation='horizontal')
		else:
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

		if args.predictions:
			cbar = fig.colorbar(sm,cax=axCbar)
			cbar.ax.tick_params(labelsize=args.ticksize)

		axScatter.set_xlabel('Date of Publication / Year',fontsize=args.labelsize,labelpad=args.labelsize)
		axScatter.set_ylabel('Extracted Value' + (f' / {unit.latex()}' if args.unit else ''),fontsize=args.labelsize)
		#axScatter.set_ylabel('Measurement Value')

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

		if args.pretty_given:
			fig.savefig(args.pretty)

	if args.dates and args.splithist:

		from math import ceil,floor

		keydates = pandas.read_csv(args.dates, parse_dates=['date'], date_parser=getdatetime).sort_values(by='date')

		histargs = {'bins': numpy.linspace(floor(measurements.min()),ceil(measurements.max()),args.nbins[0])}

		fig = plt.figure(figsize=(5,7))

		name,date = keydates.iloc[0]
		ax = fig.add_subplot(len(keydates)+1,1,1)
		ax.hist(measurements[dates<date],**histargs)

		ax.set_title(f'Pre {name}',fontsize=args.titlesize)

		axlist = [ax]

		for i,row in keydates.iterrows():
			name,date = row
			ax = fig.add_subplot(len(keydates)+1,1,i+2)

			if i==len(keydates)-1:
				ax.hist(measurements[date<=dates],**histargs)
				ax.set_title(f'Post {name}',fontsize=args.titlesize)
				ax.set_xlabel('Measurement Value' + (' / km s$^{-1}$Mpc$^{-1}$' if args.unit else ''),fontsize=args.labelsize)
				#ax.set_xlabel('Measurement Value')
			else:
				ax.hist(measurements[numpy.logical_and(date<=dates,dates<keydates.iloc[i+1].date)],**histargs)
				ax.set_title(f'{name} to {keydates.iloc[i+1]["name"]}',fontsize=args.titlesize)

			axlist.append(ax)

		xlim = (min([a.get_xlim()[0] for a in axlist]),max([a.get_xlim()[1] for a in axlist]))
		for ax in axlist:
			ax.set_xlim(xlim)
			ax.set_ylabel('Count',fontsize=args.labelsize)

			ax.xaxis.set_tick_params(labelsize=args.ticksize)
			ax.yaxis.set_tick_params(labelsize=args.ticksize)

		fig.tight_layout()

		if args.splithist_given:
			fig.savefig(args.splithist)
	elif args.splithist and not args.dates:
		print("No dates file provided - cannot print split histograms.")

	if args.show: plt.show()
