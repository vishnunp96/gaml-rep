if __name__ == "__main__":

	import os
	import numpy
	import math
	from localstats import mean
	from datetime import datetime,timedelta
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import scipy.stats
	from matplotlib.ticker import NullFormatter
	import matplotlib.lines as mlines
	import pandas

	from units.dimensionless import DimensionlessUnit
	from parsing import parse_measurement,parse_unit

	from utilities.argparseactions import ArgumentParser,DirectoryAction
	from utilities.dateutilities import getdatetime
	from utilities.fileutilities import listdir
	from utilities.jsonutils import load_json

	from astroML.density_estimation import XDGMM

	parser = ArgumentParser(description="Plot histogram of values in file.")
	parser.add_argument('datadir',action=DirectoryAction, mustexist=True, help='Directory containing data to plot.')
	parser.add_argument('-o','--outdir',action=DirectoryAction, mustexist=False, mkdirs=True, help='Directory in which to place plots.')
	parser.add_argument('-e','--extension',default='.pdf',help='File extension to use.')
	parser.add_argument('-f','--fit',action='store_true',help='Flag to indicate that trendline fits should be plotted.')
	args = parser.parse_args()

	fileext = args.extension

	def makefile(filename):
		if args.outdir:
			return os.path.join(args.outdir, filename)
		else:
			return filename

	def load_data():
		#values = pandas.read_csv(filename)
		#values['parsed'] = values['parsed'].apply(parse_measurement)
		#values['date'] = values['date'].apply(getdatetime)
		#return values

		for filename in (f for f in listdir(args.datadir) if not os.path.basename(f).startswith('.')):
			name = os.path.splitext(os.path.basename(filename))[0]

			print(f'Load {name}')

			data = load_json(filename)
			values = pandas.DataFrame(data['data'])
			values['parsed'] = values['parsed'].apply(parse_measurement)

			values = values[values['parsed'].apply(bool)]

			values['date'] = values['date'].apply(getdatetime)

			bounds = data['bounds']
			datebounds = tuple(getdatetime(d) for d in data['datebounds']) if data['datebounds'] else None
			unit = parse_unit(data['unit']) if data['unit'] else DimensionlessUnit(1)
			reference = data['reference'] if 'reference' in data else None

			latex_name = data['latex']

			yield name,latex_name,values,bounds,datebounds,unit,reference

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

	def daterange_quarterly(start, end):
		for year in range(start.year, end.year+1):
			#for month in range(1,13):
			for month in [1,4,7,10]:
				if (year==start.year and month<start.month) or (year==end.year and month>end.month):
					continue
				yield datetime(year,month,1)

	def daterange_yearly(start, end):
		for year in range(start.year, end.year+1):
			month = 1
			if (year==start.year and month<start.month) or (year==end.year and month>end.month):
				continue
			yield datetime(year,month,1)

	def bin_data_past(data, period, daterange):
		## Period is timedelta object
		mindate = data['date'].min() + period
		maxdate = data['date'].max()
		for date in daterange(mindate, maxdate):
			yield date, data[(data['date'] >= (date-period)) & (data['date'] <= date)]

	def bin_data_moving(data, period, daterange):
		## Period is timedelta object
		mindate = data['date'].min()
		maxdate = data['date'].max()
		halfp = period/2
		for date in daterange(mindate, maxdate):
			yield date, data[(data['date'] >= (date-halfp)) & (data['date'] <= (date+halfp))]

	def robust_mean(data):
		data = data.loc[data['parsed'].apply(float).sort_values().index]
		crop = math.floor(0.05*len(data))
		if crop > 0: data = data[crop:len(data)-crop]
		average = data['parsed'].apply(float).mean()
		unc = data.apply(lambda row: mean(get_uncertainty(row['parsed'],row['confidence'],row['bound'])),axis='columns').mean()
		return average,unc

	def median(data):
		m = data['parsed'].apply(float).mean()
		return m,0

	def get_trendline(data, period, bin_data, daterange, func):
		dates = []
		values = []
		uncs = []
		for date,period_data in bin_data(data, period, daterange):
			if len(period_data) > 0:
				average,unc = func(period_data)
				dates.append(date)
				values.append(average)
				uncs.append(unc)
			elif len(values) > 0:
				dates.append(date)
				values.append(values[-1])
				uncs.append(uncs[-1])

		return numpy.array(dates),numpy.array(values),numpy.array(uncs)

	def calc_xd(data, n_components, max_iter=100, tol=1e-05, random_state=42):

		x = numpy.array([float(p) for p in data['parsed']])
		x_err = numpy.array([u for u in data.apply(lambda row: mean(get_uncertainty(row['parsed'],row['confidence'],row['bound'])),axis='columns')])

		x = x.reshape(-1,1)
		x_err = x_err.reshape(-1,1,1)

		xd = XDGMM(n_components, max_iter=max_iter, tol=tol, random_state=random_state)
		xd.fit(x, x_err)

		## This logic is taken from the sklearn GMM source code
		_, n_features = xd.mu.shape
		cov_params = xd.n_components * n_features * (n_features + 1) / 2.
		mean_params = n_features * xd.n_components
		free_params = int(cov_params + mean_params + xd.n_components - 1)

		k = free_params
		lnL = xd.logL(x, x_err) # Hopefully this is the correct one?
		aic = 2*k - 2*lnL

		return xd,aic

	method_colors = {
		'CMB': 'w',
		'LSS': 'k',
		'Pec.Vels.': 'r',
		'SN': 'lime',
		'Lensing': 'b',
		'BBN': 'cyan',
		'Clusters': 'magenta',
		'BAO': 'yellow',
		'ISW': 'orange',
		'z dist.': 'dimgrey',
		'Other': 'silver',
		'Unknown': 'brown'
		}
	def get_method_colors(methods):
		return numpy.array([method_colors[m] for m in methods])
	def get_method_legend_handles():
		handles = [mlines.Line2D([], [], color=c, marker='o', ls='', markeredgewidth=0.5, markeredgecolor='k', markersize=5, label=k) for k,c in method_colors.items()]
		return handles
		#ax.legend(handles=get_method_legend_handles())


	def make_plots(name, latex_name, data, bounds=None, datebounds=None, unit=None, reference=None, trendtype='', trend_func=None, fit=False, show_method=False):

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
		if datebounds:
			data = data[data['date'].apply(lambda d: datebounds[0] <= d <= datebounds[1])]

		upperlimits = numpy.array([b == 'U' for b in data['bound']])
		lowerlimits = numpy.array([b == 'L' for b in data['bound']])
		constraints = upperlimits | lowerlimits

		uncertainties = numpy.array([get_uncertainty(p,c,b) for p,c,b in zip(data['parsed'],data['confidence'],data['bound'])]).T

		measurements = numpy.array([float(p) for p in data['parsed']])
		#dates = numpy.array(list(data['date'].values)) # numpy.array([getdatetime(d) for d in data['date']])
		dates = data['date'].dt.to_pydatetime()

		if show_method:
			methodology = list(data['methodology'])
		else:
			methodology = None

		print(f'{name} values: {len(dates)}')

		# Plot central values
		axScatter.errorbar(dates[~constraints],measurements[~constraints],yerr=uncertainties.T[~constraints].T,lw=0.5,alpha=0.5,marker=None,fmt="none",zorder=0)
		if methodology:
			axScatter.scatter(dates[~constraints],measurements[~constraints],s=15,marker='o',c=get_method_colors(methodology)[~constraints],edgecolor='k',lw=0.7,alpha=0.75,zorder=100)
		else:
			axScatter.scatter(dates[~constraints],measurements[~constraints],s=10,marker='o',alpha=0.5,zorder=100)

		# Plot constraint values
		#ymin,ymax = axScatter.get_ylim()
		#linelength = (ymax - ymin) * 0.05
		#axScatter.errorbar(dates[constraints],measurements[constraints],yerr=uncertainties.T[constraints].T*linelength,uplims=upperlimits[constraints],lolims=lowerlimits[constraints],lw=0.5,marker=None,fmt="none",capsize=2,zorder=0)
		#axScatter.scatter(dates[constraints],measurements[constraints],s=40,marker='_',alpha=0.5,zorder=100)

		# Plot constraint values with arrow markers
		if methodology:
			axScatter.scatter(dates[upperlimits],measurements[upperlimits],s=30,marker='v',c=get_method_colors(methodology)[upperlimits],edgecolor='k',lw=0.7,alpha=0.75,zorder=100)
		else:
			axScatter.scatter(dates[upperlimits],measurements[upperlimits],s=30,marker='v',alpha=0.5,zorder=100)
		if methodology:
			axScatter.scatter(dates[lowerlimits],measurements[lowerlimits],s=30,marker='^',c=get_method_colors(methodology)[lowerlimits],edgecolor='k',lw=0.7,alpha=0.75,zorder=100)
		else:
			axScatter.scatter(dates[lowerlimits],measurements[lowerlimits],s=30,marker='^',alpha=0.5,zorder=100)

		# Draw reference line
		if reference is not None:
			axScatter.axhline(reference,lw=0.75,linestyle='--',c='k',zorder=110)

		if trend_func and trendtype:
			years = 5
			trend_data = data[data['bound'].apply(lambda b: b in trendtype)]
			if len(trend_data) > 0:
				trend_x, trend_y, trend_u = get_trendline(trend_data,timedelta(weeks=years*52),bin_data_moving,daterange_quarterly,trend_func)
				axScatter.plot(trend_x, trend_y, zorder=101, alpha=0.7)
				if trend_u.sum() > 0:
					axScatter.fill_between(trend_x, trend_y-trend_u, trend_y+trend_u, zorder=50, alpha=0.2)

		if fit:
			years = 5
			fit_data = data[data['bound'].apply(lambda b: b in fit)]
			if len(fit_data) > 1:
				fit_dates = []
				fit_means = []
				fit_uncs = []
				fit_vals = []
				fit_models = []
				for date,period_data in bin_data_moving(fit_data, timedelta(weeks=years*52), daterange_yearly):
					if len(period_data) > 1:
						print(name + ' fit: ' + str(date) + f' ({len(period_data)})')
						xd, aic = min((calc_xd(period_data, n) for n in [1,2]), key=lambda x: x[1])
						for i in range(xd.n_components):
							print(f'{xd.mu[i,0]:10.4f}, {xd.V[i,0,0]:10.4f}, {math.sqrt(xd.V[i,0,0]):10.4f}, {xd.alpha[i]:10.4f}')
							if xd.alpha[i] > 0.1:
								fit_dates.append(date)
								fit_means.append(xd.mu[i,0])
								fit_uncs.append(math.sqrt(xd.V[i,0,0]))
						fit_vals.append(period_data)
						fit_models.append(xd)
				fit_dates = numpy.array(fit_dates)
				fit_means = numpy.array(fit_means)
				fit_uncs = numpy.array(fit_uncs)
				axScatter.errorbar(fit_dates,fit_means,yerr=fit_uncs,lw=1,alpha=1,marker=None,fmt="none",zorder=140)
				axScatter.scatter(fit_dates,fit_means,s=20,marker='o',alpha=1,zorder=150)

				hist_fig, hist_axs = plt.subplots(nrows=len(fit_vals), ncols=1, sharex=True, squeeze=True, figsize=(5,len(fit_vals)))
				xvals = fit_data['parsed'].apply(float)
				xmin,xmax = xvals.min(), xvals.max()
				xdelta = 0.1 * (xmax - xmin)
				x = numpy.linspace(xmin - xdelta, xmax+ xdelta, 2000)
				for ax,date,vals,xd in zip(hist_axs, fit_dates, fit_vals, fit_models):
					ax.hist(vals['parsed'].apply(float),bins='auto',density=True)
					for i in range(xd.n_components):
						ax.plot(x,scipy.stats.norm.pdf(x, xd.mu[i,0], math.sqrt(xd.V[i,0,0]))*xd.alpha[i])
					if reference is not None:
						ax.axvline(reference,linestyle='--',lw=0.75,c='k')
				hist_fig.tight_layout()
				hist_fig_filename = makefile(name+'_fits'+fileext)
				hist_fig.savefig(hist_fig_filename)


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

		x_tick_locator = plt.MaxNLocator(6)
		axScatter.xaxis.set_major_locator(x_tick_locator)

		axHistx.hist(dates, bins='auto')
		axHisty.hist(measurements, bins='auto', orientation='horizontal')

		axHistx.set_xlim(axScatter.get_xlim())
		axHisty.set_ylim(axScatter.get_ylim())

		nullfmt = NullFormatter()
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)

		axHistx.xaxis.set_major_locator(x_tick_locator)

		#axScatter.locator_params(axis='x', nbins=6)
		#axHistx.locator_params(axis='x', nbins=6)

		ticksize = 10
		axHistx.yaxis.set_tick_params(labelsize=ticksize)
		axHisty.xaxis.set_tick_params(labelsize=ticksize)
		axScatter.xaxis.set_tick_params(labelsize=ticksize)
		axScatter.yaxis.set_tick_params(labelsize=ticksize)

		import matplotlib.dates as mdates
		myFmt = mdates.DateFormatter('%Y')
		axScatter.xaxis.set_major_formatter(myFmt)

		labelsize = 10
		paramtext = f'${latex_name}$'
		if isinstance(unit,DimensionlessUnit):
			paramtext += ' / Dimensionless'
		elif unit:
			paramtext += f' / {unit.latex()}'
		axScatter.set_xlabel('Date of Publication / Year',fontsize=labelsize,labelpad=labelsize)
		axScatter.set_ylabel('Extracted ' + paramtext,fontsize=labelsize)
		#axScatter.set_ylabel('Measurement Value')

		if methodology:
			axScatter.legend(handles=get_method_legend_handles(), loc='upper right', fontsize=6, ncol=2).set_zorder(999)


		hist_filename = makefile(name+fileext)
		fig.savefig(hist_filename)


	def plot_errors(name, data, bounds=None, datebounds=None, unit=None):
		## Plot error distribution figure

		if bounds:
			data = data[data['parsed'].apply(lambda p: bounds[0] <= float(p) <= bounds[1])]
		if datebounds:
			data = data[data['date'].apply(lambda d: datebounds[0] <= d <= datebounds[1])]

		upperlimits = numpy.array([b == 'U' for b in data['bound']])
		lowerlimits = numpy.array([b == 'L' for b in data['bound']])
		constraints = upperlimits | lowerlimits

		uncertainties = numpy.array([get_uncertainty(p,c,b) for p,c,b in zip(data['parsed'],data['confidence'],data['bound'])]).T

		measurements = numpy.array([float(p) for p in data['parsed']])

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
		ax.plot(x,scipy.stats.norm.pdf(x, 0, 1))
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


	cached_values = dict()
	for name,latex_name,values,bounds,datebounds,unit,reference in load_data():
		cached_values[name] = values

		values_unc = values[values['bound'].apply(lambda b: b not in 'UL')]

		# Basic plot
		make_plots(name, latex_name, values, bounds=bounds, datebounds=datebounds, unit=unit, reference=reference)
		# Basic plot with only central values
		make_plots(name+'_unc', latex_name, values_unc, bounds=bounds, datebounds=datebounds, unit=unit, reference=reference)
		# Basic plot showing methodology, if available
		if 'methodology' in values:
			make_plots(name+'_method', latex_name, values, bounds=bounds, datebounds=datebounds, unit=unit, reference=reference, show_method=True)

		if args.fit:
			# Basic plot with median trendline (for each bound type)
			for b in 'CUL':
				make_plots(name+'_'+b+'trend_median', latex_name, values, bounds=bounds, datebounds=datebounds, unit=unit, reference=reference, trendtype=b, trend_func=median)
			# Basic plot with mean trendline (for each bound type)
			for b in 'CUL':
				make_plots(name+'_'+b+'trend_mean', latex_name, values, bounds=bounds, datebounds=datebounds, unit=unit, reference=reference, trendtype=b, trend_func=robust_mean)

			# Plot with XD fit
			make_plots(name+'_xd', latex_name, values_unc, bounds=bounds, datebounds=datebounds, unit=unit, reference=reference, fit='C')

		# Errors plot
		plot_errors(name, values, bounds=bounds, datebounds=datebounds, unit=unit)

