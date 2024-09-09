if __name__ == "__main__":

	import pandas
	import matplotlib.pyplot as plt
	#from matplotlib import style

	from utilities.argparseactions import ArgumentParser,FileAction

	parser = ArgumentParser()
	parser.add_argument('results',action=FileAction, mustexist=True,help="Year-scale arXiv statistics.")
	parser.add_argument('-o','--output',action=FileAction,help="Path at which to save figure.")
	parser.add_argument('--hide',dest='show',action='store_false',help='Flag to suppress showing pyplot output.')
	parser.add_argument('--label',dest='labelsize',type=int,default=10,help='Font size for plot axis titles.')
	parser.add_argument('--tick',dest='ticksize',type=int,default=10,help='Font size for plot tick labels.')
	args = parser.parse_args()

	results_year = pandas.read_csv(args.results,index_col='year')[:-1]

	#style.use('seaborn-dark-palette')

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.stackplot(results_year.index,results_year.values.T,labels=results_year.columns)
	ax.set_yscale('symlog',linthreshy=1)

	ax.set_xlabel('Publication Year',fontsize=args.labelsize)
	ax.set_ylabel('Stacked Count [Log Scale]',fontsize=args.labelsize)

	ax.set_xlim(results_year.index.values[0],results_year.index.values[-1])

	ax.xaxis.set_tick_params(labelsize=args.ticksize)
	ax.yaxis.set_tick_params(labelsize=args.ticksize)

	ax.legend(loc='lower right')
	#fig.legend(bbox_to_anchor=(1.04,1), loc="upper left")
	fig.tight_layout()

	if args.output:
		fig.savefig(args.output)

	if args.show: plt.show()
