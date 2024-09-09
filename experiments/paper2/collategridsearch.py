if __name__ == '__main__':

	import pandas

	from utilities.argparseactions import ArgumentParser,IterFilesAction
	import os

	parser = ArgumentParser(description='Collect results of hyperparameter gridsearch for paper 2 into individual dataframes.')
	parser.add_argument('models',action=IterFilesAction,recursive=True,suffix='.pt',help='Directory containing model files with accompanying stats CSVs to collate.')
	args = parser.parse_args()

	rundir = args.models_path
	if not os.path.isdir(rundir):
		raise ValueError('Must be a directory.')

	stats = {'entity': [], 'relation': [], 'attribute': []}

	for path in args.models:
		data = stats[next(iter([i for i in stats.keys() if i in path]))]

		csvpath = os.path.join(os.path.dirname(path),'overall_metrics.csv')

		if not os.path.isfile(csvpath):
			raise ValueError(f'Could not find {csvpath}')

		modelstats = pandas.read_csv(csvpath)[['score','value']].set_index('score').T.reset_index(drop=True).assign(file=[path]).iloc[0].to_dict()

		data.append(modelstats)

	stats = {key:pandas.DataFrame(data) for key,data in stats.items()}

	best_metric = 'f1micro'

	for key,data in stats.items():
		data['model'] = data['file'].apply(lambda f: os.path.splitext(f.split('/')[-1])[0])
		data['type'] = data['model'].apply(lambda m: m.split('_')[0])
		data.to_csv(os.path.join(rundir,f'{key}stats.csv'), index=False)

		best = data.sort_values(best_metric,ascending=False).reset_index(drop=True).groupby('type').first().reset_index().sort_values(best_metric,ascending=False)
		best.to_csv(os.path.join(rundir,f'{key}stats_best.csv'), index=False)
