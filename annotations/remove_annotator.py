if __name__ == "__main__":

	import os
	from shutil import copyfile

	import pandas

	from utilities.argparseactions import ArgumentParser,DirectoryAction,ListAction

	from utilities import StopWatch
	stopwatch = StopWatch()

	parser = ArgumentParser(description='Redistribute annotator abstracts amongst other annotators. Note, this does not remove the annotators files. This should be done manually.')
	parser.add_argument('source',action=DirectoryAction, help='Directory containing subdirectories for each annotator in project, along with summary files for each sample and annotator.')
	parser.add_argument('remove',help='Name of annotator to remove.')
	parser.add_argument('annotators',action=ListAction,help='Comma separated list of annotators to replace removed.')
	parser.add_argument('sample',help='Name of this sample.')
	parser.add_argument('-a','--agreement',default=3,type=int,help='Number of annotators for each sample.')
	parser.add_argument('-s','--shuffle',action='store_true',help='Shuffle data before sampling. Defaults to false.')
	args = parser.parse_args()

	annotator_summary_paths = {a:os.path.join(args.source,f'{args.sample}-{a}.csv') for a in args.annotators}
	annotator_summaries = {a:pandas.read_csv(p) for a,p in annotator_summary_paths.items()}

	for a,s in annotator_summaries.items():
		print(f'{a}: {len(s)}')
	print('Annotators total:',sum(len(d) for a,d in annotator_summaries.items()))

	remove_summary_path = os.path.join(args.source,f'{args.sample}-{args.remove}.csv')
	remove_summary = pandas.read_csv(remove_summary_path)

	sample_summary_path = os.path.join(args.source,f'{args.sample}-SUMMARY.csv')
	sample_summary = pandas.read_csv(sample_summary_path, index_col='data')

	print('Remove total:',len(remove_summary))

	i = 0
	for arXiv in remove_summary['id']:
		#annotator = args.annotators[i%len(args.annotators)]
		original_annotators = sample_summary.loc[arXiv].to_dict()
		annotator = min((i for i in args.annotators if i not in original_annotators.values()), key=lambda a: len(annotator_summaries[a]))
		print(f'Move {arXiv} to {annotator} ({original_annotators}, {annotator in original_annotators.values()})')
		annotator_summaries[annotator] = annotator_summaries[annotator].append({'id':arXiv}, ignore_index=True)
		remove_annotator_col = next(k for k,v in original_annotators.items() if v == args.remove)
		print(remove_annotator_col)
		sample_summary.at[arXiv,remove_annotator_col] = annotator
		new_annotators = sample_summary.loc[arXiv].to_dict()
		startpath = os.path.join(args.source,args.remove,args.sample,arXiv.replace('/',''))
		destpath = os.path.join(args.source,annotator,args.sample,arXiv.replace('/',''))
		print(new_annotators)
		print(startpath)
		print(destpath)
		i += 1
		print(', '.join(f'{a}: {len(s)}' for a,s in annotator_summaries.items()))

		for ext in ['.ann','.txt','.spans']:
			copyfile(startpath+ext,destpath+ext)

	for a,s in annotator_summaries.items():
		print(f'{a}: {len(s)}')

	print(sum(len(d) for a,d in annotator_summaries.items()) - len(remove_summary))

	for a,s in annotator_summaries.items():
		s.to_csv(annotator_summary_paths[a],index=False)
	sample_summary.to_csv(sample_summary_path,index=False)
