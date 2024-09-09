if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	from gaml.utilities.argparseactions import ArgumentParser,PathAction,DirectoryAction
	from gaml.metadata.oaipmh import MetadataAction
	from gaml.preprocessing.manifest import ManifestAction,Manifest

	import random,os
	from sklearn.model_selection import train_test_split

	from gaml.silver.scorenewmeasurements import get_scores
	from gaml.silver.vocabulary import find_vocab,token_vector
	from gaml.utilities.fileutilities import iter_files
	from gaml.utilities.jsonutils import dump_json
	from gaml.utilities.bsonutils import dump_bson
	from gaml.utilities.parallel import Pool

	parser = ArgumentParser()
	parser.add_argument('source',action=PathAction)
	parser.add_argument('metadata',action=MetadataAction,help='ArXiv metadata.')
	parser.add_argument('-m','--manifest',action=ManifestAction,default=('source',lambda s: Manifest(s)),help='Data manifest.')
	parser.add_argument('resultsdir',action=DirectoryAction,mustexist=True)
	parser.add_argument('-v','--minvocabcount',type=int,default=1)
	parser.add_argument('-s','--minarticlescore',type=int,default=3)
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=1000)
	args = parser.parse_args()

	resultsdir = os.path.join(args.resultsdir, os.path.basename(args.source)+'_dataset')
	os.makedirs(resultsdir,exist_ok=True)

	## Score articles
	scores = get_scores(
			iter_files(args.source, recursive=True, suffix='.xml'),
			args.metadata, processes=args.processes, chunksize=args.chunksize
		)
	stopwatch.tick('Calculate article scores.')

	## Find positive/negative samples
	pos_silver = []
	neg_silver = []
	for arXiv,score in scores.items():
		if score >= args.minarticlescore:
			pos_silver.append(arXiv)
		elif score == 0:
			neg_silver.append(arXiv)

	negative_set = random.sample(neg_silver,len(pos_silver))
	print(f'Silver: positive size = {len(pos_silver)}, negative size = {len(neg_silver)}')

	x = pos_silver + negative_set
	y = [1]*len(pos_silver) + [0]*len(negative_set)

	stopwatch.tick('Determined positive and negative samples.')

	## Make train/test split
	x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

	stopwatch.tick('Finished train/test split.')

	## Find vocab from training samples
	vocabulary = find_vocab((args.manifest[i] for i in x_train), args.minvocabcount, processes=args.processes, chunksize=args.chunksize)
	print(f'Vocabulary size = {len(vocabulary)}')
	dump_json(vocabulary,os.path.join(resultsdir,'vocabulary.json'),indent='\t')

	stopwatch.tick('Found and saved vocabulary of training set.')

	## Construct vector form of documents
	with Pool(processes=args.processes) as p:
		x_train_vectors = p.starmap(token_vector, ((args.manifest[i],vocabulary) for i in x_train), chunksize=args.chunksize)
		x_test_vectors = p.starmap(token_vector, ((args.manifest[i],vocabulary) for i in x_test), chunksize=args.chunksize)

	stopwatch.tick('Construct document vectors.')

	## Save test/train sets in json/bson format
	dump_bson(
			{
				'train': [i for i in zip(x_train_vectors,y_train)],
				'test': [i for i in zip(x_test_vectors,y_test)],
				'vocabulary_size': len(vocabulary)
			},
			os.path.join(resultsdir,'dataset.bson')
		)

	stopwatch.tick('Save test/train sets to file.')

	stopwatch.report()
