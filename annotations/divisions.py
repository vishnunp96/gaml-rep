if __name__ == "__main__":

	import os
	from random import shuffle
	from statistics import mean

	import pandas
	from itertools import combinations
	from lxml import etree as let
	import json

	from gaml.preprocessing import latexmlpy as latexml
	from gaml.preprocessing.manifest import ManifestAction

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from gaml.utilities.filehandler import readFile
	from gaml.utilities.fileutilities import changeext

	from gaml.utilities import StopWatch
	stopwatch = StopWatch()

	parser = ArgumentParser(description='Divide articles up amongst annotators, with redundancy.')
	parser.add_argument('source',action=ManifestAction, help='Source directory for XML documents, with manifest file.')
	parser.add_argument('data',action=FileAction,mustexist=True,help='Data to divide up. A single column will be used as a key (defaults to \'id\').')
	parser.add_argument('annotators',action=FileAction,mustexist=True,help='File containing list of annotators.')
	parser.add_argument('output',action=DirectoryAction,mustexist=True,help='Output directory.')
	parser.add_argument('name',help='Name of this set.')
	parser.add_argument('-c','--column',default='id',help='Column to use as key for data. Defaults to \'id\'.')
	parser.add_argument('-a','--agreement',default=3,type=int,help='Number of annotators for each sample.')
	parser.add_argument('-s','--shuffle',action='store_true',help='Shuffle data before sampling. Defaults to false.')
	parser.add_argument('-e','--element', help='Element tag to take from document. By default, first element instance will be processed. If no elements of this type exist, or flag is unspecified, whole document will be processed.')
	parser.add_argument('-n','--elemnumber', default=0, help='Element index (in document order) to take from the list of elements specified. Ignored if --element not given. Defaults to 0.')
	parser.add_argument('-m','--make',action='store_true',help='Make annotation directories (directories containing xml, txt, and standoff files) for each annotator. Defaults to false.')
	args = parser.parse_args()

	data = pandas.read_csv(args.data)
	if args.shuffle:
		data = data.sample(frac=1)

	annotators = [l for l in readFile(args.annotators).splitlines() if l]

	all_groupings = list(combinations(annotators,args.agreement))
	shuffle(all_groupings)
	counts = {pair:0 for pair in all_groupings}
	per_grouping = len(data.index)/len(all_groupings)
	per_annotator = len(data.index)*args.agreement/len(annotators)

	all_annotations = []
	individual_annotations = {name:[] for name in annotators}

	i = 0
	groupings = all_groupings.copy()
	for d in data[args.column]:
		group = groupings[i%len(groupings)]
		all_annotations.append([d] + list(group))
		counts[group] += 1

		for name in group:
			individual_annotations[name].append(d)
			if len(individual_annotations[name])>=per_annotator and len(groupings)>1:
				groupings = [g for g in groupings if name not in g]

		i += 1

	col_names = [f'annotator{i+1}' for i in range(args.agreement)]

	summary = pandas.DataFrame(all_annotations,columns=['data']+col_names)
	#print(df)
	#print(per_pair)
	#print(counts)
	#print(summary.groupby(col_names).size().reset_index().rename(columns={0:'count'}))
	#summary.to_csv(os.path.join(args.output,'summary.csv'),index=False)

	for name,annotations in individual_annotations.items():
		annotatordata = pandas.DataFrame(annotations,columns=[args.column])

		if args.make:
			annotatordir = os.path.join(args.output,name,args.name)
			if not os.path.exists(annotatordir): os.makedirs(annotatordir)

			for arXiv in annotatordata[args.column]:
				path = args.source[arXiv]

				if path:
					root = let.parse(path).getroot()

					if args.element:
						elems = root.findall('.//'+args.element)
						if elems:
							root = elems[args.elemnumber]
						else:
							print('Could not find element with tag \'' + args.element + '\' for ' + path)

					text, spans = latexml.tostring(root)

					txtpath = os.path.join(annotatordir,changeext(os.path.basename(path),'.txt'))
					spanpath = os.path.join(annotatordir,changeext(os.path.basename(path),'.spans'))
					annpath = os.path.join(annotatordir,changeext(os.path.basename(path),'.ann'))

					with open(txtpath,'w',encoding='utf-8') as f:
						f.write(text)
					with open(spanpath,'w') as f:
						json.dump(spans, f, indent=2)
					open(annpath, 'a').close()
				else:
					print(f'No good path found for {arXiv}')

		annotatordata.to_csv(os.path.join(args.output,f'{args.name}-{name}.csv'),index=False)

	summary.to_csv(os.path.join(args.output,f'{args.name}-SUMMARY.csv'),index=False)

	print(f'{len(data.index)} annotations, {len(annotators)} annotators')
	print(f'{per_annotator:.1f} ({mean([len(a) for a in individual_annotations.values()])}) per annotator ({per_grouping:.1f} per pair)')
	stopwatch.report()
