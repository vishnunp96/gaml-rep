if __name__ == '__main__':

	from utilities.argparseactions import ArgumentParser,IterFilesAction

	from annotations.bratutils import Standoff

	from collections import Counter,defaultdict
	import itertools
	import os

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
	args = parser.parse_args()

	# Read in data
	anns = [(os.path.basename(os.path.splitext(path)[0]),Standoff.open(path)) for path in args.ann]
	anns = [(arXiv,a) for arXiv,a in anns if a]

	types = ['ParameterName','ParameterSymbol','ObjectName']

	counter = Counter(itertools.chain.from_iterable([[(e.type,e.text) for e in a.entities if e.type in types and len(e.text)>1] for arXiv,a in anns]))

	instances = defaultdict(set)

	for arXiv,a in anns:
		for e in a.entities:
			if e.type in types and len(e.text)>1:
				instances[(e.type,e.text)].add(arXiv)

	for (type_,text),value in counter.items():
		if value>5:
			print(f'{type_:20} {text:30} {value}')

	print('-------------------')

	for (type_,text),papers in instances.items():
		value = len(papers)
		if value>=3:
			print(f'{type_:20} {text:30} {value}')
