if __name__ == "__main__":

	from utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from preprocessing import latexmlpy as latexml
	from preprocessing.manifest import ManifestAction
	from utilities.fileutilities import changeext

	from lxml import etree as let
	import pandas

	import json
	import os

	parser = ArgumentParser(description='Create .txt and .spans files for LateXML documents.')
	parser.add_argument('listing',action=FileAction, mustexist=True, help='Path to csv file containing \'id\' column of arXiv ids to be selected.')
	parser.add_argument('source',action=ManifestAction, help='Source directory for XML documents, with manifest file.')
	parser.add_argument('destination',action=DirectoryAction, mustexist=False, mkdirs=True, help='Directory to contain annotation documents.')
	parser.add_argument('-e','--element', help='Element tag to take from document. By default, first element instance will be processed. If no elements of this type exist, or flag is unspecified, whole document will be processed.')
	parser.add_argument('-n','--elemnumber', default=0, help='Element index (in document order) to take from the list of elements specified. Ignored if --element not given. Defaults to 0.')
	parser.add_argument('-c','--column', default='id', help='Column name containing arXiv IDs. Defaults to \'id\'.')
	parser.add_argument('--overwrite', action='store_true', help='Flag to overwrite any existing annotation files (.ann) in directory.')
	args = parser.parse_args()

	data = pandas.read_csv(args.listing)

	for arXiv in data['id']:
		path = args.source[arXiv]

		root = let.parse(path).getroot()

		if args.element:
			elems = root.findall('.//'+args.element)
			if elems:
				root = elems[args.elemnumber]
			else:
				print('Could not find element with tag \'' + args.element + '\' for ' + path)

		text, spans = latexml.tostring(root)

		txtpath = os.path.join(args.destination,changeext(os.path.basename(path),'.txt'))
		spanpath = os.path.join(args.destination,changeext(os.path.basename(path),'.spans'))
		annpath = os.path.join(args.destination,changeext(os.path.basename(path),'.ann'))

		with open(txtpath,'w',encoding='utf-8') as f:
			f.write(text)
		with open(spanpath,'w') as f:
			json.dump(spans, f, indent=2)
		open(annpath, 'w' if args.overwrite else 'a').close()
