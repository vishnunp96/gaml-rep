if __name__ == "__main__":

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction
	from gaml.preprocessing import latexmlpy as latexml
	from gaml.utilities.fileutilities import changeext

	from lxml import etree as let

	import json
	from gaml.utilities.systemutilities import printacross

	parser = ArgumentParser(description='Create .txt and .spans files for LateXML documents.')
	parser.add_argument('xmlpath',action=IterFilesAction, mustexist=True, suffix='.xml', help='Path to xml file or directory to parse.')
	parser.add_argument('-e','--element', help='Element tag to take from document. By default, first element instance will be processed. If no elements of this type exist, or flag is unspecified, whole document will be processed. Ignored if force-manual flag set.')
	parser.add_argument('-n','--elemnumber', default=0, help='Element number (in document order) to take from the list of elements specified. Ignored if --element not given. Defaults to 0. Ignored if force-manual flag set.')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-m','--manual-on-failure',action='store_true',dest='on_fail')
	group.add_argument('-M','--force-manual',action='store_true',dest='force')
	args = parser.parse_args()

	for path in args.xmlpath:
		root = let.parse(path).getroot()

		manual = args.force or (args.on_fail and args.element)
		if args.element and not args.force:
			elems = root.findall(args.element)
			if elems:
				root = elems[args.elemnumber]
				manual = False
			else:
				print('Could not find element with tag \'' + args.element + '\' for ' + path)
		if manual:
			for i in root:
				text, spans = latexml.tostring(i)
				printacross('=')
				print('Element tag:',i.tag)
				print('Element text:')
				print(text)
				print()
				if input('Accept element? (y) ').lower().strip() == 'y':
					root = i
					break

		text, spans = latexml.tostring(root)

		txtpath = changeext(path,'.txt')
		spanpath = changeext(path,'.spans')
		annpath = changeext(path,'.ann')

		with open(txtpath,'w') as f:
			f.write(text)
		with open(spanpath,'w') as f:
			json.dump(spans, f, indent=2)
		open(annpath, 'a').close()
