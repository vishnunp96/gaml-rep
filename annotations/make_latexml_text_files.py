import os.path

if __name__ == "__main__":

	from utilities.argparseactions import ArgumentParser,IterFilesAction, FileAction
	from preprocessing import latexmlpy as latexml
	from utilities.fileutilities import changeext

	from lxml import etree as let

	from utilities.systemutilities import printacross

	parser = ArgumentParser(description='Create .txt files for LateXML documents.')
	parser.add_argument('xmlpath',action=IterFilesAction, recursive=True, suffix='.xml', help='Path to xml file or directory to parse.')
	parser.add_argument('outdir',action=FileAction, mustexist=False, help='Output location for text files.')
	parser.add_argument('-e','--element', help='Element tag to take from document. By default, first element instance will be processed. If no elements of this type exist, or flag is unspecified, whole document will be processed. Ignored if force-manual flag set.')
	parser.add_argument('-n','--elemnumber', default=0, help='Element number (in document order) to take from the list of elements specified. Ignored if --element not given. Defaults to 0. Ignored if force-manual flag set.')
	args = parser.parse_args()

	numfiles = 0
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

		text, _ = latexml.tostring(root)
		filename = os.path.basename(path)
		txtdir = args.outdir
		txtpath = os.path.join(txtdir, changeext(filename, '.txt'))

		with open(txtpath,'w') as f:
			print("Writing to --> ", txtpath)
			if numfiles % 100 == 0:
				print("Number of files written: ", numfiles, flush=True)
			f.write(text)
		numfiles += 1
	print("Number of files written: ", numfiles, flush=True)
