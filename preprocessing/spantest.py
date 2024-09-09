if __name__ == "__main__":

	import argparse
	from gaml.utilities.argparseactions import FileAction,DirectoryAction,RegexListAction

	parser = argparse.ArgumentParser()
	parser.add_argument('filepath',action=FileAction, mustexist=True)
	args = parser.parse_args()

	from lxml import etree as et
	from gaml.preprocessing import latexmlpy as latexml

	with open(args.filepath,'r') as f:
		data = et.parse(f)
	root = data.getroot()

	element = latexml.sections(root)[1]

	text,spans = latexml.tostring(element)

	print(text)
	print()
	print(latexml.format(element))

	for span in spans:
		print()
		print(f'Span: {span}')
		print(text[span[0]:span[1]])

	for line,span,linespans in latexml.tolines(element):
		print()
		print(f'Span: {span}')
		print(linespans)
		print(line)
