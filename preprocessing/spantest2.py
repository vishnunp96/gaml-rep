if __name__ == "__main__":

	import argparse
	from gaml.utilities.argparseactions import FileAction,DirectoryAction,RegexListAction

	parser = argparse.ArgumentParser()
	parser.add_argument('filepath',action=FileAction, mustexist=True)
	args = parser.parse_args()

	from lxml import etree as et
	from gaml.keywordsearch.rulesBasedSearch import fastxmliter
	import gaml.preprocessing.latexmlpy as latexml

	def describe(element):

		print('\n+++++++ Describe +++++++++\n')

		text,spans = latexml.tostring(element)

		print(text)
		print()
		print(latexml.format(element))

		print('\n******* SPANS ********')
		for span in spans:
			print()
			print(f'Span: {span}')
			print(text[span[0]:span[1]])

		print('\n******* LINES ********')
		for line,span,linespans in latexml.tolines(element):
			print()
			print(f'Span: {span}')
			print(f'Linespans: {linespans}')
			print(f'Line: {line}')

		print('\n******* SPANITER ********')
		for start,end in latexml.spaniter(text,'\n'):
			print(f'Start = {start}, End = {end}')
			print(text[start:end])

		print('\n******* POSITER ********')
		for pos in latexml.positer(text,'\n'):
			print(f'Pos = {pos}')


	ps = fastxmliter(args.filepath, events=("end",), tag='p')
	event,p1 = next(ps)
	describe(p1)
	event,p2 = next(ps)
	describe(p2)












