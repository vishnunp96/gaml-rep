import re
import preprocessing.latexmlpy as latexml
from utilities.terminalutils import printacross

def conclusion_title(s):
	return bool(re.search('conclusions?|summary',s,flags=re.IGNORECASE))

def print_conclusion(filepath):
	tree = latexml.openxml(filepath,getroot=False)
	conclusion_paths = [
			tree.getpath(i) + '/..'
			for i in tree.findall('//title')
			if
				conclusion_title(latexml.tostring(i)[0]) and
				i.getparent().tag == 'section']

	if conclusion_paths:
		for path in conclusion_paths:
			printacross('=')
			for elem in tree.xpath(path):
				text,spans = latexml.tostring(elem)
				print(text.strip().encode('ascii',errors='replace').decode('ascii',errors='replace'))
				printacross('*')
	else:
		print('No conclusion found.')

if __name__ == "__main__":

	from utilities.argparseactions import ArgumentParser,FileAction
	from utilities import StopWatch

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Find conclusion sections of LaTeXML articles.")
	parser.add_argument('article',action=FileAction, mustexist=True, help='Article to process.')
	args = parser.parse_args()

	print_conclusion(args.article)
