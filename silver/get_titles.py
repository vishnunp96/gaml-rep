if __name__ == "__main__":

	import re
	from utilities.argparseactions import ArgumentParser,PathAction
	from utilities import StopWatch
	from utilities.gxml import fastxmliter
	from utilities.fileutilities import iter_files
	import preprocessing.latexmlpy as latexml

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Use keyword search to find values of provided keywords in a corpus of LateXML documents.")
	parser.add_argument('source',action=PathAction, mustexist=True, allowed=['file','dir'], help='Source from which to pull titles (file or directory).')
	parser.add_argument('-r','--recursive',action='store_true', help='Output filepath.')
	args = parser.parse_args()

	for filepath in iter_files(args.source,recursive=args.recursive,suffix='.xml'):
		for event,title in fastxmliter(filepath, events=("end",), tag='title'):
			print(re.sub('\s+',' ',latexml.tostring(title)[0]).strip())
			break
