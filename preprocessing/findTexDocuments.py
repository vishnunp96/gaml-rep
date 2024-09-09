import os, re
import regex
import utilities.filehandler as filehandler
from utilities.systemutilities import eprint
from utilities.utils import timeout
from utilities.fileutilities import listdir

import metadata.oaipmh as oaipmh

import preprocessing.latexmlpy as latexml

## Global variables
global sourcepath
global targetpath
global subjects
global metadata

global verbose

## Run information
global filecounter
filecounter = 0


def article_score(filepath, identifier):
	#print('Begin reading file.',flush=True)
	filetext = filehandler.readFile(filepath, allow_replace=True) ## This should attempt a proper decoding, or just replace with '?' if one cannot be found
	#print('Read file.',flush=True)

	scores = []
	scores.append(bool(re.search('^[^\%\n]*\\\\begin{document}', filetext, flags=re.MULTILINE)))
	scores.append(bool(re.search('abstract', filetext, flags=re.MULTILINE|re.IGNORECASE)))

	#print('Begin title search.', flush=True)
	global metadata
	scores.append(bool(timeout(regex.search,['(?e)(?:'+regex.escape(metadata.get(identifier,field='title'))+'){e<=10}', filetext],max_duration=10,default=None)))
	#print('End title search.',flush=True)

	return sum(scores)


def requestedSubject(identifier):

	global subjects
	global metadata

	if not subjects:
		return True
	else:
		## Do any of the specified matches appear in any of the papers subjects?
		categories = metadata.get(identifier,field='categories')
		if categories:
			smatch = re.search(subjects, categories) ## would find() be quicker?
			return bool(smatch)
		else:
			return False

def processDirectory(directory):

	if verbose: print("Directory: " + directory,flush=True)

	contents = listdir(directory)

	doctexfiles = []

	for c in contents:
		if os.path.isdir(c):
			processDirectory(c)
		elif os.path.isfile(c): # c is a file
			if c.endswith(".tex"):
				doctexfiles.append(c)

	identifier = os.path.basename(directory)
	if requestedSubject(identifier) and doctexfiles:
		if len(doctexfiles) > 1:
			#eprint('More than one (' + str(len(doctexfiles)) + ') tex document in ' + directory, flush=True)

			texfilepath = max(doctexfiles, key = lambda p: article_score(p,identifier))

			#eprint('Chose ' + texfilepath + ' for ' + directory, flush=True)

			#texfilepath = max(doctexfiles, key = lambda p: os.path.getsize(p))
		else:
			texfilepath = doctexfiles[0]

		try:
			processFile(texfilepath)
		except (KeyboardInterrupt, SystemExit):
			raise
		except:
			eprint('Failure for file ' + texfilepath)


def processFile(filepath):

	if verbose: print('File: ' + filepath, flush=True)

	element = latexml.opentex(filepath)
	if element is None:
		eprint('Could not parse ' + filepath)
		return

	targetfilepath = os.path.dirname(filepath.replace(sourcepath,targetpath)) + ".xml"
	destination = os.path.dirname(targetfilepath)

	if not os.path.exists(destination):
		os.makedirs(destination)

	if verbose: print('Save to: ' + targetfilepath)
	latexml.savexml(element, targetfilepath)

	if os.path.isfile(targetfilepath): # File created successfully - this is probably overkill
		global filecounter
		filecounter += 1


if __name__ == "__main__":

	import argparse
	from utilities.argparseactions import FileAction,DirectoryAction,RegexListAction

	parser = argparse.ArgumentParser(description="Recursively process directories containing .tex files into .xml (preserving source directory structure in target).")
	parser.add_argument("sourcepath",action=DirectoryAction, mustexist=True, help='Path to source directory.')
	parser.add_argument("targetpath",action=DirectoryAction, mustexist=False, mkdirs=True, help='Path to directory where .xml files will be saved.')
	parser.add_argument('-s','--subjects',action=RegexListAction, help='Comma-separated list of subjects to include.')
	parser.add_argument('-m','--metadatapath',action=FileAction, mustexist=True, help='Path to metadata if not in standard location.')
	parser.add_argument('-v','--verbose',action="store_true", help='Verbose output of actions.')
	args = parser.parse_args()

	sourcepath = args.sourcepath
	targetpath = args.targetpath
	subjects = args.subjects

	if args.metadatapath:
		metadata = oaipmh.ArXivMetadata(args.metadatapath)
	else:
		metadata = oaipmh.ArXivMetadata(os.path.normpath(os.path.join(sourcepath,'../metadata.pickle')))

	verbose = args.verbose

	processDirectory(sourcepath)

	print('Completed: ' + str(filecounter) + ' files created.')

