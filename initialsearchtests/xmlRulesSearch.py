import re
import sys
import os
import gaml.preprocessing.latexmlpy as latexml
import xml.etree.ElementTree as et

import gaml.metadata.oaipmh as oaipmh

global number
number = '(?<![_{])' + '(?<!\.)-?\d+(?:\.\d+)?' ## First part just makes sure it's not inside a tex math struct - not ideal solution


def shortestDistance(s1, e1, s2, e2):
	if s1 < s2:
		return abs(s2-e1)
	else:
		return abs(s1-e2)

def getSpan(s1, e1, s2, e2,offset=0):
	if s1 < s2:
		return (max(0,s1-offset),e2+offset)
	else:
		return (max(0,s2-offset),e1+offset)

def findValues(filepath, search):
	
	root = et.parse(filepath).getroot()

	foundmention = False

	for p in root.findall('.//p'): ## This is not ideal, as it can't cope with equations breaking up paragraphs
		for line in latexml.tostring(p).split('\n'):
			numbers = [(match.start(0), match.end(0), match.group(0)) for match in re.finditer(number, line)]

			if len(numbers) > 0:
				for match in re.finditer(search.keywords, line):
					foundmention = True

					min = sys.maxsize
					val = None
					span = None
					for num in numbers:
						dist = shortestDistance(match.start(0), match.end(0), num[0], num[1])
						if dist < min:
							min = dist
							val = num[2]
							span = getSpan(match.start(0), match.end(0), num[0], num[1],offset=10)
							## Here we also want to store span + neighbourhood
					identifier = oaipmh.arXivID(os.path.splitext(os.path.basename(filepath))[0])
					search.values[match.group()].append({
							'value': val,
							'mention': '...' + line[span[0]:span[1]].strip().replace('\n\n',' [NEWLINE] ').replace('\n',' ') + '...',
							'identifier': identifier,
							'date': str(metadata.get(identifier,field='date'))
						})

	search.filecounter += 1
	if foundmention:
		search.mentionfilecounter += 1


def processDirectory(directory, search):

	contents = [directory + os.sep + n for n in os.listdir(directory)]

	for c in contents:
		if os.path.isdir(c):
			processDirectory(c, search)
		else: # c is a file
			if c.endswith('.xml'):
				findValues(c, search)

class Search:
	def __init__(self, keywords, metadata):
		## Search information
		self.keywords = keywords
		self.metadata = metadata
		## Search results
		self.values = defaultdict(list)
		self.filecounter = 0
		self.mentionfilecounter = 0

def keywordSearch(source, keywords, metadata):
	results = Search(keywords, metadata)
	processDirectory(source, results)
	return results


if __name__ == "__main__":

	from collections import defaultdict

	import argparse
	from gaml.utilities.argparseactions import FileAction,DirectoryAction,RegexListAction

	import pprint

	parser = argparse.ArgumentParser(description="Use keyword search to find values of provided keywords in a corpus of LateXML documents.")
	parser.add_argument("sourcepath",action=DirectoryAction, mustexist=True, help='Path to source directory.')
	parser.add_argument("keywords", action=RegexListAction, help='Comma-separated list of keywords to search for.')
	parser.add_argument('-v','--valuesfilepath',action=FileAction, mustexist=False, help='File in which to store found values.')
	parser.add_argument('-s','--mentionsfilepath',action=FileAction, mustexist=False, help='File in which to store value mentions.')
	parser.add_argument('-m','--metadatapath',action=FileAction, mustexist=True, help='Path to metadata if not in standard location.')
	args = parser.parse_args()

	## Get metadata object
	metadata = oaipmh.ArXivMetadata(args.metadatapath) if args.metadatapath else oaipmh.ArXivMetadata(os.path.normpath(os.path.join(args.sourcepath,'../metadata.pickle')))

	## Find results
	results = keywordSearch(args.sourcepath, args.keywords, metadata)

	## Print statistics
	print(str(results.filecounter) + ' files searched.')
	print('Mentions found in ' + str(results.mentionfilecounter) + ' files.')

	allValues = []
	for keyword, wordvalues in results.values.items():
		allValues += wordvalues

	print(str(len(allValues)) + ' values found.')

	for key in results.values:
		print(str(key) + ': ' + str(len(results.values[key])))

	## Determine filepaths for saving results
	valuesfilepath = args.valuesfilepath if args.valuesfilepath else os.path.basename(args.sourcepath) + '_values.json'
	mentionsfilepath = args.mentionsfilepath if args.mentionsfilepath else os.path.basename(args.sourcepath) + '_mentions.txt'

	import json

	## Save results
	with open(valuesfilepath, "w") as valuesFile:
		json.dump(dict(results.values),valuesFile,indent=4)

	with open(mentionsfilepath, "w") as mentionsFile:
		for entrylist in results.values.values():
			for entry in entrylist:
				mentionsFile.write(entry['mention'] + '\n')





