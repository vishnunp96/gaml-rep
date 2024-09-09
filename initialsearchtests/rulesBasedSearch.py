import re
import sys
import os

## Global variables
global sourcepath
global keywords
global values
global valuesfilepath
global mentions
global mentionsfilepath
# Defined here
global number
number = '(?<![_{])' + '(?<!\.)-?\d+(?:\.\d+)?' ## First part just makes sure it's not inside a tex math struct - not ideal solution

# Global file counter
global filecounter
filecounter = 0
global mentionfilecounter
mentionfilecounter = 0

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

def findValues(filepath):

	global number
	global values
	global mentions

	with open(filepath, encoding='UTF-8') as file:
		text = file.read()

	numbers = [(match.start(0), match.end(0), match.group(0)) for match in re.finditer(number, text)]

	foundmention = False

	if len(numbers) > 0:
		#for keyword in keywords:

		for match in re.finditer(keywords, text):
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
			values[match.group()].append(val)
			#print(span[0].__class__)
			#print(span[1].__class__)
			mentions.append('...' + text[span[0]:span[1]].strip().replace('\n\n',' [NEWLINE] ').replace('\n',' ') + '...')

	global filecounter
	filecounter += 1
	if foundmention:
		global mentionfilecounter
		mentionfilecounter += 1


def processDirectory(directory):

	contents = [directory + os.sep + n for n in os.listdir(directory)]

	for c in contents:
		if os.path.isdir(c):
			processDirectory(c)
		else: # c is a file
			if c.endswith('.txt'):
				findValues(c)


if __name__ == "__main__":

	from collections import defaultdict
	import argparse
	from gaml.utilities.argparseactions import FileAction,DirectoryAction,RegexListAction

	parser = argparse.ArgumentParser(description="Use keyword search to find values of provided keywords in a corpus")
	parser.add_argument("sourcepath",action=DirectoryAction, mustexist=True, help='Path to source directory.')
	parser.add_argument("keywords", action=RegexListAction, help='Comma-separated list of keywords to search for.')
	parser.add_argument('-v','--valuesfilepath',action=FileAction, mustexist=False, help='File in which to store found values.')
	parser.add_argument('-m','--mentionsfilepath',action=FileAction, mustexist=False, help='File in which to store value mentions.')
	args = parser.parse_args()

	sourcepath = args.sourcepath
	#keywords = [k.strip() for k in sys.argv[2].split(',')]
	keywords = args.keywords

	#print(keywords)

	valuesfilepath = args.valuesfilepath if args.valuesfilepath else os.path.basename(sourcepath) + '_values.txt'
	mentionsfilepath = args.mentionsfilepath if args.mentionsfilepath else os.path.basename(sourcepath) + '_mentions.txt'

	# Initialise global values dict
	#values = {}
	values = defaultdict(list)
	#for keyword in args.keywords_strings:
	#	values[keyword] = []

	# Initialise mentions list
	mentions = []

	### Begin processing files
	processDirectory(sourcepath)

	with open(valuesfilepath, "w") as valuesFile:
		valuesFile.write(str(dict(values)))

	with open(mentionsfilepath, "w") as mentionsFile:
		mentionsFile.write('\n'.join(mentions))

	print(str(filecounter) + ' files searched.')
	print('Mentions found in ' + str(mentionfilecounter) + ' files.')

	allValues = []
	for keyword, wordvalues in values.items():
		allValues += wordvalues

	print(str(len(allValues)) + ' values found.')

	for key in values:
		print(str(key) + ': ' + str(len(values[key])))


