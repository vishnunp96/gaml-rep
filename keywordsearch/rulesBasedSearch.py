import re
import sys
import itertools

import gaml.preprocessing.latexmlpy as latexml
import gaml.metadata.oaipmh as oaipmh

from collections import defaultdict

import gaml.utilities.parallel as parallel
from gaml.utilities.iterutilities import iterator_slice
from gaml.keywordsearch.filters import cullMatch
from gaml.utilities.numberutils import str_to_float
from gaml.preprocessing.mathtokenise import regularize_math

numberre = re.compile('(?<![\^_{] )(?<![\-0-9\.{])-?\d+(?:\.\d+)?(?:\ [0]+)*(?![\.\-}0-9])(?!\ \})') ## Include features for new regularise_math text
#numberre = re.compile('(?<![\^_{\-0-9\.])-?\d+(?:\.\d+)?(?:\ [0]+)*') ## Includes trailing zeroes
#numberre = re.compile('(?<![\^_{\-0-9\.])-?\d+(?:\.\d+)?')

def shortest_distance(s1, e1, s2, e2):
	if s1 < s2:
		return abs(s2-e1)
	else:
		return abs(s1-e2)

def get_text_span(s1, e1, s2, e2,offset=0):
	if s1 < s2:
		return (max(0,s1-offset),e2+offset)
	else:
		return (max(0,s2-offset),e1+offset)

def is_overlapping(s1,e1,s2,e2):
	if s1 <= s2:
		return s2 < e1
	else:
		return s1 < e2

def findInFile(filepath, keywords, metadata, search=None, tag='p'):

	if search is None: search = Search()

	#print('Start:',filepath)

	foundmention = False

	identifier = oaipmh.arXivID_from_path(filepath)
	date = str(metadata.get(identifier,field='date'))

	#root = let.parse(filepath).getroot()

	##foundtag = False
	#for event,p in fastxmliter(filepath, events=("end",), tag=tag):
	#	#foundtag = True
	#	for line,span,linespans in latexml.tolines(p):
	#		foundintext = findInText_measurement(line, span, linespans, search, keywords, identifier, date)
	#		foundmention = foundmention or foundintext

	for line,span,linespans in latexml.tolinesiter(filepath,tag):
		foundintext = findInText_measurement(line, span, linespans, search, keywords, identifier, date)
		foundmention = foundmention or foundintext

	#if not foundtag: print(f'No tag: {filepath}')
	#elif not foundmention: print(f'No val: {filepath}')

	search.filecounter += 1
	if foundmention:
		search.mentionfilecounter += 1

	#print('End:',filepath)

	return search


def findInText_number(text, textspan, encompassingspans, search, keywords, identifier, date):

	#print('Start:',text)

	foundmention = False

	keywordmatches = [i for i in re.finditer(keywords, text)]

	if keywordmatches:
		#print('Text:',text)
		#print('Keywords:',keywordmatches)
		#print('Initial:',[i for i in numberre.finditer(text)])
		all_numbers = [n for n in numberre.finditer(text) if not any(is_overlapping(n.start(),n.end(),k.start(),k.end()) for k in keywordmatches)]
		#print('Non-overlapping:',all_numbers)

		## Has to be a list, because we need to reuse it for each keyword instance
		numbers = [(m.start(0), m.end(0), m.group(0)) for m in itertools.filterfalse(cullMatch,all_numbers)]
		#print('Final:',numbers)

		for match in keywordmatches:
			min = sys.maxsize
			val = None
			span = None
			for num in numbers:
				dist = shortest_distance(match.start(0), match.end(0), num[0], num[1])
				if dist < min:
					min = dist
					val = num
			if val:
				foundmention = True

				span = get_text_span(match.start(0), match.end(0), val[0], val[1],offset=30)

				search.values[match.group()].append({
						'value': str_to_float(val[2]),
						'value_text': val[2],
						'mention': '...' + text[span[0]:span[1]].strip().replace('\n\n',' [NEWLINE] ').replace('\n',' ') + '...',
						'identifier': identifier,
						'date': date,
						'origins': [s for s in latexml.get_local_overlap((textspan[0]+val[0],textspan[0]+val[1]),encompassingspans)]
					})

				#### Should we remove the value from the list of numbers now?
				#numbers.remove(val)
				#### Or is it OK that two keyword instances can point to the same number?


	#print('End:',text)

	return foundmention


#measurementre = re.compile(r'(?P<value>[0-9\.]+)(?:(?:(?:\s*\\pm\s*[0-9\.]+(?:\s*\(\s*[^\)]+\s*\))?)+)|(?:\s*(?:\\pm)?\s*[\^\_]\s*\{\s*[\-\+]?[0-9\.]+\s*\})|(?:\s*\\times\s*10\s*\^\s*\{\s*[\-\+]?[0-9\.]+\s*\}))+')
#measurementre = re.compile(r'(?P<value>[0-9\.]+)(?:(?:\s*\\pm\s*[0-9\.]+(?:\s*\(\s*[^\)]+\s*\))?)|(?:\s*(?:\\pm)?(?:\s*[\^\_]\s*\{\s*[\-\+]?[0-9\.]+\s*\}(?:\s*\(\s*[^\)]+\s*\))?){2})|(?:\s*\\times\s*10\s*\^\s*\{\s*[\-\+]?[0-9\.]+\s*\}))+')
#measurementre = re.compile(r'(?P<value>[0-9\.]+)(?:(?:\s*\\pm\s*[0-9\.]+(?:\s*\(\s*[^\)]+\s*\))?)|(?:\s*(?:\\pm)?(?:\s*[\^\_]\s*\{\s*[\-\+]?[0-9\.]+\s*\}(?:\s*\(\s*[^\)]+\s*\))?){2}))+')

#numberre = re.compile('(?<![\^_{] )(?<![\-0-9\.{])-?\d+(?:\.\d+)?(?:\ [0]+)*(?![\.\-}0-9])(?!\ \})')
measurementre = re.compile(r'''
		(?: ## This group will contain either a value (maybe with uncertainties), or a range
			(?P<value>
				(?<![\^_{]\ )
				(?<![0-9\.\-])
				-?
				\d+
				(?:\.\d+)?
				(?:\ [0]+)*
				(?![0-9\.\-])
				(?!\ \})
				#(?!\s)
			) # End value group
			(?P<details> # Begin details group
				(?:
					(?: # Begin pm regex
						\s*
						\\pm
						\s*
						[0-9\.]+
						(?: ## " _ { r }" or similar
							\s*
							\_
							\s*
							\{
							\s*
							[A-Za-z\.]+
							\s*
							\}
						)?
						(?:
							\s*
							\(
								\s*
								[^\)]+
								\s*
							\)
						)?
					) # End pm regex
				|
					(?: # Begin upper/lower pm regex
						\s*
						(?:\\pm)?
						(?:
							\s*
							[\^\_]
							\s*
							\{
								\s*
								[\-\+]?
								[0-9\.]+
								\s*
							\}
							(?:
								\s*
								\(
									\s*
									[^\)]+
									\s*
								\)
							)?
						){2}
					) # End upper/lower pm regex
				)* # Allow multiple detail blocks
			) # End details group
		|
			(?P<range>
				(?<![\^_{]\ )
				[0-9]+(?:\.[0-9]+)? # Number
				\s*
				(?:\u002D|\u2013|\u2014|\u2212) # Any hyphen character (need more?)
				\s*
				[0-9]+(?:\.[0-9]+)? # Number
			) # End range group
		)
		(?P<standardform>
			\s*
			\\times
			\s*
			10
			\s*
			\^
			\s*
			\{
				\s*
				[\-\+]?
				[0-9\.]+
				\s*
			\}
		)? # End standard form regex
		### This is where we would add a units regex
		(?P<units>
			(?:
				\s*
				(?:
					(?<![A-Za-z])[A-Za-z]{1,4}(?![A-Za-z])
				|
					[\{\}\[\]\~\/\^\_\%]
				|
					\\[A-Za-z]+
				|
					[\-\+]?[0-9\.]+
				)
			)+
		)?
		''',flags=re.VERBOSE)

def findInText_measurement(text, textspan, encompassingspans, search, keywords, identifier, date):

	#print('Start:',text)

	foundmention = False

	keywordmatches = [i for i in re.finditer(keywords, text)]

	if keywordmatches:

		## Why does this help?
		text = re.sub(r'\u002D|\u2013|\u2014|\u2212','-',text) # Replace alternate hyphens with single ascii hyphen

		all_measurements = [m for m in measurementre.finditer(text) if not any(is_overlapping(m.start(),m.end(),k.start(),k.end()) for k in keywordmatches)]
		## Has to be a list, because we need to reuse it for each keyword instance
		#numbers = [(m.start(0), m.end(0), m.group('value'), m.group('units')) for m in all_measurements]
		#print('Final:',numbers)

		measurements = [m for m in itertools.filterfalse(cullMatch,all_measurements)]

		for match in keywordmatches:
			min = sys.maxsize
			val = None
			span = None
			for num in measurements:
				dist = shortest_distance(match.start(0), match.end(0), num.start(0), num.end(0))
				if dist < min:
					min = dist
					val = num
			if val:
				foundmention = True

				span = get_text_span(match.start(0), match.end(0), val.start(0), val.end(0),offset=30)

				search.values[match.group()].append({
						'value': str_to_float(val.group('value')) if val.group('value') else '',
						'value_text': val.group('value') if val.group('value') else '',
						'range_text': val.group('range') if val.group('range') else '',
						'units': val.group('units').strip() if val.group('units') else '',
						'details': val.group('details').strip() if val.group('details') else '',
						'standardform': val.group('standardform').strip() if val.group('standardform') else '',
						'mention': '...' + text[span[0]:span[1]].strip().replace('\n\n',' [NEWLINE] ').replace('\n',' ') + '...',
						'match': val.group(0),
						'identifier': identifier,
						'date': date,
						'origins': [s for s in latexml.get_local_overlap((textspan[0]+val.start(0),textspan[0]+val.end(0)),encompassingspans)],
						'text': text.strip().replace('\n\n',' [NEWLINE] ').replace('\n',' '),
						'keywordtextspan': (match.start(0),match.end(0)),
						'valuetextspan': (val.start(0),val.end(0))
					})

				#if val.group('range'): print(identifier,val.group(0).encode('ascii',errors='ignore').decode())

				#### Should we remove the value from the list of numbers now?
				#numbers.remove(val)
				#### Or is it OK that two keyword instances can point to the same number?


	#print('End:',text)

	return foundmention


def processFiles(filelist,keywords,metadata,tag='p',verbose=False):

	if verbose: print('Processing',len(filelist),'files.')

	search = Search()
	for filepath in filelist:
		#print('Process:',filepath)
		findInFile(filepath, keywords, metadata, search=search, tag=tag)

	if verbose: print('Returning',len(filelist),'files with',sum((len(v) for k,v in search.values.items())),'results.')
	return search

def regularize_keyword(keyword):
	r = regularize_math(keyword)
	if keyword[0] == ' ':
		r = ' ' + r
	if keyword[-1] == ' ':
		r = r + ' '
	return r


def search_files(sources,keyword_strings,metadata,tag='p',chunksize=1000,processes=1,verbose=False):

	all_keyword_strings = list(itertools.chain.from_iterable(tuple(set([k,regularize_keyword(k)])) for k in keyword_strings.split(',')))
	keywords = '(?:' + '|'.join([re.escape(s) for s in all_keyword_strings]) + ')'

	if verbose: print(all_keyword_strings)

	if processes > 1:
		results = Search()

		with parallel.Pool(processes=processes) as p:
			for r in p.istarmap_unordered(processFiles,((i, keywords, metadata, tag, verbose) for i in iterator_slice(sources, chunksize))):
				if verbose: print('Results received: ' + str(sum((len(v) for k,v in r.values.items()))))
				results.append(r)
	else:
		files = list(sources)
		results = processFiles(files,keywords,metadata, tag, verbose=verbose)

	return results


class Search:
	def __init__(self):
		self.values = defaultdict(list)
		self.filecounter = 0
		self.mentionfilecounter = 0

	def append(self, other):
		for k,v in other.values.items():
			self.values[k].extend(v)
		self.filecounter += other.filecounter
		self.mentionfilecounter += other.mentionfilecounter

	def values_count(self):
		return sum((len(v) for k,v in self.values.items()))

if __name__ == "__main__":

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,IterFilesAction
	from gaml.utilities.jsonutils import dump_json
	from gaml.utilities import StopWatch

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Use keyword search to find values of provided keywords in a corpus of LateXML documents.")
	parser.add_argument("sources",action=IterFilesAction, recursive=True, suffix='.xml', help='Path to xml source(s).')
	parser.add_argument("keywords", help='Comma-separated list of keywords to search for.')
	parser.add_argument('metadata',action=oaipmh.MetadataAction,help='Path to metadata.')
	parser.add_argument('valuesfilepath',action=FileAction, mustexist=False, help='File in which to store found values.')
	parser.add_argument('-t','--tag',default='p',help='Tag of XML elements to search for values. Defaults to \'p\' (every block of text).')
	parser.add_argument('-s','--mentionsfilepath',action=FileAction, mustexist=False, help='File in which to store value mentions.')
	parser.add_argument('-m','--matchesfilepath',action=FileAction, mustexist=False, help='File in which to store value matches.')
	parser.add_argument('-i','--identifiersfilepath',action=FileAction, mustexist=False, help='File in which to store list of articles which values are drawn from.')
	parser.add_argument('-p','--processes',type=int,default=1,help='Number of processes to use. Default to one.')
	parser.add_argument('-c','--chunksize',type=int,default=1000,help='Size of chunks to send to processes. Default to 1000.')
	args = parser.parse_args()

	results = search_files(args.sources, args.keywords, args.metadata, tag=args.tag, chunksize=args.chunksize, processes=args.processes, verbose=True)

	## Print statistics
	print(str(results.filecounter) + ' files searched.')
	print('Mentions found in ' + str(results.mentionfilecounter) + ' files.')
	print(str(results.values_count()) + ' values found.')
	for key in results.values:
		print(str(key) + ': ' + str(len(results.values[key])))

	## Save results
	dump_json(dict(results.values),args.valuesfilepath,indent=4)

	if args.mentionsfilepath:
		with open(args.mentionsfilepath, "wb") as mentionsFile:
			for entrylist in results.values.values():
				for entry in entrylist:
					mentionsFile.write((f"{entry['value']:<10.10} {str(entry['details']).strip():35} | {str(entry['units']).strip():35} {entry['mention']}\n").encode('ascii',errors='ignore'))
					#mentionsFile.write((str(entry['value']) + ' ' + str(entry['units']) + ' ' + entry['mention'] + '\n').encode('ascii',errors='ignore'))

	if args.matchesfilepath:
		with open(args.matchesfilepath, "wb") as matchsFile:
			for entrylist in results.values.values():
				for entry in entrylist:
					matchsFile.write((f"{entry['match']}\n").encode('ascii',errors='ignore'))

	if args.identifiersfilepath:
		ids = set()
		for entrylist in results.values.values():
			for entry in entrylist:
				ids.add(entry['identifier'])
		with open(args.identifiersfilepath, 'w') as id_file:
			for identifier in ids:
				id_file.write(f'{identifier}\n')


	stopwatch.report()
