import re
from utilities.numberutils import str_to_float

datere = re.compile('\([^\)]+\d{4}[^\)]+\)') #('\((?:[^\)0-9]+\d+[^\)0-9]+)+\)') #\(\s*[^\)]*\s*[A-Z][A-Za-z]+\s*\d+\s*[^,\)\d]*\s*\) ## Requires spaces inside brackets, but oh well
datednamere = re.compile('\d{4}\s+[A-Z][a-z]+') # e.g. '2013 Planck'
identifierre = re.compile('[A-Z]+\s*\d+') ## Matches any number preceeded by an uppercase string (with or without string)
arxivre = re.compile('[a-zA-Z]\/[0-9]{7,}|[a-zA-Z][0-9]{7,}') # Attempts to match arXiv ids - which will have an alphabetical sequence optionally followed by a '/', and then 7 numbers (or possible more)
mathcharre = re.compile('\\\\mathchar\s*[0-9]+') ## Is that \ correct?

def cullMatch(match):
	''' Return True if the argument matches any of the filters. '''

	#print('In cullMatch')
	filters = [dateFilter, datedNameFilter, identifierFilter, arXivFilter, mathcharFilter]

	answer = any((f(match) for f in filters))

	#print('Returning answer')
	return answer

def dateFilter(match):

	#print('In dateFilter')
	#print(match.string)

	for m in datere.finditer(match.string):
		#print('Comparing:',match,m)
		#print(m.group(0))
		if overlap(match,m):
			#print('Overlap found.')
			num = re.match('[0-9]{4}',match.group(0).strip())
			if num:
				return 1400 < str_to_float(num.group(0)) < 2100
			#return 1400 < str_to_float(match.group(0)) < 2100
		#print('No overlap.')
	return False

def datedNameFilter(match):

	#print('In datedNameFilter')

	for m in datednamere.finditer(match.string):
		if overlap(match,m):
			num = re.match('[0-9]{4}',match.group(0).strip())
			if num:
				return 1400 < str_to_float(num.group(0)) < 2100
			#return 1400 < str_to_float(match.group(0)) < 2100
	return False

def identifierFilter(match):

	#print('In identifierFilter')

	for m in identifierre.finditer(match.string):
		if overlap(match,m):
			return True
	return False

def arXivFilter(match):

	#print('In arXivFilter')

	for m in arxivre.finditer(match.string):
		if overlap(match,m):
			return True
	return False

def mathcharFilter(match):

	for m in mathcharre.finditer(match.string):
		if overlap(match,m):
			return True
	return False


def overlap(m1,m2):
	if m1.start() < m2.start():
		return m1.end() > m2.start()
	else:
		return m2.end() > m1.start()
