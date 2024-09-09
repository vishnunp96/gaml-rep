import re

def whitespace_escape(text,repitiions='+'):
	#### Is this useful?
	'''
	Escape whitespace characters to accept given repititions of whitespace.
	'''
	return re.sub('\s+','\s'+repitiions,text)
