import nltk
from pprint import pprint

sentence = '43 \\pm 4 ^ { +1 } _ { -2 }'

pattern = [
		(r'[\+\-]?[0-9\.]+','NUM'),
		(r'\\pm','PM'),
		(r'[\{\}]','LBR'),
		(r'\^','UP'),
		(r'_','LO'),
	]
tagger = nltk.RegexpTagger(pattern)
output = tagger.tag(sentence.split())

pprint(output)

grammar = nltk.CFG.fromstring("""
S -> NUM DETAIL
DETAIL -> UNCERTAINTY UNITS
DETAIL -> UNCERTAINTY
DETAIL -> UNITS
UNCERTAINTY -> UNCERTAINTY UNCERTAINTY
UNCERTAINTY -> PM NUM
UNCERTAINTY -> UP NUM
UNCERTAINTY -> LO NUM
NUM -> LBR NUM LBR
""")

parser = nltk.parse.RecursiveDescentParser(grammar)

for t in parser.parse(output):
	print(t)
