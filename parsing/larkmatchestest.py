import re
from lark.exceptions import UnexpectedCharacters,UnexpectedToken
from lark import Lark,Transformer,Tree
from preprocessing.mathtokenise import regularize_math
from utilities.filehandler import readFile
from functools import reduce

from units import unit,named_unit
from units.predefined import define_units
from units.compatibility import compatible
define_units()
named_unit('sec',['s'],[])

hubble_unit = unit('km') / unit('s') / unit('Mpc')

textparser = Lark(readFile('measurementgrammar.ebnf'), parser="lalr", lexer="contextual", start='measurement')
#parser = Lark(readFile('measurementgrammar.ebnf'), start='measurement')

class MyTransformer(Transformer):
	def __init__(self,*args,**kwargs):
		self.count = 0
		super(MyTransformer,self).__init__(*args,**kwargs)
	def unit(self, children):
		return unit(str(children[0]))
	def inverseunit(self,children):
		return unit(str(children[0])).invert()
	def raisedunit(self, children):
		exponent = int(children[1].children[0])
		if exponent < 0:
			return unit(str(children[0])).invert() ** -exponent
		else:
			return unit(str(children[0])) ** exponent
	def units(self, children):
		return Tree('units',reduce(lambda x, y: x*y, children))
	def number(self, children):
		return float(children[0])
	def bracketed(self, children):
		return Tree('bracketed',str(children[0]))
	def centralvalue(self,children):
		return Tree('centralvalue',children[0])
	def junk(self, children):
		return None
	def measurement(self, children):
		result = {c.data: c.children for c in children if c is not None}
		if 'units' in result and compatible(hubble_unit, result['units']):
			result['modified_value'] = hubble_unit(result['units'](result['centralvalue'])).num
			result['units'] = str(result['units'])
		return result


if __name__ == "__main__":

	from utilities.argparseactions import ArgumentParser,FileAction
	from utilities import StopWatch

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Use keyword search to find values of provided keywords in a corpus of LateXML documents.")
	parser.add_argument('matches',action=FileAction, mustexist=False, help='Matches text file.')
	parser.add_argument('output',action=FileAction, mustexist=False, help='Output filepath.')
	args = parser.parse_args()

	with open(args.matches,'r') as matches, open(args.output,'w') as output:
		for match in matches:
			text = regularize_math(match)
			text = re.sub(r'[\{\}\~]','',text)
			text = re.sub(r'\\mathrm|\\textrm|\\mbox|\\hbox|\\cdot','',text)
			try:
				tree = textparser.parse(text)
				result = MyTransformer().transform(tree)
			except (UnexpectedCharacters,UnexpectedToken) as e:
				result = str(e).split('\n')[0]
			output.write(f"{match.strip():50}\t{result}\n")

