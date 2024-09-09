import re
from lark import Lark,Transformer,Tree
from gaml.preprocessing.mathtokenise import regularize_math
from gaml.utilities.filehandler import readFile
from functools import reduce

from gaml.units import unit
from gaml.units.predefined import define_units
define_units()

from gaml.units.measurement import Measurement,Uncertainty

parser = Lark(readFile('measurementgrammar.lark'), parser="lalr", lexer="contextual", start='measurement')
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
		return {c.data: c.children for c in children if c is not None}

class MeasurementTransformer(Transformer):

	def unit(self, children):
		return unit(str(children[0]))
	def inverseunit(self,children):
		return unit(str(children[0])).invert()
	def raisedunit(self, children):
		exponent = int(children[1].children)
		return unit(str(children[0])) ** exponent
	def units(self, children):
		return Tree('units',reduce(lambda x, y: x*y, children))

	def number(self, children):
		return float(children[0])
	def upper(self, children):
		return Tree('upper',children[0])
	def lower(self, children):
		return Tree('lower',children[0])

	def bracketed(self, children):
		return Tree('bracketed',re.sub(r'[\(\)]','',str(children[0])).strip())
	def centralvalue(self,children):
		return Tree('centralvalue',children[0])
	def junk(self, children):
		return None

	def uncertainty(self, children):
		return Tree('uncertainty',Uncertainty(value=children[0]))
	def named_uncertainty(self, children):
		return Tree('uncertainty',Uncertainty(value=children[0],note=children[1].children))
	def upperlower_uncertainty(self, children):
		values = {c.data: c.children for c in children}
		return Tree('uncertainty',Uncertainty(upper=values['upper'],lower=values['lower']))
	def named_upperlower_uncertainty(self, children):
		values = {c.data: c.children for c in children}
		return Tree('uncertainty',Uncertainty(upper=values['upper'],lower=values['lower'],note=values['bracketed']))

	def measurement(self, children):
		values = {c.data: c.children for c in children if c is not None}
		return Measurement(
				value=values['centralvalue'],
				unit=values.get('units'),
				uncertainties=[i.children for i in children if i is not None and i.data == 'uncertainty'],
				note=values.get('bracketed'))

if __name__ == "__main__":

	lines = [
			"43 \\pm 4 _ { -1 } ^ { +2 } km s Mpc leading to",
			"43 \\pm 2",
			"70.8 ^ { +2.1 } _ { -2.0 } \\mathrm { km / s / Mpc } from this",
			"70.8 ^ { +2.1 } _ { -2.0 } \\mathrm { km s Mpc } ( statistical )",
			"76.9 \\pm ^ { 3.9 } _ { 3.4 } \\pm ^ { 10.0 } _ { 8.0 } km s ^ { -1 } Mpc ^ { -1 }",
			"79 \\pm 4 ( statistical ) \\pm 2 ( systematic )",
			"76 \pm 1.3 ( 1- \sigma statistical ) \pm 6 ( systematic ) km s ^ { -1 } Mpc ^ { -1 }",
			"66 \pm 8 { km } ~ { } { s ^ { -1 } } ~ { } { Mpc ^ { -1 } } ( 1 \sigma error )",
			"10 %",
			"42 and potentially ( { 32 } % ) other text",
			"72 \pm 8 km s ^ { -1 } Mpc ^ { -1 } and 72 \pm 5 km s ^ { -1 } Mpc ^ { -1 }",
			"63 km per sec per Mpc .",
			"45 km/s/Mpc .",
			"69.62 \pm 0.59 ( { { km } } / { { s } } ) { { Mpc } } ^ { -1 }",
			"15000 km s ^ { -1 }",
			"79 \\pm ^ { +1 } _ { -2 } ( statistical ) ^ { +3 } _ { -4 } \\pm 2 ( systematic )",
		]

	for text in lines:
		print(text)

		text = regularize_math(text)
		text = re.sub(r'[\{\}\~]','',text)
		text = re.sub(r'\\mathrm|\\textrm|\\mbox|\\hbox','',text)
		text = re.sub(r'\s+',' ',text)

		print(text)

		tree = parser.parse(text)

		print(tree.pretty())

		print(tree)

		for i in tree.find_data("upperlower_uncertainty"):
			print()
			print(i)
			for c in i.children:
				print('\t',c)

		print()
		print(MeasurementTransformer().transform(tree))
		print()
