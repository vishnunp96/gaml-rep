import re,os
import itertools
from lark.exceptions import UnexpectedCharacters,UnexpectedToken,VisitError
from lark import Lark,Transformer,Tree
from gaml.preprocessing.mathtokenise import regularize_math
from gaml.utilities.filehandler import readFile
from functools import reduce

from gaml.units import unit, named_unit, scaled_unit, alias_unit
from gaml.units.leaf_unit import LeafUnit
from gaml.units.registry import REGISTRY
from gaml.units.measurement import Measurement,Uncertainty
from gaml.units.dimensionless import DimensionlessUnit

def regularize_unit(unit_text):
	text = regularize_math(unit_text)
	text = re.sub(r'[\{\}\~]','',text)
	text = re.sub(r'\\mathrm|\\textrm|\\mbox|\\hbox|\\cdot','',text)
	text = re.sub(r'\s+',' ',text).strip()
	return text

def define_parsing_units():
	# meter, gram, second, ampere, kelvin, mole, candela
	for sym in ["m", "g", "s", "A", "K", "mol", "cd"]:
		LeafUnit(sym, is_si=True)
	scaled_unit('tonne', 'kg', 1000) # == 1Mg.

	# More complex SI units
	for sym in ["rad", "sr"]: # Radians and steradians
		LeafUnit(sym, is_si=True)
	named_unit("Hz", [], ["s"]) #hertz
	named_unit("N", ["m", "kg"], ["s", "s"]) # Newton
	named_unit("Pa", ["N"], ["m", "m"]) # pascal
	named_unit("J", ["N", "m"], []) # Joule # Dangerous, 3J is a complex number
	named_unit("W", ["J"], ["s"]) # Watt
	named_unit("C", ["s", "A"], []) # Coulomb
	named_unit("V", ["W"], ["A"]) # Volt
	named_unit("F", ["C"], ["V"]) # Farad
	named_unit("Ohm", ["V"], ["A"])
	named_unit("S", ["A"], ["V"])	# Siemens
	named_unit("Wb", ["V", "s"], []) # Weber
	named_unit("T", ["Wb"], ["m", "m"]) # Tesla
	named_unit("H", ["Wb"], ["A"]) # Henry
	named_unit("lm", ["cd", "sr"], []) # lumen
	named_unit("lx", ["lm"], ["m", "m"]) # lux
	named_unit("Bq", [], ["s"]) # Becquerel
	named_unit("Gy", ["J"], ["kg"]) # Gray
	named_unit("Sv", ["J"], ["kg"]) # Sievert
	named_unit("kat", ["mol"], ["s"]) # Katal

	# Define time units
	alias_unit('sec','s',is_si=False)
	scaled_unit('min', 's', 60.)
	scaled_unit('hr', 'min', 60.)
	alias_unit('hour','hr',is_si=False)
	alias_unit('hours','hr',is_si=False) # Just to be safe
	scaled_unit('day', 'hr', 24.)
	alias_unit('days','day',is_si=False) # Just to be safe
	alias_unit('d','day',is_si=False)
	scaled_unit('wk', 'day', 7.)
	scaled_unit('yr', 'day', 365.25, is_si=True)
	alias_unit('year','yr',is_si=False)

	# Define astro units
	scaled_unit('ly', 'm', 9460730472580800) # light-year
	scaled_unit('AU', 'm', 149597870691) # Astronomical unit
	scaled_unit('pc', 'm', 3.08568025 * 10 ** 16, is_si=True) # parsec
	named_unit("Jy", ['W'], ['m','m','Hz'], 1e-26) # Jansky
	LeafUnit('h',is_si=False) ## Little h (Hubble parameter)
	scaled_unit('erg', 'J', 1e-7, is_si=False)
	alias_unit('ergs','erg',is_si=False)
	LeafUnit('mag', is_si=False) # Magnitude
	LeafUnit('dex', is_si=False) # dex (Should/can this be a named Dimensionless quantity, or something?)

	# Define physics units
	scaled_unit('eV', 'J', 1.6021766209e-19, is_si=True) # Electron-Volt

	# Solar units
	REGISTRY[regularize_unit('M_\odot')] = scaled_unit('M_\odot','g', 1.9885e30) # Solar Mass
	REGISTRY[regularize_unit('M_\sun')] = REGISTRY[regularize_unit('M_\odot')] # Another subscript for Solar Mass
	REGISTRY[regularize_unit('M_\Sun')] = REGISTRY[regularize_unit('M_\odot')] # Another subscript for Solar Mass
	REGISTRY[regularize_unit('L_\odot')] = scaled_unit('L_\odot','W', 3.828e26) # Solar Luminosity
	REGISTRY[regularize_unit('L_\sun')] = REGISTRY[regularize_unit('L_\odot')] # Another subscript for Solar Luminosity
	REGISTRY[regularize_unit('L_\Sun')] = REGISTRY[regularize_unit('L_\odot')] # Another subscript for Solar Luminosity
	REGISTRY[regularize_unit('R_\odot')] = scaled_unit('R_\odot','m', 6.957e8) # Solar Radius
	REGISTRY[regularize_unit('R_\sun')] = REGISTRY[regularize_unit('R_\odot')] # Another subscript for Solar Radius
	REGISTRY[regularize_unit('R_\Sun')] = REGISTRY[regularize_unit('R_\odot')] # Another subscript for Solar Radius

	named_unit('c',['m'],['s'],multiplier=299792458,is_si=False) # Speed of Light # Is this one a bad idea? Does it conflict?

define_parsing_units()

__grammar_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'advancedmeasurementgrammarfromsymbol.lark')

def get_lark_grammar():
	grammar = readFile(__grammar_file)
	return grammar

textparser = Lark(get_lark_grammar(), parser="lalr", lexer="contextual", start='measurement')
unitparser = Lark(get_lark_grammar(), parser="lalr", lexer="contextual", start='units')

def check_sign(sign):
	if sign == '+': return '+'
	else: return '-'

class MeasurementTransformer(Transformer):

	def unit(self, children):
		if len(children)==1:
			return unit(str(children[0]))
		else:
			exponent = int(children[1].children) ## TODO: What about fractional exponents?
			return unit(str(children[0])) ** exponent
	def inverseunit(self,children):
		return children[0].invert()
	def units(self, children):
		return Tree('units',reduce(lambda x, y: x*y, children))

	def number(self, children):
		#return float(re.sub(r'\s','',children[0]))
		return float(''.join([c if c.type!='SIGN' else check_sign(c) for c in children]))
	def upper(self, children):
		return Tree('upper',children[0])
	def lower(self, children):
		return Tree('lower',children[0])
	def magnitude(self, children):
		return Tree('magnitude',children[-1].children)

	def bracketed(self, children):
		return Tree('bracketed',re.sub(r'[\(\)]','',str(children[0])).strip())
	def undercomment(self, children):
		return Tree('uncercomment',re.sub(r'[\{\}\_]','',str(children[0])).strip())
	def centralvalue(self,children):
		return Tree('centralvalue',children[0])
	def junk(self, children):
		return None

	def value(self, children):
		if len(children) == 1:
			if children[0].data == 'value': # Case: ( value )
				return self.value(children[0].children)
			elif children[0].data == 'magnitude': # Case: magnitude
				return Tree('value',[Tree('centralvalue',1), children[0]])
			else:
				return Tree('value', [children[0]])
		else:
			return Tree('value',list(itertools.chain.from_iterable([c.children if c.data=='value' else [c] for c in children])))

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
		parts = {c.data: c.children for c in children if c is not None}
		values = {c.data: c.children for c in parts['value']}

		units = parts.get('units', DimensionlessUnit(1))
		units = units*10**values['magnitude'] if 'magnitude' in values else units

		if 'centralvalue' in values:
			return Measurement(
					value = values['centralvalue'],
					unit = units,
					#uncertainties = [i for k,i in values.items() if i is not None and k == 'uncertainty'],
					uncertainties = [c.children for c in parts['value'] if c is not None and c.data == 'uncertainty'],
					note = parts.get('bracketed'))
		elif 'range' in values:
			lower,upper = values['range']
			if lower > upper:
				#raise ValueError("Incorrectly formatted range: "+str(values['range']))
				lower,upper = upper,lower
			central = (lower+upper)/2
			return Measurement(
					value = central,
					unit = units,
					uncertainties = [Uncertainty(upper=upper-central,lower=lower-central)],
					note = parts.get('bracketed'))

def parse_measurement(text, return_errors=False):
	text = balance_brackets(text)
	text = regularize_math(text)
	text = re.sub(r'^\s*=\s*','',text)
	text = re.sub(r'[\{\}\~]','',text)
	text = re.sub(r'(?:^|\s)\.(?:$|\s)',' ',text)
	text = re.sub(r'\\mathrm|\\textrm|\\mbox|\\hbox|\\cdot|\\text','',text)
	text = re.sub(r'\s+',' ',text).strip()
	try:
		tree = textparser.parse(text)
		return MeasurementTransformer().transform(tree)
	except (VisitError,UnexpectedCharacters,UnexpectedToken,ValueError) as e:
		if return_errors:
			return str(e).split('\n')[0]
		else:
			return None

def parse_unit(unit_text):
	text = regularize_unit(unit_text)
	try:
		tree = unitparser.parse(text)
		return MeasurementTransformer().transform(tree).children
	except (VisitError,UnexpectedCharacters,UnexpectedToken) as e:
		return None

def balance_brackets(text):
	## Brace styles: 1:"()", 2:"{}", 3:"[]"
	brackets = {1:0, 2:0, 3:0}
	for c in text:
		if c=='(':
			brackets[1] += 1
		elif c==')':
			brackets[1] -= 1
		elif c=='{':
			brackets[2] += 1
		elif c=='}':
			brackets[2] -= 1
		elif c=='[':
			brackets[3] += 1
		elif c==']':
			brackets[3] -= 1
	if brackets[1]!=0:
		text = '('*abs(min(0,brackets[1])) + text + ')'*max(0,brackets[1])
	if brackets[2]!=0:
		text = '{'*abs(min(0,brackets[2])) + text + '}'*max(0,brackets[2])
	if brackets[3]!=0:
		text = '['*abs(min(0,brackets[3])) + text + ']'*max(0,brackets[3])
	return text

if __name__ == "__main__":

	from gaml.utilities.terminalutils import printacross

	cases = [
			'10',
			'-5.5',
			'- 5.5',
			'10.5',
			'5.5',
			'4-7',
			'4 - 7',
			'110 M _ { \\odot }',
			'45 m s^{-1}',
			'45 \\pm 4 m',
			'45 ^ +4 _ -5 m/s',
			'45 _ 5 ^ 4 km/s',
			'45 - 67 m/s',
			'10 (comment)',
			'10 \\pm 4 (comment)',
			'10 _ 4 ^ 5 (stat) ^ 3 _ 2 (rand) km (with comment)',
			'10 ^ 4 _ 5 (stat) ^ 3 _ 2 (rand) km',
			'10 ^ 4 _ 5 ^ 3 _ 2 km',
			'1.8 \\times 10 ^ { 8 } M _ { \\odot }',
			'(5.6 - 6.2) \\times 10 ^ { 3 } g',
			'( 1.8 \\pm 0.2 ) \\times 10 ^ { 4 } m',
			'10 ^ { 7 } kg',
			'( 10 kg )',
			'10 h',
			'10 h M_\\odot',
			'10 h M _ { \\odot }',
			'7.5 \\times 10 ^ { 14 } h ^ { -1 } { M _ { \\hbox { $ \\odot$ } } }',
			'0.102 \\pm 0.010 \\mbox { $ { M } _ { \\odot } $ } \mbox { $ { pc } $ } ^ { -3 }',
			'( 0.060 \\pm 0.003 ) \\hbox { $ h$ } ^ { -3 / 2 }',
			'52 ^ { +5 } _ { -4 } \\mathrm { km s ^ { -1 } Mpc ^ { -1 } }',
			'68 \\pm 2 ( { random } ) \\pm 5 ( { systematic } ) km s ^ { -1 } Mpc ^ { -1 }',
			'( 69 \\pm 9 ) km s ^ { -1 } Mpc ^ { -1 }',
			'=58.5 \\pm 6.3 km s ^ { -1 } Mpc ^ { -1 }',
			'52 ^ { +14 } _ { -8 } ~ { } { km } ~ { } { s ^ { -1 } } ~ { } { Mpc ^ { -1 } }',
			'( 0.50 \\pm 0.04 ) \\Omega _ { 0 } ^ { -0.47 + 0.10 \\Omega _ { 0 } }',
			'( 0.58 \\pm 0.06 ) \\times \\Omega _ { 0 } ^ { -0.47 + 0.16 \\Omega _ { 0 } }',
			'0.68 + 1.16 ( 0.95 - n ) \\pm 0.04',
			'0.495 ^ { +0.034 } _ { -0.037 } ) \\Omega _ { M } ^ { -0.60 }',
			'0.57 \\pm 0.04 \\right ) \\Omega _ { M } ^ { \\left ( 0.24 \\mp 0.18 \\right ) % \\Omega _ { M } -0.49 }',
			'0.76 \\pm 0.01 + 0.50 ( 1 - \\alpha _ { \\mathrm { M } } )',
			'0.67 ^ { +0.18 } _ { -0.13 } respectively',
			'0.40 \\pm 0.09 h _ { 50 } ^ { -0.5 }',
			'0.3 ,~ { }',
			'0.31 ^ { +0.27 } _ { -0.14 } ( 68 % ) ^ { +0.12 } _ { -0.10 }',
			'0.27 h _ { \\mathrm { 70 } } ^ { -1 }',
			'0.30 _ { -0.05 } ^ { +0.09 } b _ { S } ^ { 5',
			'0.27 \\pm 0.02 ~ { } ( \\mathrm { statistical } )',
			'0.2629 _ { -0.0153 } ^ { +0.0155 } ( 1 \\sigma ) _ { -0.0223 } ^ { +0.0236 }',
			'0.24 \\pm 0.02 \\times b _ { M / L',
			'6.9 ( \\pm 1.6 ) \\times 10 ^ { 19 } cm ^ { -2 }',
			'1.091 \\pm 0.046 R _ { \\mathrm { Jup } } )',
			'4.8 ^ { + \\infty } _ { -2 }',
			'7 ^ { +4 } _ { -6 } per cent',
			'7 ^ { +4 } _ { -6 }',
			'8.1 \\times 10 ^ { -7 } per cent',
			'+1.00 \\pm 0.15 min per day',
			'0.2 - 0.3 dex dispersion'
		]

	rawparser = Lark(get_lark_grammar(), parser="lalr", lexer="contextual", start='measurement', debug=True)

	for text in cases:
		printacross('=',80)
		print(f'Text: {text}')

		text = regularize_math(text)
		#text = re.sub(r'[\{\}\~]','',text)
		#text = re.sub(r'(?:^|\s)\.(?:$|\s)',' ',text)
		#text = re.sub(r'\\mathrm|\\textrm|\\mbox|\\hbox|\\cdot|\\text','',text)
		#text = re.sub(r'\s+',' ',text).strip()
		text = re.sub(r'per\s*cent','%',text).strip()
		print(f'Regularised text: {text}')

		try:
			tree = rawparser.parse(text)
			print(tree)
			print(tree.pretty())
			#try:
			#	transformed = MeasurementTransformer().transform(tree)
			#	print(repr(transformed))
			#	print(transformed)
			#except (VisitError,UnexpectedCharacters,UnexpectedToken) as e:
			#	print('Error:',e)
		except (UnexpectedCharacters,UnexpectedToken) as e:
			print('Error:',e)
		#print(parse_measurement(text))
		print()

