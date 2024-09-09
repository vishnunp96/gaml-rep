import re,os
from lark.exceptions import UnexpectedCharacters,UnexpectedToken,VisitError
from lark import Lark,Transformer,Tree,Discard
from gaml.preprocessing.mathtokenise import regularize_math
from gaml.utilities.filehandler import readFile

import gaml.parsing.symbols

__grammar_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'symbolgrammar.lark')

def get_lark_grammar():
	grammar = readFile(__grammar_file)
	return grammar

def check_sign(sign):
	if sign == '+': return '+'
	else: return '-'

latex_expressions_ignore = set([
		'\\mbox',
		'\\hbox',
		'\\rm',
		'\\cal',
		'\\mathcal',
		'\\scriptstyle',
		'\\scriptscriptstyle'
	])

latex_functions_ignore = set([
		'\\mbox',
		'\\hbox',
		'\\rm',
		'\\cal',
		'\\mathnormal',
		'\\mathrm',
		'\\mathit',
		'\\mathbf',
		'\\mathsf',
		'\\mathtt',
		'\\mathfrak',
		'\\mathcal',
		'\\mathbb',
		'\\mathscr',
		'\\text',
		'\\textnormal',
		'\\textrm',
		'\\textsf',
		'\\texttt',
		'\\textup',
		'\\textit',
		'\\textsl',
		'\\textsc',
		'\\textbf',
		'\\textmd',
		'\\textlf',
		'\\emph',
		'\\scriptstyle',
		'\\scriptscriptstyle'
	])

class SymbolTransformer(Transformer):

	def character_expression(self, children):
		symbol = children[0]
		if symbol.type == 'LATEX' and symbol in latex_expressions_ignore:
			raise Discard
		else:
			return gaml.parsing.symbols.LeafSymbol(str(symbol))
	def number_expression(self, children):
		return gaml.parsing.symbols.LeafNumber(str(children[0]))

	def latex_function(self, children):
		if children[0] in latex_functions_ignore:
			return children[1]
		else:
			return gaml.parsing.symbols.LatexFunction(str(children[0]), children[1])

	def function_expression(self, children):
		return gaml.parsing.symbols.FunctionSymbol(children[0], children[1])

	def raised_expression(self, children):
		return children[0].superscript(children[1])
	def lowered_expression(self, children):
		return children[0].subscript(children[1])

	def subtraction_expression(self, children):
		if len(children)==1:
			return gaml.parsing.symbols.NegativeSymbol(children[0])
		else:
			return gaml.parsing.symbols.BinaryOperatorExpression('-',children[0],children[1])
	def addition_expression(self, children):
		if len(children)==1:
			return Tree('positive_expression',children)
		else:
			return gaml.parsing.symbols.BinaryOperatorExpression('+',children[0],children[1])
	def direct_fraction_expression(self, children):
		return gaml.parsing.symbols.BinaryOperatorExpression('/',children[0],children[1])
	def latex_frac_expression(self, children):
		return gaml.parsing.symbols.BinaryOperatorExpression('/',children[0],children[1])

	def equality_expression(self, children):
		return gaml.parsing.symbols.EqualityExpression(children[0], children[1])

	def sequential_expressions(self, children):
		return gaml.parsing.symbols.SymbolSequence([i for i in children])

	def comma_separated_expression(self, children):
		return gaml.parsing.symbols.SeparatedExpressions(',', [i for i in children])

	def roundbracketed_expression(self, children):
		return gaml.parsing.symbols.BracketedExpression('()', children[0])
	def squarebracketed_expression(self, children):
		return gaml.parsing.symbols.BracketedExpression('[]', children[0])
	def anglebracketed_expression(self, children):
		return gaml.parsing.symbols.BracketedExpression(('\\langle','\\rangle'), children[0])
	def piped_expression(self, children):
		return gaml.parsing.symbols.BracketedExpression('||', children[0])

	def symbol(self, children):
		if len(children) == 0:
			return None
		elif isinstance(children[0], gaml.parsing.symbols.BracketedExpression) and children[0].left == '(':
			return children[0].value
		else:
			return children[0]

#symbolparser = Lark(get_lark_grammar(), parser="lalr", lexer="contextual", start='symbol', transformer=SymbolTransformer())
symbolparser = Lark(get_lark_grammar(), parser="lalr", lexer="contextual", start='symbol')
symboltransformer = SymbolTransformer()

def regularize_symbol(unit_text):
	text = regularize_math(unit_text)
	text = re.sub(r'\s+',' ',text).strip()
	return text

def parse_symbol(symbol_text):
	try:
		text = regularize_symbol(symbol_text)
		#return symbolparser.parse(text)
		tree = symbolparser.parse(text)
		return symboltransformer.transform(tree)
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

def balance_brackets2(text):
	## Brace styles: 1:"()", 2:"{}", 3:"[]"
	brackets = ''
	for c in text:
		if c in '({[':
			brackets += c
		elif c==')':
			if brackets[-1] == '(':
				brackets = brackets[:-1]
			else:
				pass ## Solve?
		elif c=='}':
			pass
		elif c==']':
			pass

if __name__ == "__main__":

	from gaml.utilities.terminalutils import printacross

	rawsymbolparser = Lark(get_lark_grammar(), parser="lalr", lexer="contextual", start='symbol')

	cases = [
			'V',
			'V _ { c }',
			'T _ { eff }',
			'T _ { e f f }',
			'T _ e f f',
			'V _ c',
			'T _ eff',
			'T _ { \\mathrm { eff } }',
			'T _ { \\text { eff } }',
			'\\langle z \\rangle',
			'\\Omega _ { m }',
			'\\sigma _ { 8 }',
			'\\Omega _ { \\Lambda }',
			'M _ { \\star }',
			'\\mu',
			'\\log n ( Li )',
			'\\log g',
			'Fe/H',
			'[ Fe / H ]',
			'\langle { [ Fe / H ] } \\rangle',
			'[ \\mathrm { Fe } / \\mathrm { H } ]',
			'R / R _ { 25 }',
			'\\bar { z }',
			'\\dot { M }',
			'( V-K )',
			'( V+K )',
			'( \\Omega _ { b } + \\Omega _ { dm } )',
			'\\Omega _ { M } + \\Omega _ { \\Lambda }',
			'\\Omega _ { m } h ^ 2 - \\Omega _ { \\Lambda } \\Omega _ { b }',
			'E ( B - V )',
			'( B - V ) E',
			'- \\log P',
			'N ( WC ) / N ( WN )',
			'\\langle M _ { HOST } ( tot ) \\rangle',
			'E ( f ( x ) h )',
			'F _ { x } /F _ { opt }',
			'{ L } _ { B }',
			'M _ { * }',
			'f ( R )',
			'\\Omega _ { m } h ^ 2',
			'kT',
			'R _ {',
			'f (',
			'\\mbox',
			'\\dot',
			'\\langle',
			'H / m _ { \\scriptscriptstyle { \\mathrm { Pl } } }',
			'H ( z = 2.36 )',
			'h ^ { 3 / 2 } \\Omega _ { \\mathrm { b } } / \\Omega _ { \\mathrm { m } }',
			'\\mbox { H$ { } _ { \\circ } $ }',
			'N ( { H } _ { 2 } )',
			'\\Delta \\dot { \\nu } / \\dot { \\nu }',
			'f _ { { } _ { NL } }',
			'M ( { H _ { 2 } } )',
			'log N ( He ) / N ( H )',
			'\\mbox { $ T _ { \\mathrm { eff } }',
			'n _ { { H _ { 2 } } }',
			'{ d } N / { d } E',
			'v _ { rot } \sin { i }',
			'\chi _ { min } ^ { 2 }',
			'\chi ^ { 2 } _ { min }',
			'\Delta T _ { eff }',
			'| \Delta T _ { eff } |',
			'| T |',
			'| T | | B |',
			'k _ { \\| }',
			'w _ { 0 } , w _ { a }',
			'\\Omega _ { m , 0 }',
			'M \\sim i~ {}',
			'M _ { \\nu , { sterile } } ^ { eff }',
			'f ( R , T )',
			'f ( R, T )',
			'H _ { \\lower 2.0 pt \\hbox { $ \\scriptstyle 0 $ } }',
			'H _ { 0 } \\left ( \\cos i / 0.7 \\right ) ^ { 1 / 2 }',
			'( ^ { 7 } Li/H ) _ { P }',
			'M ( r < 2.1 h ^ { -1 } Kpc )',
			'x _ { { } _ { J } }',
			'f _ { \\hbox { \sevenrm v 0 } }',
			'\\it { kT }',
			'dn _ { s } / d \\ln k',
			'< N _ { \\mathrm { { \\mathrm { C \\mathsc { iii } } } } } / N _ { \\mathrm { { \\mathrm { C \\mathsc { iv } } % } } } >',
			'\\log N _ { \\mathrm { { \\mathrm { C \\mathsc { iv } } } } }',
			'\\log \\left ( IRX \\right )',
			'< B - V >',
			'\\Omega h ^ { 0.88 }',
			'\\Omega h ^ { 1.3 }',
			'{ \\cal M } _ { A }',
			'P ^ { S } ( k _ { \\parallel } , \\mathbf { k } _ { \\perp } )',
			'd \\log \\xi / dM _ { R } ^ { 0 }',
			'\\Phi ^ { SNO } _ { th } ( { { } ^ { 8 } } { B }',
			'\\sigma _ { * } ^ { ( e ) } ( 0.3 \\la z \\la 1 )',
			'\\beta ( L = L ^ { * } ,z = 0 )',
			'dY / d O',
			'\\chi ^ { 2 } / d . o . f .',
			'\\Omega _ { M 0 }',
			'\\sigma _ { 8 } ( \\Omega _ { m } / 0.3 ) ^ { 0.65 }',
			'\\langle \\mbox { $ L ^ { \\prime } _ { \\mbox { \\tiny { CO } } } $ } \\rangle',
			'E _ { p } ^ { { } ^ { \\prime } }',
			'dq / dz',
			'\\Omega ^ { 0.6 } / b _ { \\scriptscriptstyle I }',
			'\\rho _ { c } / \phi ( > 0.2 L _ { \\ast } )',
			'\mbox { H$ { } _ { \\circ } $ }',
			'\\frac { \\Omega _ { 0 } ^ { 0.6 } } { b ^ { 0.75 } }',
			'\\Delta ( R-H ) / \\Delta ( log r )',
			'\Omega _ { 0 } +0.2 \\lambda _ { 0 }',
			'H _ { 0 } \\left ( \\cos i / 0.7 \\right ) ^ { 1 / 2 }',
			'( ^ { 7   Li/H ) _ { P }',
			'M ( r < 2.1 h ^ { -1 } Kpc )',
			'x _ { { } _ { J } }',
			'L _ { MWNSC, 4.5 \\mu { m } }',
			'\\frac { \\Omega _ m } { \\Lambda _ 0 }',
			'H \\left ( z \\right )'
		]

	for text in cases:
		printacross('=',80)
		print(f'Text: {text}')
		text = regularize_symbol(text)
		print(f'Regularised text: {text}')

		try:
			tree = rawsymbolparser.parse(text)
			print(tree)
			print(tree.pretty())
			try:
				transformed = SymbolTransformer().transform(tree)
				print(repr(transformed))
				print(transformed)
			except (UnexpectedCharacters,UnexpectedToken) as e:
				print('Error:',e)
		except (UnexpectedCharacters,UnexpectedToken) as e:
			print('Error:',e)
		print(parse_symbol(text))
		print()

	test_eq = [
			('\chi _ { min } ^ { 2 }','\chi ^ { 2 } _ { min }'),
			('T _ { eff }','T _ eff'),
			('T _ { eff }','T _ e f f'),
			('T _ { \\mathrm { eff } }','T _ { eff }'),
			('[ Fe / H ]','[ \\mathrm { Fe } / \\mathrm { H } ]'),
			('( V+K )','V + K'),
			('E ( f ( x ) h )','E ( h f ( x ) )'),
			('H ( z = 2.36 )','H ( z = 2.360 )'),
			('h ^ { 3 / 2 }','h ^ { 1.5 }')
		]

	printacross('+',80)
	printacross('+',80)
	for s1,s2 in test_eq:
		print(f'"{s1}" == "{s2}" -> {parse_symbol(s1) == parse_symbol(s2)}')
		print()
