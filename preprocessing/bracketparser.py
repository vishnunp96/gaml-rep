from lark import Lark,Transformer,Tree
from gaml.utilities.filehandler import readFile
import os,re

__grammar_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'bracketgrammar.lark')

def get_lark_grammar():
	grammar = readFile(__grammar_file)
	return grammar

parser = Lark(get_lark_grammar(), parser="lalr", lexer="contextual", start='text')

def normalise_tex_comments(tex):
	return re.sub(r'(?<!\\)\%[^\n]*\n',' ',tex)

class BracketedTransformer(Transformer):

	def dollared(self, children):
		print(f'Dollared: {children}')
		return '$' + ' '.join([str(c) for c in children]) + '$'
		return Tree('dollared',children)
	def function(self, children):
		print(f'Function: {children}')
		return children[0] + ' '.join([str(c) for c in children[1:]])
		return Tree('function',children)
	def thing(self, children):
		print(f'Thing: {children}')
		return ' '.join([str(c) for c in children])
		return Tree('thing',children)
	def bracketed(self, children):
		print(f'Bracketed: {children}')
		return '{' + ' '.join([str(c) for c in children]) + '}'
		return Tree('bracketed',children)
	def text(self, children):
		print(f'Text: {children}')
		return ' '.join([str(c) for c in children])
		return Tree('text',children)

if __name__ == "__main__":

	s1 = 'H _ { \lower 2.0 pt \hbox { $ \scriptstyle 0 $ } } = 61.7 ^ { +1.2 } _ { -1.1 } \hbox { km } \hbox% { sec } ^ { -1 } \hbox { Mpc } ^ { -1 }'

	s2 = 'H _ { \lower 2.0 pt \hbox { $ \scriptstyle 0 $ } }'

	s3 = 'H_{\\lower 2.0pt\\hbox{$\\scriptstyle 0$}}=61.7^{+1.2}_{-1.1}\\,\\hbox{km}\\;\\,\\hbox%\n{sec}^{-1}\\,\\hbox{Mpc}^{-1}'

	s3norm = normalise_tex_comments(s3)

	print(s3norm)

	tree = parser.parse(s3norm)
	transformed = BracketedTransformer().transform(tree)

	print(transformed,end='\n\n')
	print(tree.pretty())
