import itertools

class MetaSymbol(object):
	"""Parent class/interface for symbols."""

	def contains(self, item):
		""" Return true if this symbol contains item as a subtree anywhere. """
		raise NotImplementedError

	def subscript(self, sub):
		"""Return this symbol with a subscript."""
		return LoweredRaisedSymbol(self, lowered=sub)
	def superscript(self, sup):
		"""Return this symbol with a superscript."""
		return LoweredRaisedSymbol(self, raised=sup)

	def terminals(self):
		""" Return list of all leaf nodes (str/float) contained in this symbol. """
		raise NotImplementedError

class AbstractSymbol(MetaSymbol):
	pass

class LeafSymbol(AbstractSymbol):
	def __init__(self, symbol):
		self.symbol = symbol

	def __str__(self):
		return self.symbol
	def __repr__(self):
		return f'LeafSymbol({repr(self.symbol)})'

	def __eq__(self, other):
		if isinstance(other, LeafSymbol):
			return self.symbol == other.symbol
		else:
			return NotImplemented

	def contains(self, item):
		return self == item
	__contains__ = contains

	def __hash__(self):
		return hash((self.symbol,))

	def __or__(self, other):
		if self == other:
			return self
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return [self.symbol]

class LeafNumber(AbstractSymbol):
	def __init__(self, string):
		self.string = string
		self.number = float(self.string)

	def __str__(self):
		return self.string
	def __repr__(self):
		return f'LeafNumber({repr(self.string)})'

	def __eq__(self, other):
		if isinstance(other, LeafNumber):
			return self.number == other.number
		else:
			return NotImplemented

	def contains(self, item):
		return self == item
	__contains__ = contains

	def __hash__(self):
		return hash((self.number,))

	def __or__(self, other):
		if self == other:
			return self
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return [self.number]

class LatexFunction(AbstractSymbol):
	def __init__(self, name, operand):
		self.name = name
		self.operand = operand

	def __str__(self):
		return f'{self.name} {{ {str(self.operand)} }}'
	def __repr__(self):
		return f'LatexFunction(' + ', '.join([self.name, repr(self.operand)]) + ')'

	def __eq__(self, other):
		if isinstance(other, LatexFunction):
			return self.name == other.name and self.operand == other.operand
		else:
			return NotImplemented

	def contains(self, item):
		return self == item or self.name.contains(item) or self.operand.contains(item)
	__contains__ = contains

	def __hash__(self):
		return hash((self.name,self.operand))

	def __or__(self, other):
		if isinstance(other, LatexFunction):
			if self.name == other.name:
				return LatexFunction(self.name, self.operand|other.operand)
			elif self.operand == other.operand:
				return LatexFunction(self.name|other.name, self.operand)
		if isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		return NotImplemented

	def terminals(self):
		return list(set(self.name.terminals() + self.operand.terminals()))

class FunctionSymbol(AbstractSymbol):
	def __init__(self, name, operand):
		self.name = name
		self.operand = operand

	def __str__(self):
		return str(self.name) + ' ( ' + str(self.operand) + ' )'
	def __repr__(self):
		return f'FunctionSymbol(' + ', '.join(repr(i) for i in [self.name, self.operand]) + ')'

	def __eq__(self, other):
		if isinstance(other, FunctionSymbol):
			return self.name == other.name and self.operand == other.operand
		else:
			return NotImplemented

	def contains(self, item):
		return self == item or self.name.contains(item) or self.operand.contains(item)
	__contains__ = contains

	def __hash__(self):
		return hash((self.name,self.operand))

	def __or__(self, other):
		if isinstance(other, FunctionSymbol):
			if self.name == other.name:
				return FunctionSymbol(self.name, self.operand|other.operand)
			elif self.operand == other.operand:
				return FunctionSymbol(self.name|other.name, self.operand)
		if isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		return NotImplemented

	def terminals(self):
		return list(set(self.name.terminals() + self.operand.terminals()))

class NegativeSymbol(AbstractSymbol):
	def __new__(cls, base):
		### This might be a little ridiculous...?
		if isinstance(base,LeafNumber):
			if base.number<0:
				string = base.string[1:]
			else:
				string = '-'+base.string
			return LeafNumber(string)
		else:
			return super(NegativeSymbol, cls).__new__(cls)

	def __init__(self, base):
		self.base = base

	def __str__(self):
		return '-' + str(self.base)
	def __repr__(self):
		return 'NegativeSymbol(' + repr(self.base) + ')'

	def __eq__(self, other):
		if isinstance(other, NegativeSymbol):
			return self.base == other.base
		else:
			return NotImplemented

	def contains(self, item):
		return self == item or self.base.contains(item)
	__contains__ = contains

	def __hash__(self):
		return hash(('-',self.base))

	def __or__(self, other):
		if isinstance(other, NegativeSymbol):
			return NegativeSymbol(self.base|other.base)
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return self.base.terminals()

class LoweredRaisedSymbol(AbstractSymbol):
	def __init__(self, base, raised=None, lowered=None):
		self.base = base
		self.raised = raised if raised is not None else SymbolSequence()
		self.lowered = lowered if lowered is not None else SymbolSequence()

	def __str__(self):
		result = str(self.base)
		if not isinstance(self.base, (LeafSymbol, LeafNumber, NegativeSymbol, LatexFunction)):
			result = '{ ' + result + ' }'
		if self.lowered:
			result += ' _ { ' + str(self.lowered) + ' }'
		if self.raised:
			result += ' ^ { ' + str(self.raised) + ' }'
		return result
	def __repr__(self):
		return 'LoweredRaisedSymbol(' + ', '.join(repr(i) for i in [self.base,self.lowered,self.raised]) + ')'

	def subscript(self, sub):
		if not self.lowered:
			return LoweredRaisedSymbol(self.base, raised=self.raised, lowered=sub)
		elif isinstance(self.lowered, SymbolSequence):
			return LoweredRaisedSymbol(self.base, raised=self.raised, lowered=self.lowered+sub)
		else:
			return LoweredRaisedSymbol(self.base, raised=self.raised, lowered=SymbolSequence([self.lowered,sub]))

	def superscript(self, sup):
		if not self.lowered:
			return LoweredRaisedSymbol(self.base, raised=sup, lowered=self.lowered)
		elif isinstance(self.lowered, SymbolSequence):
			return LoweredRaisedSymbol(self.base, raised=self.raised+sup, lowered=self.lowered)
		else:
			return LoweredRaisedSymbol(self.base, raised=SymbolSequence([self.raised,sup]), lowered=self.lowered)

	def __eq__(self, other):
		if isinstance(other, LoweredRaisedSymbol):
			return self.base == other.base and self.lowered == other.lowered and self.raised == other.raised
		else:
			return NotImplemented

	def contains(self, item):
		return self == item or self.base.contains(item) or self.raised.contains(item) or self.lowered.contains(item)
	__contains__ = contains

	def __hash__(self):
		return hash((self.base,self.lowered,self.raised))

	def __or__(self, other):
		if isinstance(other, LoweredRaisedSymbol):
			## TODO Not finished
			return LoweredRaisedSymbol(self.base|other.base, raised=self.raised|other.raised, lowered=self.lowered|other.lowered)
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return list(set(self.base.terminals() + self.lowered.terminals() + self.raised.terminals()))

class SymbolSequence(AbstractSymbol):
	def __new__(cls, sequence=None):
		### This might be a little ridiculous...?
		if sequence is not None and len(sequence) == 1:
			return sequence[0]
		else:
			return super(SymbolSequence, cls).__new__(cls)

	def __init__(self, sequence=None):
		self.sequence = tuple(sequence) if sequence is not None else tuple()

	def __len__(self):
		return len(self.sequence)

	def __str__(self):
		return ' '.join(str(i) for i in self.sequence)
	def __repr__(self):
		return 'SymbolSequence(' + ', '.join(repr(i) for i in self.sequence) + ')'

	def __add__(self, other):
		if isinstance(other,SymbolSequence):
			return SymbolSequence(self.sequence+other.sequence)
		elif isinstance(other,AbstractSymbol):
			return SymbolSequence(self.sequence+(other,))
		else:
			return NotImplemented

	def __eq__(self, other):
		if isinstance(other, SymbolSequence):
			return set(self.sequence) == set(other.sequence) ## What about repeated elements?
		else:
			return NotImplemented

	def contains(self, item):
		if isinstance(item, SymbolSequence):
			self_seq = set(self.sequence)
			item_seq = set(item.sequence)
			if self_seq == item_seq or self_seq.issuperset(item_seq): ## What about repeated elements?
				return True
		return item == self or any(i.contains(item) for i in self.sequence)
	__contains__ = contains

	def __hash__(self):
		return hash(self.sequence)

	def __or__(self, other):
		if self == other:
			## What if there is some overlap? Use set to find overlap and then make remaining into Multichoice?
			return self
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return list(set(itertools.chain.from_iterable(i.terminals() for i in self.sequence)))

class SeparatedExpressions(AbstractSymbol):
	def __init__(self, separator, sequence):
		self.separator = ' ' + separator.strip() + ' '
		self.sequence = tuple(sequence) if sequence else tuple()

	def __len__(self):
		return len(self.sequence)

	def __str__(self):
		return self.separator.join(str(i) for i in self.sequence)
	def __repr__(self):
		return 'SeparatedExpressions(' + ', '.join(repr(i) for i in [self.separator,self.sequence]) + ')'

	def __eq__(self, other):
		if isinstance(other, SeparatedExpressions):
			## It probably isn't that sensible to compare the separators...?
			return self.separator == other.separator and set(self.sequence) == set(other.sequence)
		else:
			return NotImplemented

	def contains(self, item):
		if isinstance(item, SymbolSequence):
			self_seq = set(self.sequence)
			item_seq = set(item.sequence)
			if self.separator == item.separator and (self_seq == item_seq or self_seq.issuperset(item_seq)): ## What about repeated elements?
				return True
		return item == self or any(i.contains(item) for i in self.sequence)
	__contains__ = contains

	def __hash__(self):
		## Again, probably shouldn't use the separator here?
		return hash((self.separator,self.sequence))

	def __or__(self, other):
		if self == other:
			return self
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return list(set(itertools.chain.from_iterable(i.terminals() for i in self.sequence)))

class BinaryOperatorExpression(AbstractSymbol):
	def __init__(self, operator, lhs, rhs):
		self.operator = operator
		self.lhs = lhs
		self.rhs = rhs

	def __str__(self):
		return str(self.lhs) + ' ' + self.operator + ' ' + str(self.rhs)
	def __repr__(self):
		return 'BinaryOperatorExpression(' + ', '.join(repr(i) for i in [self.operator,self.lhs,self.rhs]) + ')'

	def __eq__(self, other):
		if isinstance(other, BinaryOperatorExpression):
			return self.operator == other.operator and self.lhs == other.lhs and self.rhs == other.rhs
		else:
			return NotImplemented

	def contains(self, item):
		return self == item or self.lhs.contains(item) or self.rhs.contains(item)
	__contains__ = contains

	def __hash__(self):
		return hash((self.operator,self.lhs,self.rhs))

	def __or__(self, other):
		if isinstance(other, BinaryOperatorExpression) and self.operator == other.operator:
			return BinaryOperatorExpression(self.operator, self.lhs|other.lhs, self.rhs|other.rhs)
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return list(set(self.lhs.terminals() + self.rhs.terminals()))

class EqualityExpression(AbstractSymbol):
	def __init__(self, lhs, rhs):
		self.lhs = lhs
		self.rhs = rhs

	def __str__(self):
		return str(self.lhs) + ' = ' + str(self.rhs)
	def __repr__(self):
		return 'EqualityExpression(' + ', '.join(repr(i) for i in [self.lhs,self.rhs]) + ')'

	def __eq__(self, other):
		if isinstance(other, EqualityExpression):
			return self.lhs == other.lhs and self.rhs == other.rhs
		else:
			return NotImplemented

	def contains(self, item):
		return self == item or self.lhs.contains(item) or self.rhs.contains(item)
	__contains__ = contains

	def __hash__(self):
		return hash((self.lhs,self.rhs))

	def __or__(self, other):
		if isinstance(other, EqualityExpression):
			return EqualityExpression(self.lhs|other.lhs, self.rhs|other.rhs)
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return list(set(self.lhs.terminals() + self.rhs.terminals()))

class BracketedExpression(AbstractSymbol):
	def __init__(self, brackets, value):
		self.left,self.right = brackets
		self.value = value

	def __str__(self):
		return self.left + ' ' + str(self.value) + ' ' + self.right
	def __repr__(self):
		return 'BracketedExpression([' + ', '.join(repr(i) for i in [self.left,self.right]) + '], ' + repr(self.value) + ')'

	def __eq__(self, other):
		if isinstance(other, BracketedExpression):
			return self.left == other.left and self.right == other.right and self.value == other.value
		else:
			return NotImplemented

	def contains(self, item):
		return self == item or self.value.contains(item)
	__contains__ = contains

	def __hash__(self):
		return hash((self.left,self.right,self.value))

	def __or__(self, other):
		if isinstance(other, BracketedExpression) and self.left == other.left and self.right == other.right:
			return BracketedExpression((self.left,self.right),self.value|other.value)
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol([self,other])
		else:
			return NotImplemented

	def terminals(self):
		return self.value.terminals()

class MultichoiceSymbol(AbstractSymbol):
	def __init__(self, options):
		unpacked = []
		for s in options:
			if isinstance(s,MultichoiceSymbol):
				unpacked.extend(s.options)
			else:
				unpacked.append(s)
		self.options = set(unpacked)

	def __len__(self):
		return len(self.options)

	def __str__(self):
		return '% ' + ' | '.join(str(i) for i in self.options) + ' %'
	def __repr__(self):
		return 'MultichoiceSymbol(' + repr(self.options) + ')'

	def __eq__(self, other):
		if isinstance(other, MultichoiceSymbol):
			return self.options == other.options
		elif isinstance(other, AbstractSymbol):
			return any(other == option for option in self.options)
		else:
			return NotImplemented

	def contains(self, item):
		if isinstance(item, MultichoiceSymbol):
			self_opt = set(self.options)
			item_opt = set(item.options)
			if self_opt == item_opt or self_opt.issuperset(item_opt): ## What about repeated elements?
				return True
		return item == self or any(i.contains(item) for i in self.options)
	__contains__ = contains

	def __hash__(self):
		return hash(('options',tuple(self.options)))

	def __or__(self, other):
		if isinstance(other, MultichoiceSymbol):
			return MultichoiceSymbol(self.options | other.options)
		elif isinstance(other, AbstractSymbol):
			return MultichoiceSymbol(self.options.union([other]))
		else:
			return NotImplemented

import functools

def combine_options(options):
	return functools.reduce(lambda x,y: x|y, options)

#from collections import defaultdict

#def combine_options(options):
#	types = defaultdict(list)
#	for option in options: types[type(option)].append(option)
#class LeafSymbol(AbstractSymbol):
#class LeafNumber(AbstractSymbol):
#class LatexFunction(AbstractSymbol):
#class FunctionSymbol(AbstractSymbol):
#class NegativeSymbol(AbstractSymbol):
#class LoweredRaisedSymbol(AbstractSymbol):
#class SymbolSequence(AbstractSymbol):
#class SeparatedExpressions(AbstractSymbol):
#class BinaryOperatorExpression(AbstractSymbol):
#class EqualityExpression(AbstractSymbol):
#class BracketedExpression(AbstractSymbol):
#class MultichoiceSymbol(AbstractSymbol):

