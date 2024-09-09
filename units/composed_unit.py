"""Composed units are quotients of products of other units
(but not other composed units.)
Utility methods here for working with abstract fractions."""

from units.abstract import AbstractUnit
from units.compatibility import compatible
import numbers
import cgi
from collections import Counter

#import units.dimensionless
from units.dimensionless import DimensionlessUnit

def unbox(numer, denom, multiplier):
	"""Attempts to convert the fractional unit represented by the parameters
	into another, simpler type. Returns the simpler unit or None if no
	simplification is possible."""

	if not denom and not numer:
		#return multiplier
		#return units.dimensionless.DimensionlessUnit(multiplier)
		return DimensionlessUnit(multiplier)

	if not denom and len(numer) == 1 and multiplier == 1:
		return numer[0]

	return None

def cancel(numer, denom):
	"""Cancel out compatible units in the given numerator and denominator.
	Return a triple of the new numerator, the new denominator, the implied
	quantity multiplier that has been squeezed out."""
	multiplier = 1

	simple_numer = numer[:]
	simple_denom = denom[:]

	for nleaf in numer:
		remaining_denom = simple_denom[:]
		for dleaf in remaining_denom:
			if compatible(nleaf, dleaf):
				multiplier *= nleaf.squeeze() / dleaf.squeeze()
				simple_numer.remove(nleaf)
				simple_denom.remove(dleaf)
				break

	return (simple_numer, simple_denom, multiplier)

def squeeze(numer, denom, multiplier):
	"""Simplify.

	Some units imply quantities. For example, a kilometre implies a quantity
	of a thousand metres. This 'squeezes' out these implied quantities,
	returning a modified multiplier and simpler units."""
	#print("Begin squeeze:",numer,"/",denom,"*",multiplier)

	result_numer = []
	result_denom = []
	result_mult = multiplier

	for unit in numer:
		while hasattr(unit, 'composed_unit'):
			#print("\t\tBreak down:",unit)
			unit = unit.composed_unit
		#print("\tNumer:",unit)

		if hasattr(unit, 'numer'):
			result_numer += unit.numer
			result_denom += unit.denom
			#result_mult *= unit.squeeze()
			result_mult *= unit.multiplier
		elif hasattr(unit, 'factor'):
			result_mult *= unit.squeeze()
		else:
			result_numer += [unit]

	for unit in denom:
		while hasattr(unit, 'composed_unit'):
			#print("\t\tBreak down:",unit)
			unit = unit.composed_unit
		#print("\tDenom:",unit)

		if hasattr(unit, 'numer'):
			result_denom += unit.numer
			result_numer += unit.denom
			#result_mult /= unit.squeeze()
			result_mult /= unit.multiplier
		elif hasattr(unit, 'factor'):
			result_mult /= unit.squeeze()
		else:
			result_denom += [unit]

	result_numer.sort()
	result_denom.sort()
	#print("Initial squeeze:",result_numer,"/",result_denom,"*",result_mult)

	(simpler_numer, simpler_denom, cancellation_mult) = cancel(result_numer,result_denom)
	#print("Cancelled result:",simpler_numer,"/",simpler_denom,"*",cancellation_mult)

	result_mult *= cancellation_mult
	#print("Final result:",simpler_numer,"/",simpler_denom,"*",result_mult)

	return (simpler_numer, simpler_denom, result_mult)

def cleanup(numer, denom, multiplier):
	"""Cleanup possible confusions in numer and denom."""

	result_numer = []
	result_denom = []
	result_mult = multiplier

	for unit in numer:
		if hasattr(unit, 'numer'):
			result_numer += unit.numer
			result_denom += unit.denom
		elif hasattr(unit, 'factor'):
			result_mult *= unit.squeeze()
		else:
			result_numer += [unit]

	for unit in denom:
		if hasattr(unit, 'numer'):
			result_denom += unit.numer
			result_numer += unit.denom
		elif hasattr(unit, 'factor'):
			result_mult *= unit.squeeze()
		else:
			result_denom += [unit]

	simple_numer = result_numer[:]
	simple_denom = result_denom[:]

	for nleaf in result_numer:
		remaining_denom = simple_denom[:]
		for dleaf in remaining_denom:
			if nleaf == dleaf:
				simple_numer.remove(nleaf)
				simple_denom.remove(dleaf)
				break

	return (simple_numer, simple_denom, result_mult)

def simplify(unit):

	while hasattr(unit, 'composed_unit'):
		unit = unit.composed_unit

	if not hasattr(unit,'numer'):
		return unit

	(squeezed_numer, squeezed_denom, squeezed_multiplier) = squeeze(unit.numer,unit.denom,unit.multiplier)

	unboxed = unbox(squeezed_numer, squeezed_denom, squeezed_multiplier)
	if unboxed:
		return unboxed
	else:
		return ComposedUnit(squeezed_numer,squeezed_denom,squeezed_multiplier)

class ComposedUnit(AbstractUnit):
	"""A ComposedUnit is a quotient of products of units."""

	def __new__(cls, numer, denom, multiplier=1):
		"""Construct a unit that is a quotient of products of units,
		including an implicit quantity multiplier."""
		#print("New composed unit:",numer,"/",denom)

		# Move inverted units in numer/denom to other side
		# Cancel identical units (i.e. km and km, but not km and m)
		(cleaned_numer, cleaned_denom, cleaned_mult) = cleanup(numer,denom,multiplier)

		# If the result is a factor or single unit, return that
		unboxed = unbox(cleaned_numer, cleaned_denom, cleaned_mult)
		if unboxed:
			#print("Returning unboxed unit from new composed unit:",unboxed.__repr__())
			return unboxed

		# Find the simplest representation of this unit
		(squeezed_numer, squeezed_denom, squeezed_multiplier) = squeeze(numer,denom,multiplier)
		#print("Squeezed:",squeezed_numer,"/",squeezed_denom,"*",squeezed_multiplier)
		# If that simplest representation is a single unit or factor, return that
		unboxed = unbox(squeezed_numer, squeezed_denom, squeezed_multiplier)
		if unboxed:
			#print("Returning squeezed unboxed unit from new composed unit:",unboxed.__repr__())
			return unboxed

		#print("Returning new composed unit")
		return super(ComposedUnit, cls).__new__(cls)

	def __init__(self, numer, denom, multiplier=1):
		#print("Init composed unit")
		super(ComposedUnit, self).__init__(is_si=False)
		self.numer,self.denom,self.multiplier = cleanup(numer,denom,multiplier)
		self.squeezed_numer, self.squeezed_denom, self.squeezed_multiplier = squeeze(self.numer,self.denom,self.multiplier)

	def __hash__(self):
		return hash((tuple(self.numer),tuple(self.denom),self.multiplier))

	def __reduce__(self):
		return (
			ComposedUnit,
			(self.numer, self.denom, self.multiplier)
		)

	def str_includes_multiplier(self):
		return self.multiplier == 1

	def str_multiplier(self):
		return self.multiplier

	#def __str__(self):
	#	 if self.denom and self.numer:
	#		 return (' * '.join([str(x) for x in self.numer]) +
	#				 " / " +
	#				 ' * '.join([str(x) for x in self.denom]))
	#	 elif self.denom:
	#		 return '1 / ' + ' * '.join([str(x) for x in self.denom])
	#	 else:
	#		 return ' * '.join([str(x) for x in self.numer])
	#def __str__(self):
	#	 if self.multiplier != 1:
	#		 return '{:.4E} '.format(self.multiplier) + ' '.join([str(x) for x in self.numer] + [str(x)+'^-1' for x in self.denom])
	#	 else:
	#		 return ' '.join([str(x) for x in self.numer] + [str(x)+'^-1' for x in self.denom])
	#def __str__(self):
	#	return ' '.join([str(x) for x in self.numer] + [str(x)+'^-1' for x in self.denom])

	def __str__(self):
		numer_items = [
				k + "^" + str(v) if v>1 else k
				for k,v in Counter([str(x) for x in self.numer]).items()]
		denom_items = [
				k + "^-" + str(v)
				for k,v in Counter([str(x) for x in self.denom]).items()]
		return ' '.join(numer_items + denom_items)

	def latex(self):
		numer_items = [
				k + "$^{" + str(v) + "}$" if v>1 else k
				for k,v in Counter([str(x) for x in self.numer]).items()]
		denom_items = [
				k + "$^{-" + str(v) + "}$"
				for k,v in Counter([str(x) for x in self.denom]).items()]
		return ' '.join(numer_items + denom_items)

	def html(self):
		numer_items = [
				cgi.escape(k) + "<sup>" + str(v) + "</sup>" if v>1 else k
				for k,v in Counter([str(x) for x in self.numer]).items()]
		denom_items = [
				cgi.escape(k) + "<sup>-" + str(v) + "</sup>"
				for k,v in Counter([str(x) for x in self.denom]).items()]
		return ' '.join(numer_items + denom_items)

	def __repr__(self):
		return ("ComposedUnit(" +
				", ".join([repr(x) for x in [self.numer,
											self.denom,
											self.multiplier]]) +
				")")

	def canonical(self):
		"""Return the simplest representation of this unit,
		dropping any multiplier."""
		if self.squeezed_denom or len(self.squeezed_numer) != 1:
			return ComposedUnit(self.squeezed_numer,self.squeezed_denom,multiplier=1)
		else:
			return self.squeezed_numer[0].canonical()

	def normal(self):
		""" Return unit sans multipler. """
		if self.multiplier == 1:
			return self
		else:
			return ComposedUnit(self.numer, self.denom, multiplier=1)

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		if self.squeezed_denom or len(self.squeezed_numer) != 1:
			return (tuple(i.immutable() for i in self.squeezed_numer),
					tuple(i.immutable() for i in self.squeezed_denom))
		else:
			return self.squeezed_numer[0].immutable()

	def squeeze(self):
		"""Return this unit's implicit quantity multiplier."""
		#return self.multiplier
		return self.squeezed_multiplier

	def __mul__(self, other):
		if isinstance(other,numbers.Number):
			return ComposedUnit(self.numer,
								self.denom,
								self.multiplier * other)
		elif hasattr(other, "numer"):
			assert hasattr(other, "denom")
			return ComposedUnit(self.numer + other.numer,
								self.denom + other.denom,
								self.multiplier * other.multiplier)
		elif isinstance(other, AbstractUnit):
			return ComposedUnit(self.numer + [other],
								self.denom,
								self.multiplier)
		else:
			return NotImplemented

	def invert(self):
		"""Return (this unit)^-1."""
		return ComposedUnit(self.denom, self.numer, 1 / self.multiplier)

	def __truediv__(self, other):
		if isinstance(other,numbers.Number):
			return self * (1 / other)
		elif isinstance(other, AbstractUnit):
			return self * other.invert()
		else:
			return NotImplemented

	def __rmul__(self,other):
		return self * other

	def __rtruediv__(self,other):
		return other * self.invert()

	# Backwards-compatibility for <= Python 2.7
	__div__ = __truediv__
	__rdiv__ = __rtruediv__

	def __pow__(self, exponent):
		if exponent >= 0:
			return ComposedUnit(self.numer * exponent,
								self.denom * exponent,
								self.multiplier ** exponent)
		else:
			return self.invert() ** -exponent
