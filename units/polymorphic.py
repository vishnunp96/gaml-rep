"""An abstract base class to define the interface for all units."""

import itertools
import numbers

from units.abstract import MetaUnit,AbstractUnit
from units.dimensionless import DimensionlessUnit
from units.composed_unit import ComposedUnit,squeeze,unbox
from units.registry import REGISTRY

from units.named_composed_unit import NamedComposedUnit

def polymorphic_unit(name, expressions, is_si=False):
	return NamedPolymorphicUnit(name, PolymorphicUnit(expressions), is_si)

def unpack_polymorphic(unit):
	"""Convert an arbitrary unit, which may contain polymorphic units
	into a single PolymorphicUnit object."""
	return unit

def simplify(unit):

	while isinstance(unit, (NamedComposedUnit,NamedPolymorphicUnit)):
		unit = unit.composed_unit if isinstance(unit, NamedComposedUnit) else unit.polymorphic_unit
	## Unit is now either ComposedUnit or PolymorphicUnit

	if isinstance():
		return unit

	(squeezed_numer, squeezed_denom, squeezed_multiplier) = squeeze(unit.numer,unit.denom,unit.multiplier)

	unboxed = unbox(squeezed_numer, squeezed_denom, squeezed_multiplier)
	if unboxed:
		return unboxed
	else:
		return ComposedUnit(squeezed_numer,squeezed_denom,squeezed_multiplier)

def poly_compatible(unit1, unit2):
	unit1 = unpack_polymorphic(unit1)
	unit2 = unpack_polymorphic(unit2)




class PolymorphicUnit(MetaUnit):
	"""Class which allows for units with multiple expressions.
	This class will act as its first expression unless called upon
	by a polymorphic-aware function"""

	def get_expressions(self):
		"""Expressions of the polymorphic unit"""
		return self._expressions
	expressions = property(get_expressions)

	def __init__(self, expressions):
		#print("Init polymorphic unit")
		super(PolymorphicUnit, self).__init__(is_si=False)
		## assert len(set(expressions)) == len(expressions)? Shouldn't really have any duplicates here...
		self._expressions = expressions # List of additional unit expressions

	def __hash__(self):
		return hash(tuple(self.expressions))

	def __reduce__(self):
		return (PolymorphicUnit, (self.name, self.expressions, self.is_si()))

	def canonical(self):
		"""Return the simplest representation of this unit,
		dropping any multiplier."""
		return self.expressions[0].canonical()

	def polymorphic_canonical(self):
		"""Return the simplest representation of this unit,
		dropping any multiplier, keeping the units polymorphism."""
		return PolymorphicUnit([u.canonical() for u in self.expressions])

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		return self.expressions[0].immutable()

	def invert(self):
		"""Return (this unit)^-1."""
		return PolymorphicUnit([u.invert() for u in self.expressions])

	def squeeze(self):
		"""Return this unit's implicit quantity multiplier."""
		return self.expressions[0].squeeze()

	def str_includes_multiplier(self):
		"""Whether the string name of the unit already encapsulates
		the unit's multiplier."""
		return self.expressions[0].str_includes_multiplier()

	def __repr__(self):
		return 'PolymorphicUnit([' + ', '.join([repr(u) for u in self.expressions]) + '])'

	def __str__(self):
		return '[' + ', '.join([str(u) for u in self.expressions]) + ']'

	def __mul__(self, other):
		if isinstance(other,PolymorphicUnit):
			## Should try and clear up repeats in here? - but need to keep first element in place
			return PolymorphicUnit([i*j for i,j in itertools.product(self.expressions,other.expressions)])
		else:
			return PolymorphicUnit([u*other for u in self.expressions])

	def __truediv__(self, other):
		if isinstance(other,PolymorphicUnit):
			if set(self.expressions) == set(other.expressions):
				return DimensionlessUnit(1)
			else:
				## Should try and clear up repeats in here? - but need to keep first element in place
				return PolymorphicUnit([i/j for i,j in itertools.product(self.expressions,other.expressions)])
		else:
			return PolymorphicUnit([u/other for u in self.expressions])

	def __rmul__(self, other):
		# Do not need to account for other PolymorphicUnits here, as __mul__ will be called in such cases.
		return PolymorphicUnit([other*u for u in self.expressions])

	def __rtruediv__(self, other):
		# Do not need to account for other PolymorphicUnits here, as __truediv__ will be called in such cases.
		return PolymorphicUnit([other/u for u in self.expressions])

	# Backwards-compatibility for <= Python 2.7
	__div__ = __truediv__
	__rdiv__ = __rtruediv__

	def __pow__(self, exponent):
		return PolymorphicUnit([u**exponent for u in self.expressions])


class NamedPolymorphicUnit(MetaUnit):
	"""Class which allows for named units with multiple expressions.
	This class will act as its first expression unless called upon
	by a polymorphic-aware function"""

	def get_name(self):
		"""The label for the polymorphic unit"""
		return self._name
	name = property(get_name)

	def get_polymorphic_unit(self):
		"""The labeled polymorphic unit"""
		return self._polymorphic_unit
	polymorphic_unit = property(get_polymorphic_unit)
	composed_unit = property(get_polymorphic_unit)

	def __new__(cls, name, polymorphic_unit, is_si=False):
		"""Give a polymorphic unit a new symbol."""
		# pylint: disable=W0613

		if name not in REGISTRY:
			REGISTRY[name] = super(NamedPolymorphicUnit,cls).__new__(cls)
		return REGISTRY[name]

	def __init__(self, name, polymorphic_unit, is_si=False):
		#print("Init named polymorphic unit")
		super(NamedPolymorphicUnit, self).__init__(is_si)
		self._name = name
		self._polymorphic_unit = polymorphic_unit

	def __hash__(self):
		return hash((self.name,self.polymorphic_unit))

	def __reduce__(self):
		return (
			NamedPolymorphicUnit,
			(self._name, self._polymorphic_unit, self.is_si())
		)

	def canonical(self):
		"""Return the simplest representation of this unit,
		dropping any multiplier."""
		return self.polymorphic_unit.canonical()

	def polymorphic_canonical(self):
		"""Return the simplest representation of this unit,
		dropping any multiplier, keeping the units polymorphism."""
		return self.polymorphic_unit.polymorphic_canonical()

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		return self.polymorphic_unit.immutable()

	def invert(self):
		"""Return (this unit)^-1."""
		return ComposedUnit([],[self],1)

	def squeeze(self):
		"""Return this unit's implicit quantity multiplier."""
		return self.polymorphic_unit.squeeze()

	def str_includes_multiplier(self):
		"""Whether the string name of the unit already encapsulates
		the unit's multiplier."""
		return True

	def __repr__(self):
		return ("NamedPolymorphicUnit(" +
				", ".join([repr(x) for x in [self.name,
											self.polymorphic_unit,
											self.is_si()]])+
				")")

	__str__ = get_name

	def __mul__(self, other):
		if isinstance(other,numbers.Number):
			#return self.composed_unit * other
			return ComposedUnit([self],[],other)
		elif isinstance(other, AbstractUnit):
			return ComposedUnit([self, other], [])
		else:
			return NotImplemented

	def __truediv__(self, other):
		if isinstance(other,numbers.Number):
			return ComposedUnit([self],[],1/other)
		elif isinstance(other, AbstractUnit):
			return ComposedUnit([self], [other])
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
		if exponent >=0:
			return ComposedUnit([self]*exponent,[],1)
		else:
			return ComposedUnit([],[self]*-exponent,1)
