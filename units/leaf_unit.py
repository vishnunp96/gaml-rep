"""Leaf units stand alone.
They are not compatible with any other kind of unit."""

from gaml.units.abstract import AbstractUnit
from gaml.units.registry import REGISTRY
from gaml.units.composed_unit import ComposedUnit

import numbers
import cgi

class LeafUnit(AbstractUnit):
	"""Leaf units are not compatible with other units, but they can be
	composed to make other units."""

	def get_specifier(self):
		"""Return the symbol of the unit."""
		return self._specifier
	specifier = property(get_specifier)
	name = property(get_specifier)

	def __new__(cls, specifier, is_si):
		# pylint: disable=W0613
		if specifier not in REGISTRY:
			REGISTRY[specifier] = super(LeafUnit, cls).__new__(cls)
		return REGISTRY[specifier]

	def __init__(self, specifier, is_si):
		"""Make a new LeafUnit with the given unit specifier and
		SI-compatibility. A unit that is SI compatible can be prefixed,
		e.g. with k to mean 1000x.
		"""
		super(LeafUnit, self).__init__(is_si)

		self._specifier = specifier

	def __reduce__(self):
		return (LeafUnit, (self._specifier, self.is_si()))

	__str__ = get_specifier
	latex = get_specifier
	def html(self):
		return cgi.escape(self.specifier)

	def str_includes_multiplier(self):
		return True

	def str_multiplier(self):
		return 1

	#def __lt__(self, other):
	#	 """In Python <= 2.7, objects without a __cmp__ method could be
	#	 implicitly compared by their ids. It gave a consistent order within a
	#	 single program run. The __lt__ method is now required for sorting, and
	#	 here we just use the identity which should have the same effect.
	#	 """
	#	 return id(self) < id(other)

	def __repr__(self):
		return ("LeafUnit(" +
				", ".join([repr(x) for x in [self.specifier,
											 self.is_si()]]) +
				")")

	def __mul__(self, other):
		if isinstance(other,numbers.Number):
			return ComposedUnit([self], [], other)
		elif isinstance(other, ComposedUnit): #hasattr(other, "numer"):
			return other * self
		elif isinstance(other, AbstractUnit):
			return ComposedUnit([self, other], [])
		else:
			return NotImplemented

	def __truediv__(self, other):
		if isinstance(other,numbers.Number):
			return self * (1 / other)
		elif hasattr(other, "invert"): # Not sure why both of these need to be here?
			return self * other.invert()
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

	def invert(self):
		"""Return (this unit)^-1"""
		return ComposedUnit([], [self])

	def canonical(self):
		"""Return the simplest representation of this unit.
		A LeafUnit is its own canonical form."""
		return self

	def normal(self):
		"""Return the unit sans multiplier.
		A LeafUnit is its own normal form."""
		return self

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		return (self.specifier,self.is_si())

	def squeeze(self):
		"""A LeafUnit has no implicit quantity."""
		return 1

	def __pow__(self, exponent):
		if exponent >= 0:
			return ComposedUnit([self] * exponent, [], 1)
		else:
			return self.invert() ** -exponent

	def __hash__(self):
		return hash((self.specifier,))
