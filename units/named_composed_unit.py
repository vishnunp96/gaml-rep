"""Assign arbitrary new symbols to composed units."""

from gaml.units.abstract import AbstractUnit
from gaml.units.composed_unit import ComposedUnit,simplify
from gaml.units.registry import REGISTRY

import numbers
import cgi

class NamedComposedUnit(AbstractUnit):
	"""A NamedComposedUnit is a composed unit with its own symbol."""

	def get_name(self):
		"""The label for the composed unit"""
		return self._name
	name = property(get_name)

	def get_composed_unit(self):
		"""The labeled composed unit"""
		return self._composed_unit
	composed_unit = property(get_composed_unit)

	def __new__(cls, name, composed_unit, is_si=False):
		"""Give a composed unit a new symbol."""
		# pylint: disable=W0613

		if name not in REGISTRY:
			REGISTRY[name] = super(NamedComposedUnit,cls).__new__(cls)
		return REGISTRY[name]

	def __init__(self, name, composed_unit, is_si=False):
		super(NamedComposedUnit, self).__init__(is_si)
		self._name = name
		self._composed_unit = simplify(composed_unit)

	def __hash__(self):
		return hash((self.name,self.composed_unit))

	def __reduce__(self):
		return (
			NamedComposedUnit,
			(self._name, self._composed_unit, self.is_si())
		)

	def invert(self):
		"""Return the invert of the underlying composed unit."""
		return ComposedUnit([],[self],1)

	def canonical(self):
		"""Return the canonical representation of the underlying composed unit."""
		return self.composed_unit.canonical()

	def normal(self):
		""" A NamedComposedUnit cannot have a multiplier. """
		return self

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		return self.composed_unit.immutable()

	def squeeze(self):
		"""Return the squeeze of the underlying composed unit."""
		return self.composed_unit.squeeze()

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

	__str__ = get_name
	latex = get_name
	def html(self):
		return cgi.escape(self.name)

	def str_includes_multiplier(self):
		return True

	def str_multiplier(self):
		return 1

	def __repr__(self):
		return ("NamedComposedUnit(" +
				", ".join([repr(x) for x in [self.name,
											self.composed_unit,
											self.is_si()]])+
				")")

	def __pow__(self, exponent):
		if exponent >=0:
			return ComposedUnit([self]*exponent,[],1)
		else:
			#return self.invert() ** -exponent
			return ComposedUnit([],[self]*-exponent,1)
