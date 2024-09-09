"""An abstract base class to define the interface for all units."""

from units.quantity import Quantity

import numbers

class MetaUnit(object):
	"""Parent class/interface for units."""

	def __init__(self, is_si):
		self._si = is_si

	def __call__(self, quantity):
		"""Overload the function call operator to convert units."""
		if hasattr(quantity,'convert'):
			return quantity.convert(self)
		elif isinstance(quantity, numbers.Number):
			return Quantity(quantity, self)
		elif quantity is None:
			return None
		else:
			raise ValueError('Quantity must be a number, None, or implement a \'convert(unit)\' method.')

	def canonical(self):
		"""Return the simplest representation of this unit,
		dropping any multiplier."""
		raise NotImplementedError

	def normal(self):
		"""Return the unit sans any multipliers.
		i.e. unit.str_multiplier() == 1"""
		raise NotImplementedError

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		raise NotImplementedError

	def invert(self):
		"""Return (this unit)^-1."""
		raise NotImplementedError

	def is_si(self):
		"""Whether it makes sense to give this unit an SI prefix."""
		return self._si

	def squeeze(self):
		"""Return this unit's implicit quantity multiplier."""
		raise NotImplementedError

	def str_includes_multiplier(self):
		"""Whether the string name of the unit already encapsulates
		the unit's multiplier."""
		raise NotImplementedError

	def str_multiplier(self):
		"""Any multiplier not encapsulated in the string name."""
		raise NotImplementedError

	def __lt__(self, other):
		"""In Python <= 2.7, objects without a __cmp__ method could be
		implicitly compared by their ids. It gave a consistent order within a
		single program run. The __lt__ method is now required for sorting, and
		here we just use the identity which should have the same effect.
		"""
		## Move to using str representation of a unit for comparison
		## Does this break anywhere...?
		assert isinstance(other,AbstractUnit)
		return str(self) < str(other)

	#def __hash__(self):
	#	return hash(self.immutable())
	def __eq__(self, other):
		return hash(self) == hash(other)

class AbstractUnit(MetaUnit):
	pass
