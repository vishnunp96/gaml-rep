"""Aliased units are just holders for another unit,
	which they farm all their functionality to.
	They are useful as placeholders in the registry.
	Their only addition is the ability to declare
	whether they are SI-prefix compatible separately
	to their base."""

from units.abstract import AbstractUnit
from units.registry import REGISTRY

class AliasedUnit(AbstractUnit):
	"""Holder for another unit, that can specify
	SI-prefic compatibility separately."""

	def get_base(self):
		"""Return the aliased base unit."""
		return self._base_unit
	base = property(get_base)

	def __new__(cls, alias, base_unit, is_si=False):
		# pylint: disable=W0613
		if alias not in REGISTRY:
			REGISTRY[alias] = super(AliasedUnit, cls).__new__(cls)
		return REGISTRY[alias]

	def __init__(self, alias, base_unit, is_si=False):
		"""Make a new AliasedUnit with the given name and
		SI-compatibility. A unit that is SI compatible can be prefixed,
		e.g. with k to mean 1000x.
		"""
		super(AliasedUnit, self).__init__(is_si)

		self._alias = alias
		self._base_unit = base_unit

	def __reduce__(self):
		return self.base.__reduce__()

	def __str__(self):
		return self.base.__str__()
	def latex(self):
		return self.base.latex()
	def html(self):
		return self.base.html()

	def str_includes_multiplier(self):
		return self.base.str_includes_multiplier()

	def str_multiplier(self):
		return self.base.str_multiplier()

	def __repr__(self):
		return ("AliasedUnit(" +
				", ".join([repr(x) for x in [self._alias,
											self.base,
											self.is_si()]]) +
				")")

	def __mul__(self, other):
		return self.base * other

	def __truediv__(self, other):
		return self.base / other

	def __rmul__(self,other):
		return other * self.base

	def __rtruediv__(self,other):
		return other * self.base.invert()

	# Backwards-compatibility for <= Python 2.7
	__div__ = __truediv__
	__rdiv__ = __rtruediv__

	def invert(self):
		"""Return (this unit)^-1"""
		return self.base.invert()

	def canonical(self):
		"""Return the simplest representation of this unit."""
		return self.base.canonical()

	def normal(self):
		return self.base.normal()

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		return self.base.immutable()

	def squeeze(self):
		return self.base.squeeze()

	def __pow__(self, exponent):
		return self.base**exponent

	def __hash__(self):
		return hash(self.base)

