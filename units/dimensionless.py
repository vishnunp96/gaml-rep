from gaml.units.abstract import AbstractUnit
import numbers

class DimensionlessUnit(AbstractUnit):

	def __init__(self, factor):
		super(DimensionlessUnit, self).__init__(is_si=False) # Doesn't make sense for factor to be SI

		self.factor = factor

	def __hash__(self):
		return hash((self.factor,))

	def canonical(self):
		"""Return the simplest representation of this unit,
		dropping any multiplier."""
		return DimensionlessUnit(1)

	def normal(self):
		"""Return the unit sans any multipliers."""
		return DimensionlessUnit(1)

	def immutable(self):
		"""Return an immutable, comparable derivative of this unit,
		dropping any multiplier."""
		return tuple()

	def invert(self):
		"""Return (this unit)^-1."""
		return DimensionlessUnit(1/self.factor)

	def is_si(self):
		"""Whether it makes sense to give this unit an SI prefix."""
		return self._si

	def squeeze(self):
		"""Return this unit's implicit quantity multiplier."""
		return self.factor

	def str_includes_multiplier(self):
		"""Whether the string name of the unit already encapsulates
		the unit's multiplier."""
		return False

	def str_multiplier(self):
		return self.factor

	def __repr__(self):
		return ("DimensionlessUnit(" +
				", ".join([repr(x) for x in [self.factor]]) +
				")")

	def __str__(self):
		return ''
	def latex(self):
		return ''
	def html(self):
		return ''

	def __mul__(self, other):
		if isinstance(other,numbers.Number):
			return DimensionlessUnit(self.factor * other)
		else:
			return self.factor * other

	def __truediv__(self, other):
		if isinstance(other,numbers.Number):
			return DimensionlessUnit(self.factor / other)
		else:
			return self.factor / other

	def __rmul__(self,other):
		return self * other

	def __rtruediv__(self,other):
		return other * self.invert()

	# Backwards-compatibility for <= Python 2.7
	__div__ = __truediv__
	__rdiv__ = __rtruediv__

	def __pow__(self, exponent):
		return DimensionlessUnit(self.factor ** exponent)

	#def __eq__(self,other):
	#	if isinstance(other,DimensionlessUnit):
	#		return self.factor == other.factor
	#	else:
	#		return False
