from units.compatibility import compatible
from units.exception import IncompatibleUnitsError

import cgi

class Uncertainty:
	def __init__(self,value=None,upper=None,lower=None,note=None):
		if value is not None:
			self.upper = abs(value)
			self.lower = abs(value)
		else:
			if upper is None or lower is None:
				raise ValueError('Badly formed uncertainty provided.')
			self.upper = abs(upper)
			self.lower = abs(lower)

		self.note = note

	def tostring(self, multiplier=1):
		result = ''
		if self.upper == self.lower:
			result += '\\pm ' + str(multiplier * self.upper)
		else:
			result += '^ +' + str(multiplier * self.upper) + ' _ -' + str(multiplier * self.lower)
		if self.note:
			result += ' (' + str(self.note) + ')'
		return result

	def tohtml(self, multiplier=1):
		result = ''
		if self.upper == self.lower:
			result += '\u00B1 ' + str(multiplier * self.upper)
		else:
			result += '<sup>+' + str(multiplier * self.upper) + '</sup><sub>-' + str(multiplier * self.lower) + '</sub>'
		if self.note:
			result += ' (' + cgi.escape(str(self.note)) + ')'
		return result

	def __str__(self):
		return self.tostring()

	def __repr__(self):
		return ("Uncertainty(" +
				", ".join([repr(x) for x in [self.upper,
											self.lower,
											self.note] if x])+
				")")

class Measurement:
	def __init__(self,value,unit,uncertainties=None,note=None):
		self.value = value
		self.unit = unit
		self.uncertainties = uncertainties if uncertainties else []
		self.note = note

		## Can we include a "factor" here, which could include simple number or named factor? (e.g. h or 10^5)

	def convert(self, unit):
		if compatible(unit, self.unit):
			return Measurement(
					value=self.value * self.unit.squeeze() / unit.squeeze(),
					unit=unit,
					uncertainties=[
						Uncertainty(
							upper=u.upper * self.unit.squeeze() / unit.squeeze(),
							lower=u.lower * self.unit.squeeze() / unit.squeeze(),
							note=u.note)
						for u in self.uncertainties],
					note=self.note)
		else:
			raise IncompatibleUnitsError()

	def __float__(self):
		return float(self.value)

	def __str__(self):
		##### This needs sorting
		##### But we need some method for dealing with dimensionless quantities first
		if self.unit and not self.unit.str_includes_multiplier():
			result = str(self.value * self.unit.str_multiplier())
		else:
			result = str(self.value)
		#result = str(self.value)
		if self.uncertainties:
			if self.unit and not self.unit.str_includes_multiplier():
				result += ' ' + ' '.join([u.tostring(self.unit.str_multiplier()) for u in self.uncertainties])
			else:
				result += ' ' + ' '.join([u.tostring() for u in self.uncertainties])
		unitstr = str(self.unit) if self.unit else ''
		if unitstr:
			result += ' ' + unitstr
		if self.note:
			result += ' (' + str(self.note) + ')'
		return result

	def html(self):
		if self.unit and not self.unit.str_includes_multiplier():
			result = str(self.value * self.unit.str_multiplier())
		else:
			result = str(self.value)
		if self.uncertainties:
			if self.unit and not self.unit.str_includes_multiplier():
				result += ' ' + ' '.join([u.tohtml(self.unit.str_multiplier()) for u in self.uncertainties])
			else:
				result += ' ' + ' '.join([u.tohtml() for u in self.uncertainties])
		unitstr = self.unit.html() if self.unit else ''
		if unitstr:
			result += ' ' + unitstr
		if self.note:
			result += ' (' + cgi.escape(str(self.note)) + ')'
		return result

	def __repr__(self):
		return ("Measurement(" +
				", ".join([repr(x) for x in [self.value,
											self.uncertainties,
											self.unit,
											self.note]])+
				")")

	def __eq__(self, other):
		if hasattr(other,'value'):
			if compatible(self.unit, other.unit):
				### This needs improving
				#return self.unit(self.value)==other.unit(other.value)
				return self.value==self.unit(other).value
		return NotImplemented
