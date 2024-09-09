import re
import numbers
import six
import numpy

def getNumeric(s):
	if s:
		if isinstance(s, six.string_types):
			m = re.search('(?P<number>[0-9]+)',s.strip())
			if m:
				return int(m.group('number'))
		elif isinstance(s, numbers.Number):
			return s
	return int(-1)


def is_outlier(points, threshold=3.5):
	"""
	Answer from: https://stackoverflow.com/a/22357811
	Returns a boolean array with True if points are outliers and False
	otherwise.

	Parameters:
	-----------
		points : An numobservations by numdimensions array of observations
		threshold : The modified z-score to use as a threshold. Observations with
			a modified z-score (based on the median absolute deviation) greater
			than this value will be classified as outliers.

	Returns:
	--------
		mask : A numobservations-length boolean array.

	References:
	----------
		Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
		Handle Outliers", The ASQC Basic References in Quality Control:
		Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
	"""
	if len(points.shape) == 1:
		points = points[:,None]
	median = numpy.median(points, axis=0)
	diff = numpy.sum((points - median)**2, axis=-1)
	diff = numpy.sqrt(diff)
	med_abs_deviation = numpy.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation
	return modified_z_score > threshold
