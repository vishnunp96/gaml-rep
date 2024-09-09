import numpy
from scipy import interpolate
from tar2sql import register_numpy_array_type

# lambda_range = (start=3905,stop=9000)
wavelengths = numpy.arange(3905,9000+1,1)

def gaussian_smooth(flux, ivar, wavelength, L=50):
	"""
	Smooth a spectrum with a running Gaussian.
	:param flux: The observed flux array.
	:type flux: ndarray
	:param ivar: The inverse variances of the fluxes.
	:type ivar: ndarray
	:param wavelength: An array of the wavelengths.
	:type wavelength: ndarray
	:param L: The width of the Gaussian in pixels.
	:type L: int
	:returns: An array of smoothed fluxes
	:rtype: ndarray
	"""
	# Credit: https://github.com/henrysky/astroNN/blob/master/astroNN/lamost/chips.py
	# Partial Credit: https://github.com/chanconrad/slomp/blob/master/lamost.py
	w = numpy.exp(-0.5 * (wavelength[:, None] - wavelength[None, :]) ** 2 / L ** 2)
	denominator = numpy.dot(ivar, w.T)
	numerator = numpy.dot(flux * ivar, w.T)
	bad_pixel = denominator == 0
	smoothed = numpy.zeros(numerator.shape)
	smoothed[~bad_pixel] = numerator[~bad_pixel] / denominator[~bad_pixel]
	return smoothed

def normalise(flux, continuum):
	norm_flux = flux / continuum

	bad_pixel = ~numpy.isfinite(norm_flux)
	norm_flux[bad_pixel] = 1.0

	return norm_flux

if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import os
	import sqlite3
	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	#from pprint import pprint

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	register_numpy_array_type()

	parser = ArgumentParser()
	parser.add_argument('source',action=FileAction, mustexist=True)
	parser.add_argument('output',action=DirectoryAction, mustexist=False, mkdirs=True)
	parser.add_argument('-snr','--signal-to-noise',type=float,default=10.0, help='Minimum signal-to-noise ratio required for spectra.')
	args = parser.parse_args()

	# Can use check_same_thread=False here, as this is a read-only connection
	# (and the same thread requirement is intended for write operations)
	connection = sqlite3.connect('file:'+args.source+'?mode=ro', uri=True, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
	cursor = connection.cursor()

	cursor.execute('SELECT id,snr_mean,data FROM fits LIMIT 1000')
	rows = iter(cursor)

	accepted = 0

	for row_id,snr_mean,data in rows:
		### Try cutting on snr_z -> plot snr band values and look at results
		if snr_mean >= args.signal_to_noise:
			accepted +=1

			continuum = gaussian_smooth(data[0,:], data[1,:], data[2,:])
			norm_flux = normalise(data[0,:], continuum) - 1 ## Make standard value zero for better machine learning integration

			regularized = interpolate.interp1d(data[2,:],norm_flux,bounds_error=True)(wavelengths)

			fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.4, 6.4))

			ax1.plot(data[2], data[0], label='Initial')
			ax1.plot(data[2], continuum, label='Continuum')
			ax1.set_ylabel('Flux')
			ax1.legend()

			ax2.plot(data[2], data[1], label='Inverse Variance')
			ax2.set_ylabel('Inverse Variance')
			ax2.legend()

			ax3.plot(wavelengths, regularized, label='Normalized')
			ax3.set_ylabel('Normalized Flux')
			ax3.legend()

			xmin = min([min(a.get_xlim()) for a in (ax1,ax2,ax3)])
			xmax = max([max(a.get_xlim()) for a in (ax1,ax2,ax3)])

			for a in (ax1,ax2,ax3):
				a.set_xlim((xmin, xmax))

			print(f'Saving {row_id}')
			fig.savefig(os.path.join(args.output,f'spectra{row_id:06d}.png'), dpi=300)

			if accepted >=100:
				break

	connection.close()

	stopwatch.report()
