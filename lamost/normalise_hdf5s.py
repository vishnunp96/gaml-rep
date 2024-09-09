import numpy
from scipy import interpolate
import os
import h5py

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

def pseudo_continuum(flux, ivar, wavelength, L=50):
	"""
	Pseudo-Continuum normalise a spectrum by dividing by a Gaussian-weighted smoothed spectrum.
	:param flux: The observed flux array.
	:type flux: ndarray
	:param ivar: The inverse variances of the fluxes.
	:type ivar: ndarray
	:param wavelength: An array of the wavelengths.
	:type wavelength: ndarray
	:param L: [optional] The width of the Gaussian in pixels.
	:type L: int
	:param dr: [optional] dara release
	:type dr: int
	:returns: Continuum normalized flux and flux uncerteinty
	:rtype: ndarray
	"""
	# Credit: https://github.com/henrysky/astroNN/blob/master/astroNN/lamost/chips.py
	# Partial Credit: https://github.com/chanconrad/slomp/blob/master/lamost.py
	smoothed_spec = gaussian_smooth(wavelength, flux, ivar, L)
	norm_flux = flux / smoothed_spec
	norm_ivar = smoothed_spec * ivar * smoothed_spec

	bad_pixel = ~numpy.isfinite(norm_flux)
	norm_flux[bad_pixel] = 1.0
	norm_ivar[bad_pixel] = 0.0

	return norm_flux, norm_ivar

def fit_chebyshev(flux, ivar, wavelength, deg, niter, usigma, lsigma):
	"""Fit the continuum with an iterative upper/lower rejection"""
	chpoly = numpy.polynomial.Chebyshev.fit(wavelength, flux, deg, w=ivar) # Initial fit
	for i in range(niter):
		continuum = chpoly(wavelength)
		residuals = flux - continuum
		sigma = numpy.std(residuals)
		mask = numpy.logical_And(residuals < usigma*sigma, residuals > -lsigma*sigma)
		flux[~mask] = chpoly(wavelength[~mask])
		chpoly = numpy.polynomial.Chebyshev.fit(wavelength, flux, deg, w=ivar) # Re-fit
	return chpoly

def chebyshev_smooth(flux, ivar, wavelength, deg, niter, usigma, lsigma):
	return fit_chebyshev(wavelength)

def fit_gaussian(flux, ivar, wavelength, L=50):
	return interpolate.interp1d(wavelength, gaussian_smooth(flux,ivar,wavelength,L=L), bounds_error=True)

def normalise(flux, continuum):
	norm_flux = flux / continuum

	bad_pixel = ~numpy.isfinite(norm_flux)
	norm_flux[bad_pixel] = 1.0

	return norm_flux

def normalise_data(data):
	continuum = gaussian_smooth(data[0,:], data[1,:], data[2,:])
	norm_flux = normalise(data[0,:], continuum)
	norm_flux = norm_flux - 1 ## Make standard value zero for better machine learning integration
	return interpolate.interp1d(data[2,:],norm_flux,bounds_error=True)(wavelengths)

def process_hdf5(hdf5path,results):

	print(f'Start: {os.path.basename(hdf5path)}',flush=True)
	cleancounter = 0
	filecounter = 0
	failurecounter = 0
	with h5py.File(hdf5path, 'r+', libver='latest') as archive:
		for groupname in archive:
			try:
				group = archive[groupname]

				if group.attrs['SNR_MEAN'] > 10:

					data = group['data'][0:3]

					continuum = gaussian_smooth(data[0,:], data[1,:], data[2,:])
					norm_flux = normalise(data[0,:], continuum)

					norm_flux = norm_flux - 1 ## Make standard value zero for better machine learning integration
					regularized = interpolate.interp1d(data[2,:],norm_flux,bounds_error=True)(wavelengths)

					group.create_dataset('continuum',dtype=numpy.float32,data=continuum, chunks=True, compression='lzf')
					group.create_dataset('normalized',dtype=numpy.float32,data=regularized, chunks=True)

					cleancounter += 1
				filecounter += 1
			except (KeyboardInterrupt, SystemExit):
				raise
			except Exception as e:
				print(f'Failure for {groupname} in {hdf5path}: {e}')
				failurecounter += 1

	print(f'End: {os.path.basename(hdf5path)} ({cleancounter} clean, {filecounter} files, {failurecounter} failures)',flush=True)
	results['clean'] += cleancounter
	results['files'] += filecounter
	results['failures'] += failurecounter

if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import sys
	print(' '.join(sys.argv))

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction
	from gaml.utilities.parallel import parallel_results
	from pprint import pprint

	parser = ArgumentParser()
	parser.add_argument('source',action=IterFilesAction, suffix='.hdf5')
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=10)
	args = parser.parse_args()

	results = parallel_results(process_hdf5, args.source, chunksize=args.chunksize, processes=args.processes)

	pprint(results)

	stopwatch.report()
