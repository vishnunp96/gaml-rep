import numpy
from scipy import interpolate
from utilities.sqliteutils import register_numpy_array_type

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

def process_rows(rows, results, snr_cutoff, databasepath, verbose=False):

	print('Start')

	def insert_rows(to_add):
		## Add rows to database
		for i in range(10):
			try:
				with sqlite3.connect(databasepath, timeout=300) as connection: # Timeout 5 minutes
					cursor = connection.cursor()
					cursor.executemany(
						'''INSERT INTO normalized (
								id,
								data
							) VALUES(?,?);''', to_add)
					cursor.close()

					return
			except sqlite3.OperationalError as e:
				print(e)
		raise Exception('Could not write data.')

	accepted = 0
	rejected = 0

	normalized = []

	with numpy.errstate(divide='ignore', invalid='ignore'):
		for row_id,snr_mean,data in rows:

			if snr_mean >= snr_cutoff:
				accepted += 1

				continuum = gaussian_smooth(data[0,:], data[1,:], data[2,:])
				norm_flux = normalise(data[0,:], continuum) - 1 ## Make standard value zero for better machine learning integration

				regularized = interpolate.interp1d(data[2,:],norm_flux,bounds_error=True)(wavelengths)

				#group.create_dataset('continuum',dtype=numpy.float32,data=continuum, chunks=True, compression='lzf')
				#group.create_dataset('normalized',dtype=numpy.float32,data=regularized, chunks=True)

				normalized.append((row_id, regularized))
				### SHOULD WE SAVE EVERY Nth ENTRY BE SAVED SEPARATELY FOR TESTING PURPOSES?

			else:
				rejected += 1

	if verbose: print(f'Processed {accepted+rejected} entries (rejected {rejected})', flush=True)

	if len(normalized) > 0:
		insert_rows(normalized)

	results['accepted'] += accepted
	results['rejected'] += rejected
	if verbose: print(f'Added {accepted} entries to database (rejected {rejected})', flush=True)

	return results

def init_norm_database(cursor):

	## Needed for some strange reason
	cursor.execute('PRAGMA foreign_keys = 1')
	cursor.execute('PRAGMA busy_timeout = 300000') # 5 min

	# Table to store the wavelength values for the normalized spectra
	cursor.execute('CREATE TABLE wavelengths(lambda REAL NOT NULL)')
	cursor.executemany('INSERT INTO wavelengths (lambda) VALUES(?)', ((float(l),) for l in wavelengths))

	# Table to hold number of rows in normalized table (for quicker access later)
	cursor.execute('''
		CREATE TABLE rowcounts
			(
				tablename TEXT PRIMARY KEY NOT NULL, -- Table name from this database
				numrows INTEGER NOT NULL -- Number of rows in the corresponding table in this databse
			)''')

	cursor.execute('''
		CREATE TABLE normalized
			(
				id INTEGER PRIMARY KEY NOT NULL, -- Integer ID should reference parent database of raw FITS data
				data ARRAY NOT NULL -- NumPy array of normalized spectra data
			)''')


if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import os
	import sqlite3
	from utilities.argparseactions import ArgumentParser,FileAction
	from utilities.parallel import parallel_results_batches
	#from pprint import pprint

	register_numpy_array_type()

	parser = ArgumentParser()
	parser.add_argument('source',action=FileAction, mustexist=True)
	parser.add_argument('output',action=FileAction)
	parser.add_argument('-snr','--signal-to-noise',type=float,default=10.0, help='Minimum signal-to-noise ratio required for spectra.')
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=1000)
	args = parser.parse_args()

	if os.path.exists(args.output):
		raise ValueError(f'{args.output} already exists.')

	# Create the appropriate tables for normalized data in the database (deleting any which may already be present)
	with sqlite3.connect(args.output, detect_types=sqlite3.PARSE_DECLTYPES) as connection:
		cursor = connection.cursor()
		init_norm_database(cursor)
		cursor.close()

	# Can use check_same_thread=False here, as this is a read-only connection
	# (and the same thread requirement is intended for write operations)
	connection = sqlite3.connect('file:'+args.source+'?mode=ro', uri=True, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
	cursor = connection.cursor()

	cursor.execute('SELECT id,snr_mean,data FROM fits')
	rows = iter(cursor)

	print('Setup complete')

	results = parallel_results_batches(process_rows, rows, additional_args=(args.signal_to_noise,args.output,True), chunksize=args.chunksize, processes=args.processes)

	connection.close()

	with sqlite3.connect(args.output, timeout=300) as connection: # Timeout 5 minutes
		cursor = connection.cursor()
		cursor.execute(
			'''INSERT INTO rowcounts (
					tablename,
					numrows
				) VALUES(?,?);''', ('normalized', results['accepted']))
		cursor.close()

	#pprint(results)
	print({i:results[i] for i in ['accepted', 'rejected']})

	stopwatch.report()
