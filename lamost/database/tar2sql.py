import sqlite3
import gzip
from astropy.io import fits
import tarfile
from utilities.sqliteutils import register_numpy_array_type

def init_database(cursor):

	## Needed for some strange reason
	cursor.execute('PRAGMA foreign_keys = 1')
	cursor.execute('PRAGMA busy_timeout = 300000') # 5 min

	cursor.execute('''
		CREATE TABLE fits
			(
				id INTEGER PRIMARY KEY NOT NULL,
				filename TEXT, -- FITS filename
				date TEXT, -- Date when FITS file was created
				camver TEXT, -- Camera program version for this spectra
				date_obs TEXT, -- Observation median UTC
				date_beg TEXT, -- Observation start local time
				date_end TEXT, -- Observation end local time
				mjd INTEGER, -- Modified Julian Day
				planid TEXT, -- Plan ID
				ra REAL, -- RA from input catalog
				dec REAL, -- Declination from input catalog
				ra_obs REAL, -- Fiber RA during observation
				dec_obs REAL, -- Fiber declination during observation
				offset TEXT, -- Boolean, indicating fiber offset for target
				offset_v REAL, -- Offset value
				desig TEXT, -- Designation of LAMOST target
				fiberid INTEGER, -- Fiber ID
				cell_id TEXT, -- Fiber unit ID on the focal plane
				x_value REAL, -- X-coord on the focal plane
				y_value REAL, -- Y-coord on the focal plane
				objname TEXT, -- Name of object from RA, DEC, and HTM method
				objtype TEXT, -- Class of objects in input catalog
				t_source TEXT, -- Name of person/organisation that submitted input catalog
				t_comment TEXT, -- Target ID from other catalog?
				t_from TEXT, -- Input catalog
				fibertype TEXT, -- Fiber type (keywords given in LAMOST data description)
				magtype TEXT, -- Magnitude type of the target
				mag1 REAL, mag2 REAL, mag3 REAL, mag4 REAL, mag5 REAL, mag6 REAL, mag7 REAL, -- Magnitude values
				obs_type TEXT, -- Type of observation target
				radecsys TEXT, -- Equatorial coordinate system based on the J2000 position
				spid INTEGER, -- Spectrograph ID
				verspipe TEXT, -- Version of the LAMOST pipeline
				class TEXT, -- Object class as determined by LAMOST
				subclass TEXT, -- Object subclass (only used for stars) as determined by LAMOST
				z REAL, -- Redshift of object, as determined by LAMOST (set to -9999 if unknown)
				z_err REAL, -- Redshift error for object, as determined by LAMOST
				zflag TEXT, -- Method used to determine redshift?
				sn_u REAL, sn_g REAL, sn_r REAL, sn_i REAL, sn_z REAL, -- SNR for colour bands
				snr_mean REAL, -- Calculated mean of LAMOST SNR values
				data ARRAY NOT NULL, -- NumPy array of spectra data
				naxis1 INTEGER -- Number of wavelength values for this spectra
			)''')


def process_tar(tarpath, results, databasepath, verbose=False):

	if verbose: print(f'Processing: {tarpath}', flush=True)

	accepted = 0
	rows = []

	def insert_rows(to_add):
		## Add rows to database
		for i in range(10):
			try:
				with sqlite3.connect(databasepath, timeout=300) as connection: # Timeout 5 minutes
					cursor = connection.cursor()
					cursor.executemany(
						'''INSERT INTO fits (
								filename,
								date,
								camver,
								date_obs,
								date_beg,
								date_end,
								mjd,
								planid,
								ra,
								dec,
								ra_obs,
								dec_obs,
								offset,
								offset_v,
								desig,
								fiberid,
								cell_id,
								x_value,
								y_value,
								objname,
								objtype,
								t_source,
								t_comment,
								t_from,
								fibertype,
								magtype,
								mag1,
								mag2,
								mag3,
								mag4,
								mag5,
								mag6,
								mag7,
								obs_type,
								radecsys,
								spid,
								verspipe,
								class,
								subclass,
								z,
								z_err,
								zflag,
								sn_u,
								sn_g,
								sn_r,
								sn_i,
								sn_z,
								snr_mean,
								data,
								naxis1
							) VALUES(''' + ','.join('?'*50) + ');', to_add)
					cursor.close()
					#connection.commit()
					#connection.close()

					return
			except sqlite3.OperationalError:
				pass
		raise Exception(f'Could not write entries for {tarpath}')


	with tarfile.open(tarpath, mode='r') as tf:
		for i,entry in enumerate(tf):
			if entry.isfile() and entry.name:
				try:
					with fits.open(gzip.open(tf.extractfile(entry),mode='r'), mode='readonly') as hdu_list:
						header = hdu_list[0].header

						if header['SIMPLE']: # True if the fule conforms to the FITS/LAMOST standard

							data = hdu_list[0].data
							snr_mean = (header['SNRU'] + header['SNRG'] + header['SNRR'] + header['SNRI'] + header['SNRZ']) / 5

							rows.append((
									header['FILENAME'],
									header['DATE'],
									header['CAMVER'],
									header['DATE-OBS'],
									header['DATE-BEG'],
									header['DATE-END'],
									header['MJD'],
									header['PLANID'],
									header['RA'],
									header['DEC'],
									header['RA_OBS'],
									header['DEC_OBS'],
									header['OFFSET'],
									header['OFFSET_V'],
									header['DESIG'],
									header['FIBERID'],
									header['CELL_ID'],
									header['X_VALUE'],
									header['Y_VALUE'],
									header['OBJNAME'],
									header['OBJTYPE'],
									header['TSOURCE'],
									header['TCOMMENT'],
									header['TFROM'],
									header['FIBERTYP'],
									header['MAGTYPE'],
									header['MAG1'],
									header['MAG2'],
									header['MAG3'],
									header['MAG4'],
									header['MAG5'],
									header['MAG6'],
									header['MAG7'],
									header['OBS_TYPE'],
									header['RADECSYS'],
									header['SPID'],
									header['VERSPIPE'],
									header['CLASS'],
									header['SUBCLASS'],
									header['Z'],
									header['Z_ERR'],
									header['ZFLAG'],
									header['SNRU'],
									header['SNRG'],
									header['SNRR'],
									header['SNRI'],
									header['SNRZ'],
									snr_mean,
									#'\n'.join([','.join(numpy.char.mod('%f', i)) for i in data]),
									data,
									header['naxis1']
								))

							accepted += 1
				except OSError:
					print(f'Could not read {entry.name}')

				if len(rows) > 0 and (len(rows) % 500) == 0:
					insert_rows(rows)
					rows = []

	if len(rows) > 0:
		insert_rows(rows)

	results['accepted'] += accepted
	if verbose: print(f'Finished {tarpath} (added {accepted} entries to database)', flush=True)


if __name__ == '__main__':

	import os
	import utilities.parallel as parallel
	from utilities.argparseactions import ArgumentParser,FileAction,IterFilesAction

	register_numpy_array_type()

	def make_database(path, overwrite=False):
		if overwrite:
			if os.path.exists(path):
				response = input(f'Do you wish to overwrite the database at {path}? [YES/NO] ')
				if response == 'YES':
					os.remove(path)
					print(f'Remove database file at {path}. Reinitialising.')
				else:
					print('Did not overwrite database.')
					return False
		elif os.path.exists(args.database):
			print(f'File {path} already in use. You may use the --overwrite flag to reinitialise the file (all existing data will be lost).')
			return False

		connection = sqlite3.connect(path)
		cursor = connection.cursor()
		init_database(cursor)
		cursor.close()
		connection.commit()
		connection.close()

		print(f'Created database at {path}')

		return True

	def run_init(args):
		make_database(args.database, overwrite=args.overwrite)

	def run_populate(args):

		success = make_database(args.database, overwrite=args.overwrite)

		if not success:
			return

		results = parallel.parallel_results(process_tar, args.source, additional_args=(args.database, True), chunksize=args.chunksize, processes=args.processes)

		print(f'Populated database at {args.database} with {results["accepted"]} spectra')

	parser = ArgumentParser(description='Create and/or populate a database for LAMOST FITS files.')
	subparsers = parser.add_subparsers()
	parser_init = subparsers.add_parser('init',help='Initialise a new database to be written to by another program.')
	parser_init.add_argument('database', action=FileAction, mustexist=False, help='Path at which to create database.')
	parser_init.add_argument('--overwrite', action='store_true', help='Flag to indicate that an existing database at the specified path should be wiped and reinitialised (will ask for conformation).')
	parser_init.set_defaults(main=run_init)
	parser_pop = subparsers.add_parser('populate',help='Create a database and copy in data from LAMOST tar files.')
	parser_pop.add_argument('source',action=IterFilesAction, recursive=True, suffix='.tar', help='Source directory for LAMOST .tar files, searched recursively.')
	parser_pop.add_argument('database',action=FileAction,mustexist=False,help='Database file to use (fill create if it does not exist).')
	parser_pop.add_argument('-c','--chunksize', type=int, default=10, help='Chunksize for processing .tar files.')
	parser_pop.add_argument('-p','--processes', type=int, default=1, help='Number of processes to use when processing .tar files.')
	parser_pop.add_argument('--overwrite', action='store_true', help='Flag to indicate that an existing database at the specified path should be wiped and reinitialised (will ask for conformation).')
	parser_pop.set_defaults(main=run_populate)

	args = parser.parse_args()
	args.main(args)
