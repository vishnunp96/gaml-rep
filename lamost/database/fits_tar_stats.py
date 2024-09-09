import gzip
from astropy.io import fits
import tarfile

def process_tar(tarpath, results):

	print('Processing:',tarpath)

	accepted = 0

	with tarfile.open(tarpath, mode='r') as tf:
		for i,entry in enumerate(tf):
			if entry.isfile() and entry.name:
				try:
					with fits.open(gzip.open(tf.extractfile(entry),mode='r'), mode='readonly') as hdu_list:
						header = hdu_list[0].header

						if header['SIMPLE']: # True if the fule conforms to the FITS/LAMOST standard

							if 'OBJSOURC' in header: results['OBJSOURC'] += 1 # Doesn't use
							if 'TSOURCE' in header: results['TSOURCE'] += 1 # Used
							if 'TCOMMENT' in header: results['TCOMMENT'] += 1 # Used
							if 'TFROM' in header: results['TFROM'] += 1 # Used
							if 'OFFSET_V' in header: results['OFFSET_V'] += 1 # Used
							if 'Z_ERR' in header: results['Z_ERR'] += 1 # Used
							if 'Z_FROM' in header: results['Z_FROM'] += 1 # Doesn't use
							if 'ZFLAG' in header: results['ZFLAG'] += 1 # Used

							accepted += 1

				except OSError:
					print(f'Could not read {i}: {entry.name}, {entry.type}')

	results['accepted'] += accepted
	print(f'Finished {tarpath} (added {accepted} entries to database)', flush=True)


if __name__ == '__main__':

	import utilities.parallel as parallel
	from utilities.argparseactions import ArgumentParser,IterFilesAction

	parser = ArgumentParser(description='Collect stats from LAMOST FITS files.')
	parser.add_argument('source',action=IterFilesAction, recursive=True, suffix='.tar', help='Source directory for LAMOST .tar files, searched recursively.')
	parser.add_argument('-c','--chunksize', type=int, default=10, help='Chunksize for processing .tar files.')
	parser.add_argument('-p','--processes', type=int, default=1, help='Number of processes to use when processing .tar files.')
	args = parser.parse_args()

	results = parallel.parallel_results(process_tar, args.source, chunksize=args.chunksize, processes=args.processes)

	print(results)
