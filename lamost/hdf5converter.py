if __name__ == '__main__':

	import gzip
	import tarfile
	import h5py
	import numpy

	from astropy.io import fits

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	parser = ArgumentParser()
	parser.add_argument('source',action=FileAction, mustexist=True)
	parser.add_argument('output',action=FileAction, mustexist=False,help='.hdf5 file in which to store data.')
	args = parser.parse_args()

	counter = 0

	with h5py.File(args.output, 'w', libver='latest') as archive:
		with tarfile.open(args.source,mode='r') as tf:
			for i,entry in enumerate(tf):
				try:
					with fits.open(gzip.open(tf.extractfile(entry),mode='r'),mode='readonly') as hdu_list:
						header = hdu_list[0].header
						data = hdu_list[0].data

						## Save data to archive
						parts = entry.name.split('/')
						group = '/'.join(parts[1:-1] + [parts[-1].split('.')[0]])
						dataset = group + '/data'

						archive.create_dataset(dataset,dtype=numpy.float32,data=data, chunks=(1,header['NAXIS1']), compression='lzf')
						for tag in (i for i in header if i and i != 'COMMENT'):
							#print(tag,type(tag),tag == 'COMMENT')
							#print(header[tag],type(header[tag]))
							archive[group].attrs[tag] = header[tag]

						## Calculate additional metadata
						snr = (header['SNRU'] + header['SNRG'] + header['SNRR'] + header['SNRI'] + header['SNRZ']) / 5
						archive[group].attrs['SNR_MEAN'] = snr

				except OSError:
					print(f'Could not read {entry.name}')
				except Exception:
					print('\n',flush=True)
					raise
				counter += 1
				print(f'\rFiles: {counter:10d}',end='')
	print()
