if __name__ == '__main__':

	import gzip
	import tarfile

	from astropy.io import fits

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	parser = ArgumentParser()
	parser.add_argument('source',action=FileAction, mustexist=True)
	args = parser.parse_args()

	counter = 0
	found = 0

	with tarfile.open(args.source,mode='r') as tf:
		for i,entry in enumerate(tf):
			if entry.isfile():
				try:
					hdu_list = fits.open(gzip.open(tf.extractfile(entry),mode='r'),mode='readonly')
					header = hdu_list[0].header

					snr = (header['SNRU'] + header['SNRG'] + header['SNRR'] + header['SNRI'] + header['SNRZ']) / 5

					if snr > 10:
						found += 1
				except OSError:
					print(f'Could not read {entry.name}')
				except Exception:
					print('\n',flush=True)
					raise
				counter += 1
			if i%100==0:
				print(f'\rFiles: {counter:10d}    Found: {found:10d}    ({100*found/counter:5.2f}%)',end='')
	print()
