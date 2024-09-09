import os
import tarfile
import gzip
import h5py
import numpy
from astropy.io import fits

def process_tar(tarpath,results,targetpath):

	archivepath = os.path.join(targetpath,os.path.splitext(os.path.basename(tarpath))[0]+'.hdf5')
	print(f'Start: {os.path.basename(tarpath)} ({os.path.basename(archivepath)})',flush=True)
	filecounter = 0
	failurecounter = 0
	with tarfile.open(tarpath,mode='r') as tf:
		with h5py.File(archivepath, 'w', libver='latest') as archive:
			for entry in tf:
				if entry.isfile():
					try:
						with fits.open(gzip.open(tf.extractfile(entry),mode='r'),mode='readonly') as hdu_list:
							header = hdu_list[0].header
							data = hdu_list[0].data

							## Save data to archive
							group = os.path.basename(entry.name).split('.')[0]
							dataset = group + '/data'

							archive.create_dataset(dataset,dtype=numpy.float32,data=data, chunks=(1,header['NAXIS1']), compression='lzf')
							for tag in (i for i in header if i and i != 'COMMENT'):
								#print(tag,type(tag),tag == 'COMMENT')
								#print(header[tag],type(header[tag]))
								archive[group].attrs[tag] = header[tag]

							## Calculate additional metadata
							snr = (header['SNRU'] + header['SNRG'] + header['SNRR'] + header['SNRI'] + header['SNRZ']) / 5
							archive[group].attrs['SNR_MEAN'] = snr

						filecounter += 1
					except (KeyboardInterrupt, SystemExit):
						raise
					except Exception as e:
						print(f'Failure for {entry.name} in {tarpath}: {e}')
						failurecounter += 1

	print(f'End: {os.path.basename(tarpath)} ({filecounter} files, {failurecounter} failures)',flush=True)
	results['files'] += filecounter
	results['failures'] += failurecounter


if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import sys
	print(' '.join(sys.argv))

	from utilities.argparseactions import ArgumentParser,IterFilesAction,DirectoryAction
	from utilities.parallel import parallel_results
	from pprint import pprint

	parser = ArgumentParser()
	parser.add_argument('source',action=IterFilesAction, suffix='.tar')
	parser.add_argument('target',action=DirectoryAction,mustexist=False,mkdirs=True)
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=10)
	args = parser.parse_args()

	results = parallel_results(process_tar, args.source, additional_args=(args.target,), chunksize=args.chunksize, processes=args.processes)

	pprint(results)

	stopwatch.report()

