if __name__ == '__main__':

	import pandas
	import h5py

	import os

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import sys
	print(' '.join(sys.argv))

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction

	parser = ArgumentParser()
	parser.add_argument('data',action=FileAction,mustexist=True,help='CSV file containing data to be added to HDF5 files in \'source\'.')
	parser.add_argument('source',action=DirectoryAction, mustexist=True,help='Directory containing HDF5 files to which data will be added.')
	parser.add_argument('columns',nargs='+',help='Columns from data to be added to source files.')
	args = parser.parse_args()

	print('Columns to include: ' + str(args.columns))

	data = pandas.read_csv(args.data,low_memory=False)

	print('Loaded data.')

	if not all(col in data for col in args.columns):
		raise ValueError(f'Not all requested columns available in provided data.')

	#h5dict = dict()

	counter = 0
	totalcount = len(data.index)
	#try:
	for planid,plandata in data.groupby('planid'):
		with h5py.File(os.path.join(args.source,planid+'.hdf5'),'r+') as h5file:
			for index,row in plandata.iterrows():
				try:
					counter += 1

					groupname = f'spec-{row["lmjd"]}-{planid}_sp{int(row["spid"]):02d}-{int(row["fiberid"]):03d}'
					group = h5file[groupname]

					#if groupname not in h5dict[planid]:
					#	raise ValueError(f'Error for {planid}/{groupname}')

					for col in args.columns:
						group.attrs[col] = row[col]

					if counter%100 == 0:
						print(f'\rRows: {counter:10d}/{totalcount}  ({100*counter/totalcount:5.1f}%)',end='')
				except (KeyboardInterrupt, SystemExit):
					raise
				except Exception as e:
					print(f'\nError on row {index}: {e}')

	print()
	#finally:
		#print()
		#closedcount = 0
		#for f in h5dict.values():
		#	try:
		#		f.close()
		#		closedcount += 1
		#	except Exception as e:
		#		print(f'Could not close a file: {e}') # Can we safely access name?
		#print(f'Successfully closed {closedcount} files.')

	stopwatch.report()
