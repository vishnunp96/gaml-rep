if __name__ == '__main__':

	import pandas
	import h5py

	import os
	from pprint import pprint
	from gaml.utilities.parallel import parallel_results

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import sys
	print(' '.join(sys.argv))

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction

	parser = ArgumentParser()
	parser.add_argument('data',action=FileAction,mustexist=True,help='CSV file containing data to be added to HDF5 files in \'source\'.')
	parser.add_argument('source',action=DirectoryAction, mustexist=True,help='Directory containing HDF5 files to which data will be added.')
	parser.add_argument('columns',nargs='+',help='Columns from data to be added to source files.')
	parser.add_argument('-p','--processes',type=int,default=1)
	parser.add_argument('-c','--chunksize',type=int,default=10)
	args = parser.parse_args()

	print('Columns to include: ' + str(args.columns))

	data = pandas.read_csv(args.data,low_memory=False)

	print('Loaded data.')

	if not all(col in data for col in args.columns):
		raise ValueError(f'Not all requested columns available in provided data.')

	#h5dict = dict()

	def add_to_hdf5(planid_and_data,results):
		planid,plandata = planid_and_data
		counter = 0
		print(f'Start processing {planid}.')
		with h5py.File(os.path.join(args.source,planid+'.hdf5'),'r+') as h5file:
			for index,row in plandata.iterrows():
				try:
					groupname = f'spec-{row["lmjd"]}-{planid}_sp{int(row["spid"]):02d}-{int(row["fiberid"]):03d}'
					group = h5file[groupname]

					#if groupname not in h5dict[planid]:
					#	raise ValueError(f'Error for {planid}/{groupname}')

					for col in args.columns:
						group.attrs[col] = row[col]

					counter += 1
				except (KeyboardInterrupt, SystemExit):
					raise
				except Exception as e:
					print(f'Error on row {index}: {e}')
		print(f'Finish processing {planid} ({counter} rows added)')
		results['rows'] += counter
		results['files'] += 1

	results = parallel_results(add_to_hdf5, [(planid,plandata.copy()) for planid,plandata in data.groupby('planid')], chunksize=args.chunksize, processes=args.processes)

	pprint(results)

	stopwatch.report()
