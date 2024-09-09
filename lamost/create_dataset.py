if __name__ == '__main__':

	import h5py
	import pandas

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import sys
	print(' '.join(sys.argv))

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction,FileAction

	parser = ArgumentParser()
	parser.add_argument('source',action=IterFilesAction, suffix='.hdf5',help='HDF5 source file(s).')
	parser.add_argument('output',action=FileAction, mustexist=False,help='Location to save CSV dataset.')
	args = parser.parse_args()

	data = pandas.DataFrame()

	for filename in args.source:
		data_list = []
		target_list = []
		index_list = []
		try:
			with h5py.File(filename,'r') as h5file:
				for groupname in h5file:
					group = h5file[groupname]
					if 'normalized' in group and 'teff' in group.attrs:
						target_list.append(group.attrs['teff'])
						data_list.append(group['normalized'][:])
						index_list.append(groupname)
		except (KeyboardInterrupt, SystemExit):
			raise
		except Exception as e:
			print(f'Error for {filename}: {e}')

		filedata = pandas.DataFrame(data_list,index=index_list)
		filedata['teff'] = target_list

		data = data.add(filedata,fill_value=0)

		if len(data.index)>100000:
			break

	data.to_csv(args.output,float_format='%.5f',index_label='filename')

	stopwatch.report()
