if __name__ == '__main__':

	import tarfile
	import os.path as path

	from utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction

	parser = ArgumentParser()
	parser.add_argument('source',action=FileAction, mustexist=True)
	parser.add_argument('destination',action=DirectoryAction, mustexist=False,mkdirs=True)
	args = parser.parse_args()

	counter = 0
	tarcount = 0

	with tarfile.open(args.source,mode='r') as tf:

		try:

			#subtf = None
			#folder = None
			subtfdict = dict()

			for i,entry in enumerate(tf):
				if entry.isfile():
					parts = entry.name.split('/')
					filename = parts[-1]
					entryfolder = parts[-2]

					#if folder != entryfolder:
					#	if subtf is not None: subtf.close()
					#	tarname = path.join(args.destination,entryfolder+'.tar')
					#	subtf = tarfile.open(tarname,'a')
					#	folder = entryfolder
					#	tarcount += 1
					if entryfolder not in subtfdict:
						tarname = path.join(args.destination,entryfolder+'.tar')
						subtfdict[entryfolder] = tarfile.open(tarname,'x')
						tarcount += 1

					entry.name = filename
					#subtf.addfile(entry,tf.extractfile(entry))
					subtfdict[entryfolder].addfile(entry,tf.extractfile(entry))

					counter += 1

					if counter%1000 == 0:
						print(f'\rFiles: {counter:10d}    Tar files: {tarcount:10d}     ({int(counter/tarcount):10d} files/tar)',end='')
			print(f'\rFiles: {counter:10d}    Tar files: {tarcount:10d}     ({int(counter/tarcount):10d} files/tar)\n')
		finally:
			for subtf in subtfdict.values():
				subtf.close()
