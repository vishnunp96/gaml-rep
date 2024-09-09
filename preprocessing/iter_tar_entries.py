def iter_tar_entries(tarpath,metadata,category=None):
	'''
	This generator returns iterators of source file directories.
	Each entry in the returned iterators should contain information
	about the file, along with a route to opening it - but the file
	should only be opened when the user requests it.
	'''
	with tarfile.open(tarpath,mode='r') as tf:
		for entry in tf:
			if entry.isfile():
				arXiv = arXivID_from_path(entry.name)
				categories = metadata.get(arXiv,field='categories')
				if category and not categories:
					continue
				if entry.name.endswith('.pdf'):
					## What to do here?
					pass
				if category and category in categories:
					# Extract gzip
					try:
						with tarfile.open(fileobj=tf.extractfile(entry),mode='r:gz') as ntf: # Nested tar file
							for fileinfo in ntf:
								if fileinfo.isfile() and extractre.search(fileinfo.name):
									try:
										with open(destination,'wb') as f:
											f.write(ntf.extractfile(fileinfo).read())
									except FileNotFoundError:
										os.makedirs(os.path.dirname(destination))
										with open(destination,'wb') as f:
											f.write(ntf.extractfile(fileinfo).read())
					except tarfile.ReadError:
						with gzip.open(tf.extractfile(entry),mode='r') as gf:
							gf.name
							with open(os.path.join(directory,os.path.basename(gf.name)),'wb') as f:
								f.write(gf.read())

class ArXivSourceEntry:
	def __init__(self,tarfile,entries):
		self.tarfile = tarfile
		self.entries = entries
	def open(self,entry):
		return tarfile.extractfile(entry)

