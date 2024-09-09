import os,argparse
from utilities.fileutilities import walk
from utilities.jsonutils import load_json,dump_json

def make_manifest(dirpath,indexer,extensions=tuple(),errors=None,name='manifest.json'):

	if not os.path.isdir(dirpath):
		raise ValueError('Supplied path is not a directory:',dirpath)
	if os.sep in name:
		raise ValueError('Supplied manifest name ('+name+') invalid. Must not be a path.')

	fileiterator = (path for path in walk(dirpath) if (path.endswith(extensions) and extensions if extensions else True))

	manifest = {}

	for absfilepath in fileiterator:
		index = indexer(absfilepath)
		if index:
			manifest[index] = os.path.relpath(absfilepath, start=dirpath)
		elif errors != 'ignore':
			raise ValueError('Index could not be found for ' + absfilepath)

	#with open(os.path.join(dirpath,name), 'wb') as f:
	#	pickle.dump(manifest, f)
	dump_json(manifest,os.path.join(dirpath,name))

	return manifest

class Manifest:
	def __init__(self,dirpath,manifestpath=None):
		self.dirpath = os.path.abspath(dirpath)

		if manifestpath: ## Custom archive has been provided
			#with open(manifestpath, 'rb') as f:
			#	self.manifest = pickle.load(f)
			self.manifest = load_json(manifestpath)
		else: ## Load manifest from
			#with open(os.path.join(dirpath,'manifest.pkl'), 'rb') as f:
			#	self.manifest = pickle.load(f)
			self.manifest = load_json(os.path.join(dirpath,'manifest.json'))

	def __getitem__(self, identifier):
		relpath = self.manifest.get(identifier) # Returns None if key is not in dict
		if relpath:
			return os.path.join(self.dirpath,relpath)
		else:
			return None

class ManifestAction(argparse.Action):
	"""docstring for ManifestAction"""
	def __call__(self, parser, namespace, values, option_string=None):

		path = os.path.abspath(values)

		if os.path.isdir(path):
			manifest = Manifest(path)
		else:
			parser.error(f'Supplied directory does not contain a manifest: {values}')

		setattr(namespace, self.dest, manifest)


if __name__ == "__main__":

	from utilities.argparseactions import ArgumentParser,DirectoryAction
	from metadata import MetadataAction

	from utilities import StopWatch
	stopwatch = StopWatch()

	parser = ArgumentParser(description="Construct arXiv file manifest.")
	parser.add_argument('archivepath',action=DirectoryAction, mustexist=True, help='Path to source directory.')
	parser.add_argument('metadata',action=MetadataAction,help='Path to arXiv metadata.')
	parser.add_argument('extension',help='File extension for which to make manifest.')
	args = parser.parse_args()

	make_manifest(args.archivepath,lambda s: args.metadata.getID(os.path.splitext(os.path.basename(s))[0]),extensions=args.extension)

	print('Finished processing ',os.path.basename(args.archivepath))
	stopwatch.report()
