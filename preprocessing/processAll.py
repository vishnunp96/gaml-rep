import re
import os
import tarfile
import utilities.mygzip as gzip
from metadata.oaipmh import arXivID_from_path
import tempfile
from utilities.fileutilities import listdir
import utilities.filehandler as filehandler
from utilities.utils import timeout
import regex
import preprocessing.latexmlpy as latexml

def make_ext_regex(exts):
	return re.compile('(' + '|'.join(['('+re.escape(ext)+')' for ext in exts]) + ')'+'$',re.IGNORECASE)

#extractre = re.compile('(('+re.escape('.tex')+')|('+re.escape('.latex')+')|('+re.escape('.ltx')+'))'+'$',re.IGNORECASE)
texextensions = ['.tex','.latex','.ltx']
texre = make_ext_regex(texextensions)
extractre = make_ext_regex(texextensions+['.sty','.cls','.bib'])

def process_tar(tarpath,results,metadata,targetpath,category):

	print(f'Start: {os.path.basename(tarpath)}',flush=True)
	filecounter = 0
	failurecounter = 0
	with tarfile.open(tarpath,mode='r') as tf:
		for entry in tf:
			try:
				if entry.isfile():
					arXiv = arXivID_from_path(entry.name)
					categories = metadata.get(arXiv,field='categories')
					if not categories:
						print(f'Could not find categories for {arXiv}, returned \'{categories}\' ({entry.name} in {tarpath})')
						continue
					if not entry.name.endswith('.pdf') and category in categories:
						# Extract gzip
						with tempfile.TemporaryDirectory() as directory:
							try:
								with tarfile.open(fileobj=tf.extractfile(entry),mode='r:gz') as ntf:
									#readFile(ntf.extractfile(nentry))
									for fileinfo in ntf:
										if fileinfo.isfile() and extractre.search(fileinfo.name):
											## Extract file to tempdir
											## Need to deal with subdirectories here
											destination = os.path.join(directory,fileinfo.name)
											try:
												with open(destination,'wb') as f:
													f.write(ntf.extractfile(fileinfo).read())
											except FileNotFoundError:
												os.makedirs(os.path.dirname(destination))
												with open(destination,'wb') as f:
													f.write(ntf.extractfile(fileinfo).read())
							except tarfile.ReadError:
								with gzip.open(tf.extractfile(entry),mode='r') as gf:
									#readFile(gf)
									## Extract file to tempdir
									with open(os.path.join(directory,os.path.basename(gf.name)),'wb') as f:
										f.write(gf.read())

							element = get_xml(directory,arXiv,metadata)
							if element is not None:
								targetfilepath = os.path.join(os.path.splitext(tarpath.replace(os.path.dirname(tarpath),targetpath))[0],os.path.splitext(entry.name)[0])+'.xml'
								destination = os.path.dirname(targetfilepath)

								if not os.path.exists(destination):
									os.makedirs(destination)

								latexml.savexml(element, targetfilepath)
								filecounter += 1
							else:
								print(f'Could not process {arXiv}')
								failurecounter += 1
			except (KeyboardInterrupt, SystemExit):
				raise
			except Exception as e:
				print(f'Failure for {entry.name} in {tarpath}: {e}')
				failurecounter += 1

	print(f'End: {os.path.basename(tarpath)} ({filecounter} files, {failurecounter} failures)',flush=True)
	results['files'] += filecounter
	results['failures'] += failurecounter


def article_score(filepath, identifier,metadata):

	#print('Begin reading file.',flush=True)
	filetext = filehandler.readFile(filepath, fallback_errors='replace') ## This should attempt a proper decoding, or just replace with '?' if one cannot be found
	#print('Read file.',flush=True)

	scores = 0
	if re.search('^[^\%\n]*\\\\begin{document}', filetext, flags=re.MULTILINE): scores += 1
	if re.search('abstract', filetext, flags=re.MULTILINE|re.IGNORECASE): scores += 1

	#print('Begin title search.', flush=True)
	try:
		if timeout(regex.search,['(?e)(?:'+regex.escape(metadata.get(identifier,field='title'))+'){e<=10}', filetext],max_duration=10,default=False): scores += 1
	except SystemError:
		pass
	#print('End title search.',flush=True)

	return scores

def get_xml(dirpath,arXivID,metadata):
	texfiles = [f for f in listdir(dirpath) if texre.search(f)]
	if len(texfiles) < 1:
		print(f'No tex files found for {arXivID}')
		return None
	elif len(texfiles) > 1:
		texfilepath = max(texfiles, key = lambda p: article_score(p,arXivID,metadata))
		print(f'More than one tex file for {arXivID}: {[os.path.basename(f) for f in texfiles]} -> Choose {os.path.basename(texfilepath)}')
	else:
		texfilepath = texfiles[0]

	return latexml.opentex(texfilepath,timeout=180) # 3 minute timeout


if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	import sys
	print(' '.join(sys.argv))

	from utilities.argparseactions import ArgumentParser,IterFilesAction,DirectoryAction
	from metadata import MetadataAction
	from preprocessing.manifest import make_manifest
	from utilities.parallel import parallel_results
	from pprint import pprint

	import random

	parser = ArgumentParser()
	parser.add_argument('source',action=IterFilesAction, suffix='.tar',help='Source directory containing tar files to process (non-recursive search).')
	parser.add_argument('metadata',action=MetadataAction,help='Stored arXiv metdata (pickled file).')
	parser.add_argument('target',action=DirectoryAction,mustexist=False,mkdirs=True,help='Target directory in which to store output XML files.')
	parser.add_argument('category',help='Category to extract from arXiv (e.g. \'astro-ph\')')
	parser.add_argument('-p','--processes',type=int,default=1,help='Number of processes to use. Make this as high as you can.')
	parser.add_argument('-c','--chunksize',type=int,default=1,help='Chunk size. This is best left small (~1-10), as there can be large discrepancies in the sizes of the tar files.')
	args = parser.parse_args()

	## Shuffle the sources files to better distribute
	def shuffle(it):
		l = list(it)
		random.shuffle(l)
		return l

	results = parallel_results(process_tar, shuffle(args.source), additional_args=(args.metadata,args.target,args.category), chunksize=args.chunksize, processes=args.processes)

	pprint(results)

	make_manifest(args.target,lambda s: arXivID_from_path(s),extensions='.xml',errors='ignore')

	stopwatch.report()
