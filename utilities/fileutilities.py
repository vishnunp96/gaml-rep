import sys,os
import re
import contextlib

import utilities.pathutilities as pathutilities
from .extracollections import makecollection

@contextlib.contextmanager
def open_default(filename=None, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
	## Answer by: https://stackoverflow.com/a/17603000
	if filename:
		f = open(filename, mode=mode,buffering=buffering,encoding=encoding,errors=errors,newline=newline,closefd=closefd,opener=opener)
	else:
		if mode == 'r':
			f = sys.stdin
		elif mode == 'w':
			f = sys.stdout
		else:
			raise ValueError('Invalid mode for opening system file object.')

	try:
		yield f
	finally:
		if f is not sys.stdout and f is not sys.stdin:
			f.close()


def addsuffix(filepath,suffix, delimiter='_'):
	## https://stackoverflow.com/a/37488031
	parts = filepath.split('.')
	if len(parts) <= 1 or os.sep in parts[-1]:
		return filepath + delimiter + suffix
	else:
		return '.'.join(parts[:-1])+ delimiter + suffix + '.' + parts[-1]

def changeext(filepath,newext):
	delimiter = '' if newext.startswith('.') else '.'
	parts = os.path.splitext(filepath)
	return parts[0] + delimiter + newext

def get_file_base(path):
	return os.path.splitext(os.path.basename(path))[0]

def increment_file_number(filepath):
	parts = list(os.path.splitext(filepath))
	m=re.search(r'(\d+)$',parts[0])
	if m:
		parts[0] = parts[0][0:m.start(1)] + str(int(m.group(1))+1) #+ parts[0][m.end(1):]
	else:
		parts[0] = parts[0] + '1'
	return parts[0] + parts[1]

def get_available_filename(filepath):
	available = filepath
	while os.path.isfile(available):
		available = increment_file_number(available)
	return available

def listdir(dirpath):
	absdir = os.path.abspath(dirpath)
	return [os.path.join(absdir, n) for n in os.listdir(absdir)]

def walk(dirpath):
	absdir = os.path.abspath(dirpath)
	for root, dirnames, filenames in os.walk(absdir):
		for filename in filenames:
			yield os.path.join(root,filename)



def iter_files(path, recursive=False, fileregex=None, contains=None, suffix=None):
	fileregex = []
	fileregex += makecollection(fileregex)
	fileregex += [re.escape(i) for i in makecollection(contains)]
	fileregex += [re.escape(i)+'$' for i in makecollection(suffix)]
	if not fileregex:
		fileregex=[r'.']

	startingpath = os.path.realpath(path)

	pathtype = pathutilities.gettype(startingpath)
	if pathtype == 'file':
		yield startingpath
	elif pathtype == 'dir':
		if recursive:
			yield from (i for i in walk(startingpath) if any([re.search(r,i) for r in fileregex]))
		else:
			yield from (i for i in listdir(startingpath) if os.path.isfile(i) and any([re.search(r,i) for r in fileregex]))
	else:
		raise ValueError('Supplied path ('+path+') cannot be resolved to a file or directory.')
