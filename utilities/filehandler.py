import chardet
from .stringhandler import decode

def getEncoding(filepath):
	## This can be improved for efficiency: https://chardet.readthedocs.io/en/latest/usage.html#advanced-usage
	with open(filepath, 'rb') as filerb:
		rawdata = filerb.read()
	return chardet.detect(rawdata)['encoding']


def readFile(file,suggested_encoding='utf-8', return_encoding=False, fallback_errors=None):
	try:
		b = file.read()
	except AttributeError:
		with open(file,'rb') as f:
			b = f.read()
	return decode(b, suggested_encoding=suggested_encoding,return_encoding=return_encoding,fallback_errors=fallback_errors)

