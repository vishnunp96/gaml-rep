import chardet

def getEncoding(b):
	## This can be improved for efficiency: https://chardet.readthedocs.io/en/latest/usage.html#advanced-usage
	return chardet.detect(b)['encoding']

def decode(b, suggested_encoding='utf-8', return_encoding=False, fallback_errors=None):

	## There must be a better way to deal with 'errors' here
	## Perhaps we Could condition the response based on whether an errors argument is specified
	## Or provide a suggested_encoding argument such that it is tried first

	try:
		result = b.decode(encoding=suggested_encoding)
		if return_encoding:
			return result,suggested_encoding
		return result
	except UnicodeDecodeError:
		encoding=getEncoding(b)
		try:
			result = b.decode(encoding=encoding)
			if return_encoding:
				return result,encoding
			return result
		except (KeyboardInterrupt, SystemExit):
			raise
		except:
			if fallback_errors=='raise':
				raise
			elif fallback_errors:
				if not encoding: encoding = suggested_encoding
				try:
					result = b.decode(encoding=encoding, errors=fallback_errors)
					if return_encoding:
						return result,encoding
					return result
				except (KeyboardInterrupt, SystemExit):
					raise
				except:
					pass

	## If all else fails
	if return_encoding:
		return None,None
	return None
