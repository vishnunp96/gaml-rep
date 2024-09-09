import os

def gettype(path):
	if os.path.isfile(path):
		return 'file'
	elif os.path.isdir(path):
		return 'dir'
	elif os.path.islink(path):
		return 'link'
	elif os.path.ismount(path):
		return 'mount'
	else:
		return None
