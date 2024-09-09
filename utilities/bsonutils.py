import bson

def load_bson(filename):
	with open(filename,'rb') as f:
		return bson.loads(f.read())

def dump_bson(obj, filename):
	with open(filename, 'wb') as f:
		f.write(bson.dumps(obj))
