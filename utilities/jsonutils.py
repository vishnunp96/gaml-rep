import json

def load_json(filename):
	with open(filename,'r') as f:
		return json.load(f)

def dump_json(obj, filename, *args, **kwargs):
	with open(filename, 'w') as f:
		json.dump(obj,f,*args, **kwargs)
