import argparse

import os

from collections import OrderedDict

class DirectoryAction(argparse.Action):
	"""docstring for DirectoryAction"""
	def __init__(self, option_strings, mustexist=True, mkdirs=False, *args, **kwargs):
		self.mustexist=mustexist
		self.mkdirs=mkdirs
		super(DirectoryAction, self).__init__(option_strings,*args, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):
		path = os.path.abspath(values)
		if not os.path.isdir(path):
			if self.mustexist:
				parser.error('Supplied directory (' + path + ') does not exist.')
			elif self.mkdirs:
				os.makedirs(path)
		setattr(namespace, self.dest, path)


class CustomAction(argparse.Action):
	def __init__(self, option_strings, additional_arg1, additional_arg2,*args, **kwargs):
		self._a1 = additional_arg1
		self._a2 = additional_arg2
		super(CustomAction, self).__init__(option_strings=option_strings,*args, **kwargs)
	def __call__(self, parser, namespace, values, option_string=None):
		print(self._a1)
		print(self._a2)
		setattr(namespace, self.dest, values)
		setattr(namespace, self.dest + '1', values + '1')



class ArgumentParser(argparse.ArgumentParser):
	def __init__(self, *args, **kwargs):
		super(ArgumentParser, self).__init__(*args, **kwargs)
		self.defaults=OrderedDict()

	def add_argument(self, *args, **kwargs):
		default = None
		if isinstance(kwargs.get('default'), tuple) and callable(kwargs.get('default')[1]) and isinstance(kwargs.get('default')[0], str):
			default = kwargs.pop('default')
		a = super(ArgumentParser, self).add_argument(*args, **kwargs)
		if default:
			self.defaults[a.dest] = default

	def parse_args(self):
		args = super(ArgumentParser, self).parse_args()
		d = vars(args)
		for key,value in self.defaults.items():
			if not d.get(key):
				d[key] = value[1](d[value[0]])
		return args
		

parser = ArgumentParser()

parser.add_argument("-i","--interesting",default=('indexdirpath',lambda r: os.path.basename(r)+'_hello'),help='Help.')
parser.add_argument("indexdirpath",action=DirectoryAction, mustexist=False,mkdirs=True, help='Path to directory where index will be created.')
parser.add_argument("-n","--indexname",default='arXivMetadataIndex',help='Name of Whoosh directory to be created in indexdirpath.')


args = parser.parse_args()

for key,value in vars(args).items():
	print(key,value)

#print(args.indexname)
#print(args.indexdirpath)

