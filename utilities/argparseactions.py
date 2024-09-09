import argparse
import os
import re
from collections import OrderedDict

import utilities.pathutilities as pathutilities
from .fileutilities import get_available_filename,iter_files

class ArgumentParser(argparse.ArgumentParser):
	def __init__(self, *args, **kwargs):
		super(ArgumentParser, self).__init__(*args, **kwargs)
		self.defaults=OrderedDict()

	def add_argument(self, *args, **kwargs):
		default = None
		if isinstance(kwargs.get('default'), tuple) and len(kwargs.get('default'))==2 and callable(kwargs.get('default')[1]) and isinstance(kwargs.get('default')[0], str):
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


class DirectoryAction(argparse.Action):
	"""docstring for DirectoryAction"""
	def __init__(self, option_strings, mustexist=True, mkdirs=False, *args, **kwargs):
		self.mustexist=mustexist
		self.mkdirs=mkdirs
		super(DirectoryAction, self).__init__(option_strings,*args, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):

		path = os.path.abspath(values)

		exists = os.path.isdir(path)
		if not exists:
			if self.mustexist:
				parser.error('Supplied directory (' + path + ') does not exist.')
			elif self.mkdirs:
				os.makedirs(path)
				exists = True

		setattr(namespace, self.dest + '_exists', exists)
		setattr(namespace, self.dest, path)




class FileAction(argparse.Action):
	"""docstring for FileAction"""
	def __init__(self, option_strings, mustexist=False, findavailable=False, default=None, nargs=None, const=None, *args, **kwargs):
		self.mustexist=mustexist
		self.findavailable=findavailable

		if findavailable and default is not None:
			default = get_available_filename(default)

		## Should this really be here? It is maybe outside the remit of this action?
		if nargs == argparse.OPTIONAL:
			if default is None:
				default = False
			if const is None:
				const = True

		super(FileAction, self).__init__(option_strings,*args,default=default,nargs=nargs,const=const,**kwargs)

	def __call__(self, parser, namespace, values, option_string=None):

		if values != self.const:
			path = os.path.abspath(values)

			exists = os.path.isfile(path)
			if not exists and self.mustexist:
				parser.error('Supplied file (' + path + ') does not exist.')
			elif exists and self.findavailable:
				path = get_available_filename(path)
				exists = False
			given = True
		else:
			## This the right strategy?
			path = self.const
			exists = None
			given = False

		setattr(namespace, self.dest + '_given', given)
		setattr(namespace, self.dest + '_exists', exists)
		setattr(namespace, self.dest, path)


class PathAction(argparse.Action):
	"""docstring for PathAction"""
	PATH_TYPES = ['file','dir','link','mount']
	def __init__(self, option_strings, mustexist=False, allowed=[], *args, **kwargs):
		self.mustexist=mustexist

		if not set(allowed).issubset(PathAction.PATH_TYPES):
			raise ValueError(f'Supplied options for filetype are not supported. Must be subset of: {", ".join(PathAction.PATH_TYPES)}.')

		self.allowed = allowed
		super(PathAction, self).__init__(option_strings,*args, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):

		path = os.path.abspath(values)

		pathtype = pathutilities.gettype(path)
		if not pathtype and self.mustexist:
			parser.error(f'Supplied path ({path}) cannot be resolved to an existing object.')

		if self.allowed and pathtype not in self.allowed:
			parser.error(f'Supplied path is not of correct type. Allowed types are: {", ".join(self.allowed)}.')

		setattr(namespace, self.dest + '_exists', bool(pathtype))
		setattr(namespace, self.dest + '_type', pathtype)
		setattr(namespace, self.dest, path)



class IterFilesAction(argparse.Action):
	"""docstring for IterFilesAction"""
	def __init__(self, option_strings, recursive=False, fileregex=None, contains=None, suffix=None, *args, **kwargs):
		self.recursive = recursive

		self.fileregex = fileregex
		self.contains = contains
		self.suffix = suffix

		super(IterFilesAction, self).__init__(option_strings,*args, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):
		setattr(
				namespace,
				self.dest,
				iter_files(
					values,
					recursive=self.recursive,
					fileregex=self.fileregex,
					contains=self.contains,
					suffix=self.suffix
					)
				)
		setattr(namespace, self.dest + '_path', values)


class RegexListAction(argparse.Action):
	"""docstring for RegexListAction"""
	def __init__(self, option_strings, separator=',', *args, **kwargs):
		self.separator=separator
		super(RegexListAction, self).__init__(option_strings,*args, **kwargs)
	def __call__(self, parser, namespace, values, option_string=None):
		regex = '(?:' + '|'.join([re.escape(s) for s in values.split(self.separator)]) + ')'
		strings = list(values.split(self.separator))
		setattr(namespace, self.dest, regex)
		setattr(namespace, self.dest + '_strings', strings)


class ListAction(argparse.Action):
	"""docstring for ListAction"""
	def __init__(self, option_strings, type=str, separator=',', *args, **kwargs):
		super(ListAction, self).__init__(option_strings,*args, **kwargs)
		self.separator=separator
		self.type = type
	def __call__(self, parser, namespace, values, option_string=None):
		values_list = [self.type(s.strip()) for s in values.split(self.separator)]
		setattr(namespace, self.dest, values_list)


