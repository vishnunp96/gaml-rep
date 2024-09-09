from collections import defaultdict
import collections
from numpy import array
import copy

class missingdict(defaultdict):
	def __missing__(self, key):
		return self.default_factory()

class adaptivedict(dict):
	def __missing__(self, key):
		self[key] = _unknown_type(self,key)
		return self[key]
	def append(self, other):
		for key,value in other.items():
			self[key] += value
class _unknown_type:
	def __init__(self,parent,key):
		self.key = key
		self.parent = parent
	def append(self,other):
		self.parent[self.key] = [other]
	def __add__(self,other):
		return type(other)() + other
	def __iadd__(self,other):
		return type(other)() + other
	def __sub__(self,other):
		return type(other)() - other
	def __isub__(self,other):
		return type(other)() - other
	def __getattr__(self, name):
		def method(other):
			print(type(other))
			self.parent[self.key] = type(other)()
			return getattr(self.parent[self.key],name)(other)
		return method

class bidict(dict):
	## Answer from: https://stackoverflow.com/a/21894086
	def __init__(self, *args, **kwargs):
		super(bidict, self).__init__(*args, **kwargs)
		self.inverse = {}
		for key, value in self.items():
			self.inverse.setdefault(value,[]).append(key)

	def __setitem__(self, key, value):
		if key in self:
			self.inverse[self[key]].remove(key)
		super(bidict, self).__setitem__(key, value)
		self.inverse.setdefault(value,[]).append(key)

	def __delitem__(self, key):
		self.inverse.setdefault(self[key],[]).remove(key)
		if self[key] in self.inverse and not self.inverse[self[key]]:
			del self.inverse[self[key]]
		super(bidict, self).__delitem__(key)


class List2D:
	def __init__(self,data=None):
		self.data = data

	@property
	def shape(self):
		if self.data is not None:
			if len(self)==0:
				return (0,0)
			elif len(set([len(i) for i in self.data]))==1:
				return (len(self.data),len(self.data[0]))
		else:
			return (len(self.data),None)

	def __getitem__(self,x):
		if isinstance(x,tuple): # Provided index is tuple
			if isinstance(x[0],int) and isinstance(x[1],int): # tuple(int, int)
				return self.data[x[0]][x[1]]
			elif isinstance(x[1],int): # tuple(slice,int)
				return List2D([[i[x[1]]] for i in self.data[x[0]]])
			elif isinstance(x[0],int): # tuple(int,slice)
				return List2D([self.data[x[0]][x[1]]])
				#return List2D([[i[x[1]]] for i in self.data[x[0]]])
			else: # tuple(slive,slice)
				return List2D([i[x[1]] for i in self.data[x[0]]])
		elif isinstance(x,slice): # Privided index is a slice
			return List2D(self.data[x])
		elif isinstance(x,int): # Provided index is a single int
			return self.data[x]
		else: # Unknown, try anyway
			return List2D(self.data[x])

	def __setitem__(self,x,val):
		if isinstance(x,tuple): # Provided index is tuple
			if isinstance(x[0],int) and isinstance(x[1],int): # tuple(int, int)
				self.data[x[0]][x[1]] = val
			elif isinstance(x[1],int): # tuple(slice,int)
				raise NotImplementedError()
				#return List2D([[i[x[1]]] for i in self.data[x[0]]])
			else: # tuple(slive,slice)
				raise NotImplementedError()
				#return List2D([i[x[1]] for i in self.data[x[0]]])
		elif isinstance(x,slice): # Privided index is a slice
			raise NotImplementedError()
			#return List2D(self.data[x])
		elif isinstance(x,int): # Provided index is a single int
			raise NotImplementedError()
			#return self.data[x]
		else: # Unknown, try anyway
			raise NotImplementedError()
			#return List2D(self.data[x])

	def __delitem__(self,x):
		if isinstance(x,tuple): # Provided index is tuple
			if isinstance(x[0],int) and isinstance(x[1],int): # tuple(int, int)
				del self.data[x[0]][x[1]]
			elif isinstance(x[0],int):
				del self.data[x[0]][x[1]]
			else: # tuple(slice,slice) or tuple(slice,int) or tuple(int,slice)
				for i in self.data[x[0]]: del i[x[1]]
		elif isinstance(x,(slice,int)): # Privided index is a slice, or int
			del self.data[x]
		else: # Unknown, try anyway
			del self.data[x]

	def __repr__(self):
		if len(self.data)==0: return 'List2D([])'
		if self.shape:
			l = [max([len(repr(i)) for i in col]) for col in self.itercolumns()]
		else:
			#l = [max(len(repr(i)) for i in self)]*max(len(row) for row in self.iterrows())
			l = [1]*max(len(row) for row in self.iterrows())

		return ('List2D([' +
				('\n'+(' '*8)).join(
						[
							'['+', '.join(['{:>{}s}'.format(repr(col),l[i]) for i,col in enumerate(row)])+']' for row in self.iterrows()
						]
					) + '])')

		#s = 'List2D([' + repr(self.data[0])
		#for i in self.data[1:]:
		#	s += '\n' + ' '*8 + repr(i)
		#s += '])'
		#return s
	def nditer(self):
		for i,sublist in enumerate(self.data):
			for j,val in enumerate(sublist):
				yield (i,j),val
	def idxiter(self):
		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				yield (i,j)

	def iterrows(self):
		for i in self.data:
			yield i
	def itercolumns(self):
		for i in range(self.shape[1]):
			yield self[:,i].flatten()

	def flatten(self):
		return [i for i in self]

	def apply(self,func):
		for idx,val in self.nditer():
			self[idx] = func(self[idx])

	def __len__(self):
		return sum(len(i) for i in self.data)

	def __iter__(self):
		for i in self.data:
			for j in i:
				yield j

	def __array__(self):
		return array(self.data)

	def __copy__(self):
		return List2D(copy.copy(self.data))
	def __deepcopy__(self,memo):
		return List2D(copy.deepcopy(self.data,memo))

class SpanList:
	def __init__(self,iterable=None):
		self.iterable = iterable if iterable else []
		self.spans = [1 for i in self.iterable]
	def __repr__(self):
		# Improve
		return 'SpanList([' + ', '.join([(str(s)+' * ' if s else '')+repr(i) for i,s in zip(self.iterable,self.spans)]) + '])'
	def append(self,item,count=1):
		self.iterable.append(item)
		self.spans.append(count)
	def extend(self,iterable):
		self.iterable.extend(iterable)
		self.spans.extend([1 for i in iterable])
	def __delitem__(self,x):
		pass
		# Need to work out how to do this...
		#del self.iterable[x]

def makecollection(*args, concatenate=False, iterate_str=False, type=list):
	if len(args) == 0:
		return type()
	elif len(args) > 1:
		if concatenate:
			result = []
			for i in args:
				if isinstance(i,collections.abc.Iterable) and (iterate_str or not isinstance(i,str)): result += list(i)
				else: result.append(i)
			return type(result)
		else:
			return type(args)
	elif args[0] is None:
		return type()
	elif isinstance(args[0], collections.abc.Iterable) and (iterate_str or not isinstance(args[0],str)):
		return type(args[0])
	else:
		return type([args[0]])

