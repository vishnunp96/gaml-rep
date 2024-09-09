import multiprocessing.pool as _pool
from .extracollections import makecollection,adaptivedict
from .iterutilities import iterator_slice

def _starhelper(data):
	return data[0](*makecollection(data[1]))

def _process_chunk(func, to_iterate, results_type, *additional_args):
	results = results_type()
	for i in to_iterate:
		func(i, results, *additional_args)
	return results

class Pool(_pool.Pool):
	def istarmap_unordered(self, func, iterable, chunksize=1):
		yield from self.imap_unordered(_starhelper,((func,i) for i in iterable),chunksize)

	def map_all(self, func, iterable, results_type=adaptivedict, additional_args=(), chunksize=1):
		results = results_type()
		for r in self.istarmap_unordered(_process_chunk,((func,i,results_type,*additional_args) for i in iterator_slice(iterable, chunksize))):
			results.append(r)
		return results

	def map_all_batches(self, func, iterable, results_type=adaptivedict, additional_args=(), chunksize=1):
		results = results_type()
		for r in self.istarmap_unordered(func,((i,results_type(),*additional_args) for i in iterator_slice(iterable, chunksize))):
			results.append(r)
		return results

def parallel_results(func, iterable, results_type=adaptivedict, additional_args=(), chunksize=1, processes=1):
	with Pool(processes=processes) as p:
		return p.map_all(func, iterable, results_type, additional_args, chunksize)

def parallel_results_batches(func, iterable, results_type=adaptivedict, additional_args=(), chunksize=1, processes=1):
	with Pool(processes=processes) as p:
		return p.map_all_batches(func, iterable, results_type, additional_args, chunksize)

#if __name__ == '__main__':
#
#	import time
#
#	def testf(arg1, arg2):
#		time.sleep(2)
#		return str(arg1) + ' : ' + str(arg2)
#
#	def test2(arg1):
#		time.sleep(1)
#		return str(arg1)
#
#	with Pool(processes=2) as pool:
#
#		for r in pool.istarmap_unordered(testf, zip(range(10,20),range(20,30))):
#			print(r)
#
#		for r in pool.istarmap_unordered(test2, range(10,20)):
#			print(r)
