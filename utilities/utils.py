import signal

class TimeoutException(Exception):
	pass
def _handler(signum, frame):
	raise TimeoutException()

def timeout(func, args=[], kwargs={}, max_duration=0, default=None):
	''' Call a function with timeout. If max_duration is not given, will wait until function completes. '''

	original_handler = signal.getsignal(signal.SIGALRM)
	signal.signal(signal.SIGALRM, _handler)

	signal.alarm(max_duration)

	try:
		result = func(*args,**kwargs)
		signal.alarm(0)
		return result
	except TimeoutException:
		pass
	finally:
		signal.signal(signal.SIGALRM, original_handler)

	return default

if __name__ == '__main__':

	def test_func(one,two,three=None,four=None):
		import time
		print(one,two,three,four)
		while 1:
			print('sec')
			time.sleep(1)

	print(timeout(test_func,[1,2],kwargs={'three':3}, max_duration=5,default='default'))

