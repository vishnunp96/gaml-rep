import sys
import re
import itertools

print(sys.argv)

def arange(start,stop,step):
	val = start
	i = 0
	if step > 0:
		assert start <= stop
		while val < stop:
			yield val
			i += 1
			val = start + i*step
	elif step < 0:
		assert start >= stop
		while val > stop:
			yield val
			i += 1
			val = start + i*step
	else:
		yield start

def get_parts(part):
	m = re.match(r'\[(?P<list>[^\]]+)\]',part)
	if m:
		text = m.group('list')
		m = re.match(r'(?P<start>[0-9]+)\.\.(?P<end>[0-9]+)',text)
		if m:
			return [str(i) for i in range(int(m.group('start')),int(m.group('end'))+1)]
		m = re.match(r'(?P<start>[\-\+]?[0-9]+(\.[0-9]*)?)\:(?P<stop>[\-\+]?[0-9]+(\.[0-9]*)?)\:(?P<step>[\-\+]?[0-9]+(\.[0-9]*)?)',text)
		if m:
			try:
				start = m.group('start')
				stop = m.group('stop')
				step = m.group('step')
				float(start),float(stop),float(step) # Check they can all be parsed
				dp = max(len(i.split('.')[-1]) for i in (start,step))
				return [('{:.'+str(dp)+'f}').format(i) for i in arange(float(start),float(stop),float(step))]
			except ValueError:
				pass
		if ',' in text:
			return text.split(',')
	return [part]

args = [get_parts(i) for i in sys.argv[1:]]

print(args)

for i in itertools.product(*args):
	print(' '.join(i))
