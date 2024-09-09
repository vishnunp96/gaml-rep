import itertools
import random

## Answer by zwer: https://stackoverflow.com/a/44502827
def iterator_slice(iterator, length,type=list):
    iterator = iter(iterator)
    while True:
        res = type(itertools.islice(iterator, length))
        if not res:
            break
        yield res

def randomly(seq):
	#shuffled = list(seq)
	#random.shuffle(shuffled)
	#return iter(shuffled)
	length = len(seq)
	for i in random.sample(range(length),length):
		yield seq[i]
