from sklearn.model_selection import train_test_split

def split_data(data, sizes, random_state=None):
	if isinstance(sizes,(float,int)):
		if isinstance(sizes,float) and not (0 < sizes <= 1): raise ValueError('If \'sizes\' is provided as a float, it must have a value 0<sizes<=1.')
		if isinstance(sizes,float) and sizes == 1.0: return data
		return train_test_split(data, train_size=sizes, random_state=random_state)[0]
	else:
		sizes = list(sizes)
		def generate_subsets():
			remaining = data
			for i in range(len(sizes)):
				frac = sizes[i]/sum(sizes[i:])
				test_size = 1.0-frac
				if len(remaining) == 0:
					yield []
				if test_size == 0:
					yield remaining
					remaining = []
				else:
					subset,remaining = train_test_split(remaining, test_size=test_size, random_state=random_state)
					yield subset
		return tuple(generate_subsets())
