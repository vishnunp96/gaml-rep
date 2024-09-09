import numpy

from gaml.utilities.stringhandler import decode

class WordEmbeddings:
	'''
	By default, 0th index is unassigned, and the -1th index
	is used for the "Unknown" token.
	'''
	def __init_old__(self,filepath):
		embeddings = {}

		with open(filepath,'rb') as f:
			lines = iter(decode(l,fallback_errors='raise') for l in f)

			firstline = next(lines)

			self.wordcount, self.dim = (int(i) for i in firstline.rstrip().split(' '))

			for line in lines:
				parts = line.rstrip().split(' ')
				embeddings[parts[0]] = numpy.asfarray(numpy.array(parts[1:]),float)

		dict.__init__(self, (embeddings))


	def __init__(self, indexes, embeddings, fallback_index=None):
		self.indexes = indexes
		self.values = embeddings

		self.vocab_size,self.dim = embeddings.shape

		if fallback_index is not None:
			self.fallback_index = fallback_index
		else:
			self.fallback_index = self.values.shape[0]-1

	def __getitem__(self,word):
		return self.values[self.getindex(word)]

	def getindex(self,word):
		return self.indexes.get(word,self.fallback_index)

	def tokenise(self,text):
		## This probably needs improving (NEWLINE tokens, etc.)
		return [self.getindex(t) for t in text.split()]

	@property
	def padding(self):
		return self.values[0]
	padding_index = 0

	def open(filepath,unknown=50,padding_vector=None):
		''' File should be ordered in decreasing order of token frequency. '''

		if unknown<1: raise ValueError('"unknown" must be a positive integer greater than 0.')

		embeddings_dict = dict()
		indexes = dict()
		current_index = 1

		vocab_size = None
		embedding_dim = None

		with open(filepath,'rb') as f:
			lines = iter(decode(l,fallback_errors='raise') for l in f)

			firstline = next(lines)
			try:
				vocab_size,embedding_dim = (int(i) for i in firstline.split())
			except ValueError:
				values = firstline.split()
				word = values[0]
				vector = numpy.asarray(values[1:], dtype=numpy.float32)
				embeddings_dict[word] = vector
				indexes[word] = current_index
				current_index += 1

			for line in lines:
				values = line.split()
				word = values[0]
				vector = numpy.asarray(values[1:], dtype=numpy.float32)
				embeddings_dict[word] = vector
				indexes[word] = current_index
				current_index += 1

		if vocab_size is None:
			vocab_size = len(indexes) + 2
		else:
			vocab_size += 2
		# +1 for the unassigned vector at 0th index
		# and +1 for the "Unknown" token vector at -1th index

		if embedding_dim is None: embedding_dim = len(next(iter(embeddings_dict.values())))

		embeddings = numpy.zeros((vocab_size, embedding_dim))

		for word,index in indexes.items():
			embeddings[index] = embeddings_dict[word]

		# Set "Unknown" token at -1th index (average of least common embeddings)
		indexes[None] = embeddings.shape[0]-1
		embeddings[-1] = embeddings[-(unknown+2):-2].mean(axis=0)

		if padding_vector is None or padding_vector.lower() == 'zeros':
			pass
			#embeddings[0] = 0 # Is this actually necessary?
		elif padding_vector == 'random':
			embeddings[0] = numpy.random.randn(1,embedding_dim)
		elif isinstance(padding_vector, numpy.ndarray):
			embeddings[0] = padding_vector
		else:
			raise ValueError(f"padding_vector must be None, 'zeroes', 'random', or an instance of numpy.ndarray.")

		return WordEmbeddings(indexes, embeddings, fallback_index=embeddings.shape[0]-1)

if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch()

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	parser = ArgumentParser(description='Test word2vec utilities.')
	parser.add_argument('embeddings',action=FileAction,mustexist=True,help='Embeddings file.')
	args = parser.parse_args()

	embeddings = WordEmbeddings.open(args.embeddings)

	print(embeddings.vocab_size)
	print(embeddings.dim)

	print(repr(embeddings['the']))
	print(embeddings['the'].shape)
	print(repr(embeddings['wsiubwiuvbiuvb']))

	stopwatch.report()
