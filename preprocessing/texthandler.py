import nltk
import re

def isEmpty(text):
    return not text or text.isspace()

def normalise(text):
	## It is unnecessary to preserve paragraphs, as we will only ever be given text
	## from a single element, which should not contain paragraph breaks, as these
	## should be dealt with by the actual <p> paragraph XML structures.
	return '\n'.join(' '.join(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(text))

def normalise_preserve_paragraphs(text):

	paragraphs = re.split(r'\n{2,}', text)

	return '\n\n'.join('\n'.join(' '.join(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(p)) for p in paragraphs)

def normalise_old(text):

	iterator = iter(text.splitlines())

	lines = []

	# In case file is empty to begin with...
	try:
		nextline = next(iterator)
		foundEnd = False
	except StopIteration:
		nextline = ''
		foundEnd = True

	while not foundEnd:
		try:
			line = nextline
			nextline = next(iterator)

			if isEmpty(line):
				while isEmpty(nextline):
					nextline = next(iterator)
			else: # line is not empty
				while not isEmpty(nextline):
					line = line.strip() + ' ' + nextline.strip()
					nextline = next(iterator)

		except StopIteration:
			nextline = ''
			foundEnd = True

		lines.append(line.strip())

	tokenized = []

	for l in lines:
		if isEmpty(l):
			tokenized.append([])
		else:
			tokenized += [nltk.word_tokenize(s) for s in nltk.sent_tokenize(l)]

	return '\n'.join([' '.join(t) for t in tokenized])
