import sys
import nltk
import chardet
import io

def isEmpty(text):
    return not text or text.isspace()

lines = []

#sysIterator = iter(sys.stdin)
raw = sys.stdin.buffer.read() ## Does this slow down the pipe process?
try:
	text = raw.decode("UTF-8") ## errors='replace'?
except UnicodeDecodeError:
	encoding = chardet.detect(raw)['encoding']
	try:
		text = raw.decode(encoding)
	except:
		raise Exception('Problem decoding input.')
sysIterator = io.StringIO(text)


# In case file is empty to begin with...
try:
    nextline = next(sysIterator)
    foundEnd = False
except StopIteration:
    nextline = ''
    foundEnd = True

while not foundEnd:
    try:
        line = nextline
        nextline = next(sysIterator)
        
        if isEmpty(line):
            while isEmpty(nextline):
                nextline = next(sysIterator)
        else: # line is not empty
            while not isEmpty(nextline):
                line = line.strip() + ' ' + nextline.strip()
                nextline = next(sysIterator)

    except StopIteration:
        nextline = ''
        foundEnd = True
    
    lines.append(line.strip())

tokenized_file = []

for l in lines:
    if isEmpty(l):
        tokenized_file.append([])
    else:
        tokenized_file += [nltk.word_tokenize(s) for s in nltk.sent_tokenize(l)]

for t in tokenized_file:
    print(' '.join(t))








