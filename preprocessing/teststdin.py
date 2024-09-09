import sys
import chardet
import io

raw = sys.stdin.buffer.read()
try:
	text = raw.decode("UTF-8")
	print('UTF-8')
except UnicodeDecodeError:
	encoding = chardet.detect(raw)['encoding']
	text = raw.decode(encoding)
	print(encoding)
sysIterator = io.StringIO(text)

def isEmpty(text):
    return not text or text.isspace()

lines = []




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

print(lines)
