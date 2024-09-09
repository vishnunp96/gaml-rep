import re
import shutil
import sys

def eprint(*args,**kwargs):
	print(*args, file=sys.stderr, **kwargs)

def printacross(s,maxwidth=sys.maxsize,begin=''):
	screen_width = shutil.get_terminal_size(fallback=(60,10)).columns
	print((begin + s*screen_width)[:min(screen_width,maxwidth)])

def printwrapped(text, width=70):
	screen_width = shutil.get_terminal_size(fallback=(60,10)).columns
	paragraphs = [p for p in re.split('\n{2,}',text) if not p.isspace()]
	for para in paragraphs[:-1]:
		for line in wrapping(para,min(width, screen_width)):
			print(line)
		print()
	for line in wrapping(paragraphs[-1],min(width, screen_width)):
		print(line)

def wrapping(text, width):

	#print('Begin wrapping text of length',len(text),'to width',width)

	if len(text) == 0: return

	position = 0
	counter = 0
	line = ''
	lastbreaklength = 0
	lastbreakcounter = 0
	while True:
		c = text[position]
		position += 1

		if c == ' ' or c == '\n':
			line += ' '
			counter += 1
			lastbreaklength = len(line)
			lastbreakcounter = 0
		elif c == '\x1b':
			while True:
				line += c
				if c == 'm': break
				c = text[position]
				position += 1
		else:
			line += c
			counter += 1
			lastbreakcounter += 1
			if c == '-':
				lastbreaklength = len(line)
				lastbreakcounter = 0

		if counter == width:
			if lastbreaklength == 0: lastbreaklength = len(line)
			yield line[:lastbreaklength]
			line = line[lastbreaklength:]
			counter = lastbreakcounter
			lastbreaklength = 0
			lastbreakcounter = 0

		if position == len(text):
			if len(line) > 0:
				yield line
			break

def color_text(text, colorcode, regex=None, flags=0, spans=None):
	if regex:
		return re.sub(regex,'\x1b['+colorcode+'m\g<0>\x1b[0m',text,flags=flags)
	elif spans:
		spans = sorted(spans, key=lambda s: s[0])
		#sorted_spans = [(0,0)] + sorted_spans + [(len(text),len(text))]
		spans = spans + [(len(text),len(text))]
		result = text[0:spans[0][0]]
		for i in range(len(spans)-1):
			result += '\x1b['+colorcode+'m' + text[spans[i][0]:spans[i][1]] + '\x1b[0m' + text[spans[i][1]:spans[i+1][0]]
		return result
	else:
		return text

class ColorCode:
	red = '0;31;40'
	blue = '1;34;40'
	green = '0;32;40'
	grey = '0;30;40'
	purple = '1;35;40'
