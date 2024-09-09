import chardet

filename = 'badEncodingExample.tex'

with open(filename, 'rb') as filerb:
	rawdata = filerb.read()

guess = chardet.detect(rawdata)



with open(filename, encoding=guess['encoding']) as file:
	text = file.read()


print(text)
