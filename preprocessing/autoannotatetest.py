import re

from utilities.argparseactions import ArgumentParser,RegexListAction,FileAction
from utilities.filehandler import readFile

parser = ArgumentParser()
parser.add_argument("source", action=FileAction)
parser.add_argument("keywords", action=RegexListAction)
args = parser.parse_args()

numberre = re.compile('(?<![\^_{\.])(?:-?\d+(?:\.\d+)?){1,2}(?:\ [0]+)*') # This regex includes ranges

text = readFile(args.source)

n = 1
for m in numberre.finditer(text):
	print(m)
	print('\t',text[m.start()-10:m.end()+10].replace('\n',' '))
	print(f'T{n}\tCentralValue {m.start()} {m.end()}\t{m.group()}')
	n+=1
