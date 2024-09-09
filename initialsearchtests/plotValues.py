import os,sys
import matplotlib.pyplot as plt
import numpy as np
import json

import argparse
from utilities.argparseactions import FileAction
from utilities.dateutilities import getdatetime

parser = argparse.ArgumentParser(description="Plot histogram of values in file.")
parser.add_argument('sourcepath',action=FileAction, mustexist=True, help='Values to plot.')
args = parser.parse_args()

with open(args.sourcepath, encoding='UTF-8') as f:
	valuesDict = json.load(f)

#valuesDict = ast.literal_eval(text)

keywords = []
values = []
for keyword, entrylist in valuesDict.items():
	keywords.append(keyword)
	for entry in entrylist:
		values.append((getdatetime(entry['date']), float(entry['value'])))

#print(sorted(values))

mean = np.mean([f for d,f in values])
std = np.std([f for d,f in values])

print('Mean = ' + str(mean) + ', std.dev. = ' + str(std) + ', ' + str(len(values)) + ' values.')

#x = [f for f in values]
x = [(d,f) for d,f in values if f < 150 and f > -50]
#x = [np.log10(f) for f in values if f > 0]
#x = [f for f in values if abs((f - mean)/std) < 10]

plt.hist(np.array([f for d,f in x]), 500)
plt.title(','.join(keywords))
plt.show()

plt.scatter([d for d,f in x],[f for d,f in x],alpha=0.5)
plt.show()

#with open('bigvalues.txt','wb') as f:
#	for entrylist in valuesDict.values():
#		for entry in entrylist:
#			if float(entry['value']) > 2500:
#				f.write((entry['mention'] + '\n').encode('ascii',errors='ignore'))
