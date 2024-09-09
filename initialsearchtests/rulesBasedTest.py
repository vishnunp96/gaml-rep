import re
import sys

def shortestDistance(s1, e1, s2, e2):
	if s1 < s2:
		return abs(s2-e1)
	else:
		return abs(s1-e2)


keywords = ['number', 'digit', 'numerical','non-existent']

text = "0.1 is a number. This is a file containing lots of digits (numerical) (e.g. 5 or 3). We have the number 2. 3 is numerical 1."


#m = re.findall("[0-9\.]+.+"+keyword+"|"+keyword+".+[0-9\.]+", text, re.DOTALL)
#m = re.findall(keyword+".+[0-9\.]+|[0-9\.]+.+"+keyword, text, re.DOTALL)

number = '(?<!\.)\d+(?:\.\d+)?'
anything = '.*?'

#m = re.findall('(?=('+keyword+anything+number+'|'+number+anything+keyword+'))', text, re.DOTALL)
#m = re.findall(keyword+anything+number+'|'+number+anything+keyword, text, re.DOTALL)

#m = re.finditer('(?=('+keyword+anything+number+'|'+number+anything+keyword+'))', text, re.DOTALL)
#m = re.finditer(keyword+anything+number+'|'+number+anything+keyword, text, re.DOTALL)


numbers = [(match.start(0), match.end(0), match.group(0)) for match in re.finditer(number, text)]

values = {}

for keyword in keywords:
	values[keyword] = []

	for match in re.finditer(keyword, text):
		min = sys.maxsize
		val = None
		for num in numbers:
			dist = shortestDistance(match.start(0), match.end(0), num[0], num[1])
			if dist < min:
				min = dist
				val = num[2]
		values[keyword].append(val)

print(values)

strValues = str(values)

import ast

recoveredValues = ast.literal_eval(strValues)

print("strValues: " + strValues)
print(recoveredValues)

print(recoveredValues['number'])




