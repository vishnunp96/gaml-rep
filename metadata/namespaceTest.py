import xml.etree.ElementTree as ElementTree
import re

with open('content.xml') as f:
	content = f.read()



tree = ElementTree.fromstring(content)

ns = set()
for elem in tree.iter():
	ns.update(re.findall('\{.*\}', elem.tag))

nsdict = {}
for idx, val in enumerate(ns):
	nsdict[idx] = val

print(nsdict)
print(nsdict[0].__class__)

a = tree.findall('record',ns)
b = len(a)
c = str(b)
print(c)
