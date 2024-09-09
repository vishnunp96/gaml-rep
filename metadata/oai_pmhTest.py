import urllib.request
import xml.etree.ElementTree
import re

def xmlnamespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

def arXivID(identifier):
	m = re.match('[0-9]+\.[0-9]+|[a-zA-Z-]+/[0-9]+', identifier)
	if m:
		return m.group(0)

	m = re.match('[a-zA-Z-]+[0-9]+', identifier)
	if m:
		parts = re.split('(\d+)', identifier)
		return parts[0] + '/' + parts[1]

	return ''

def arXivURL(identifier):
	return "http://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:" + identifier + "&metadataPrefix=arXiv"

def arXivSubject(proposedID):
	identifier = arXivID(proposedID)
	if identifier:
		e = xml.etree.ElementTree.fromstring(urllib.request.urlopen(arXivURL(identifier)).read().decode())
		xmlns = xmlnamespace(e)
		record = e.find(xmlns + 'GetRecord')
		if record:
			return record.find(xmlns+'record').find(xmlns+'header').find(xmlns+'setSpec').text
	return 'ERROR'



print(arXivSubject("nucl-ex/0001004"))
print(arXivSubject("nucl-ex0001004"))
print(arXivSubject("1711.02656"))
print(arXivSubject("5612.02656"))
print(arXivSubject("iwrfigeriu"))











