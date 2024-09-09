import urllib.request
import xml.etree.ElementTree as ElementTree
import re
import time


##### Look at: https://github.com/dfm/data.arxiv.io/blob/master/scrape.py


def xmlnamespace(element):
	m = re.match('\{.*\}', element.tag)
	return m.group(0) if m else ''


try:
	with urllib.request.urlopen('http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv') as request:
		content = request.read().decode()
	print('Got metadata.')

	e = ElementTree.fromstring(content)
	xmlns = xmlnamespace(e)

	listRecords = e.find(xmlns+'ListRecords')
	print('ListRecords length = ' + str(len(listRecords)))

	records = listRecords.findall(xmlns+'record')
	print('# records = ' + str(len(records)))
	print(records[0].find('setSpec').text)

	resumptionToken = listRecords.find(xmlns+'resumptionToken')

	print(ElementTree.tostring(resumptionToken, encoding='unicode', method='xml'))
	print('resumptionToken text = ' + resumptionToken.text)

#	time.sleep(10)
#
#	secondurl = 'http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=' + resumptionToken.text
#	print(secondurl)
#	with urllib.request.urlopen(secondurl) as request:
#		content2 = request.read().decode()
#	print('Got second batch of metadata.')
#	e2 = ElementTree.fromstring(content2)
#	xmlns2 = xmlnamespace(e2)
#	listRecords2 = e2.find(xmlns2+'ListRecords')
#	print('Second ListRecords length = ' + str(len(listRecords2)))

except urllib.error.HTTPError as error:
	data = error.read()
	print(data)



## http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv

## http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=760571|10001




