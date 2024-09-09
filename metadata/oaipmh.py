import urllib.request
import xml.etree.ElementTree
import re
import pickle
import argparse
import os


##########
## General Functions
##########
def arXivID(proposedID):
	m = re.match('[0-9]+\.[0-9]+|[a-zA-Z-]+/[0-9]+', proposedID)
	if m:
		return m.group(0)

	m = re.match('[a-zA-Z-]+[0-9]+', proposedID)
	if m:
		parts = re.split('(\d+)', proposedID)
		return parts[0] + '/' + parts[1]

	return '' ## Should we just return the proposed ID, in case it is of a form we are not familiar with?

def arXivID_from_path(path):
	return arXivID(os.path.splitext(os.path.basename(path))[0])

##########
## Online/XML
##########
def xmlnamespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

def arXivURL(identifier):
	return "http://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:" + identifier + "&metadataPrefix=arXiv"

def arXivMetadataOnline(proposedID):
	identifier = arXivID(proposedID)
	if identifier:

		try:
			request = urllib.request.urlopen(arXivURL(identifier))
			content = request.read().decode()

			e = xml.etree.ElementTree.fromstring(content)
			xmlns = xmlnamespace(e)
			record = e.find(xmlns+'GetRecord')

			if record:
				return (xmlns,record)

		except urllib.error.URLError:
			## Retry here?
			print('Problem accessing arXiv for identifier: ' + identifier + ' (returned nothing)')

	return ('',None)

def arXivSubjectOnline(proposedID):
	(xmlns,record) = arXivMetadataOnline(proposedID)
	if record:
		return [el.text for el in record.find(xmlns+'record').find(xmlns+'header').findall(xmlns+'setSpec')]
	return []


##########
## Offline
##########
class ArXivMetadata:
	def __init__(self, path):
		self.path = path
		with open(path, 'rb') as f:
			self.data = pickle.load(f) # The protocol version used is detected automatically
		if type(self.data) is not dict:
			raise ValueError('Pickled object of wrong type (' + str(type(self.data)) + '!=dict) in: ' + path)

	def get(self, proposedID, field=None):
		identifier = arXivID(proposedID)
		if identifier:
			meta = self.data.get(identifier) # Returns None if key is not in dict
			if meta:
				if field:
					return meta[field]
				else:
					return meta
		#print('Could not find ' + str(field) + ' for ' + proposedID + '.')
		return None

	def __getitem__(self, identifier):
		return self.data.get(identifier) # Returns None if key is not in dict

	def getID(self,proposedID):
		identifier = arXivID(proposedID)
		if identifier:
			return identifier if self[identifier] else None
		else:
			return None

	def entries(self):
		return self.data.values()


class MetadataAction(argparse.Action):
	"""docstring for MetadataAction"""
	def __call__(self, parser, namespace, values, option_string=None):

		path = os.path.abspath(values)

		if os.path.isfile(path):
			metadata = ArXivMetadata(path)
		else:
			parser.error('Supplied metadata file ('+path+') does not exist.')

		setattr(namespace, self.dest, metadata)


##########
## Testing
##########
if __name__ == "__main__":

	from utilities.argparseactions import FileAction,RegexListAction

	parser = argparse.ArgumentParser(description="Module containing methods and classes for arXiv metadata handling.")
	parser.add_argument('metadatapath',action=FileAction, mustexist=True, help='Path to metadata.')
	parser.add_argument('subjects',action=RegexListAction, help='Comma-separated list of subjects to include.')
	args = parser.parse_args()

	metadata = ArXivMetadata(args.metadatapath)

	ids = ["nucl-ex/0001004", "nucl-ex0001004", "1711.02656", "5612.02656", "iwrfigeriu", 'hep-ph0001090']

	#print(metadata.get("nucl-ex/0001004",'categories'))
	#print(metadata.get("nucl-ex0001004",'categories'))
	#print(metadata.get("1711.02656",'categories'))
	#print(metadata.get("5612.02656",'categories'))
	#print(metadata.get("iwrfigeriu",'categories'))
	#print(metadata.get('hep-ph0001090','categories'))

	for id in ids:
		categories = metadata.get(id,field='categories')

		if categories is not None:
			smatch = re.search(args.subjects, categories) ## would find() be quicker?
			print(id + ': ' + categories + ' -> ' + args.subjects + ' : ' + str(bool(smatch)))
		else:
			print(id + ' could not be checked.')

