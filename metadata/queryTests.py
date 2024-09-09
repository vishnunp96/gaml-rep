import whoosh
import whoosh.fields as fields
import whoosh.index as index
import whoosh.qparser as qparser
import whoosh.query as query

import os
import sys
import argparse

import time

from directoryactions import DirectoryAction


parser = argparse.ArgumentParser(description="Some queries for arXiv data stored in a Whoosh index.")
parser.add_argument("indexdirpath",action=DirectoryAction, help='Path to directory where index will be created.')
parser.add_argument("-n","--indexname",default='arXivMetadataIndex',help='Name of Whoosh directory to be created in indexdirpath.')
args = parser.parse_args()



## arxiv_id, date, title, abstract, categories

if not index.exists_in(args.indexdirpath, indexname=args.indexname):
	print(args.indexdirpath + ' does not contain a ' + args.indexname + ' Whoosh index.')
	sys.exit()

ix = index.open_dir(args.indexdirpath,indexname=args.indexname,readonly=True)

print(str(ix.doc_count()) + ' documents in ' + args.indexdirpath + '/' + args.indexname)

with ix.searcher() as searcher:
	query = query.Every() #qparser.QueryParser("categories", ix.schema).parse("astro")
	#query = qparser.QueryParser("arxiv_id", ix.schema).parse('astro-ph/0001001')
	results = searcher.search(query,limit=None)
	categories = []
	for result in results:
		#print(result.fields())
		categories += result.fields()['categories'].split()

categoryset = set(categories)
print(str(len(categoryset)) + ' categories in documents.')
for category in sorted(categoryset):
	print(category)


#with ix.searcher() as searcher:
#	query = qparser.QueryParser("categories", ix.schema).parse("astro-ph")
#	results = searcher.search(query,limit=None)
#	for result in results:
#		print(result.fields()['arxiv_id'] + ': ' + result.fields()['title'].replace('\n',' ')[:50] + '(' + ', '.join(result.fields()['categories'].split()) + ') (' + ', '.join(result.fields()['setSpec'].split()) + ')')





