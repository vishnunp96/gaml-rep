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

start = time.time()
ix = index.open_dir(args.indexdirpath,indexname=args.indexname,readonly=True)
print('Open: {0}'.format(time.time()-start))

start = time.time()
print(str(ix.doc_count()) + ' documents in ' + args.indexdirpath + '/' + args.indexname)
print('Count: {0}'.format(time.time()-start))

start = time.time()
with ix.searcher() as searcher:
	print('Open Searcher: {0}'.format(time.time()-start))

	start = time.time()
	query = qparser.QueryParser("arxiv_id", ix.schema).parse('astro-ph/0001001')
	print('Make query: {0}'.format(time.time()-start))

	start = time.time()
	results = searcher.search(query,limit=1)
	print('Search: {0}'.format(time.time()-start))

	print(results.__class__)

	start = time.time()
	if results[0]:
		#print(results[0].fields()['date'])
		print('Print results: {0}'.format(time.time()-start))


start = time.time()
with ix.searcher() as searcher:
	print('Open Searcher: {0}'.format(time.time()-start))

	start = time.time()
	query = qparser.QueryParser("arxiv_id", ix.schema).parse('astro-ph/0001001')
	print('Make query: {0}'.format(time.time()-start))

	start = time.time()
	results = searcher.search(query,limit=1)
	print('Search: {0}'.format(time.time()-start))

	print(results.__class__)

	start = time.time()
	if results[0]:
		#print(results[0].fields()['date'])
		print('Print results: {0}'.format(time.time()-start))


