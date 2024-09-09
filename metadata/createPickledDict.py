import whoosh.index as index
import whoosh.query as query

import sys
import argparse
import pickle


from utilities.directoryactions import DirectoryAction

parser = argparse.ArgumentParser(description="Some queries for arXiv data stored in a Whoosh index.")
parser.add_argument("indexdirpath",action=DirectoryAction, mustexist=True, help='Path to directory where metadata index is stored.')
parser.add_argument("picklefilepath",help='Path of file to which pickled metadata will be written.')
parser.add_argument("-n","--indexname",default='arXivMetadataIndex',help='Name of metadata Whoosh index in indexdirpath.')
args = parser.parse_args()

## arxiv_id, date, title, abstract, categories

if not index.exists_in(args.indexdirpath, indexname=args.indexname):
	print(args.indexdirpath + ' does not contain a ' + args.indexname + ' Whoosh index.')
	sys.exit()

ix = index.open_dir(args.indexdirpath,indexname=args.indexname,readonly=True)

metadata = {}

with ix.searcher() as searcher:
	query = query.Every()

	results = searcher.search(query,limit=None)

	for result in results:
		fields = result.fields()
		metadata[fields['arxiv_id']] = {'date': fields['date'], 'title': fields['title'], 'categories': fields['categories']}

with open(args.picklefilepath, 'wb') as f:
    # Pickle the metadata dictionary using the highest protocol available.
    pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

