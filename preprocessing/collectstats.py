import tarfile
import gaml.utilities.mygzip as gzip
import os
import re
import pandas

import matplotlib.pyplot as plt

texre = re.compile('(('+re.escape('.tex')+')|('+re.escape('.latex')+')|('+re.escape('.ltx')+'))'+'$',re.IGNORECASE)
psre = re.compile(re.escape('.ps')+'$',re.IGNORECASE)
htmlre = re.compile(re.escape('.html')+'$',re.IGNORECASE)
withdrawre = re.compile(re.escape('withdraw'),re.IGNORECASE)
textre = re.compile('^'+re.escape('text')+'$',re.IGNORECASE)


def collect_from_tar(tarpath):

	results = defaultdict(int)
	extensions = defaultdict(int)

	datestr = re.search('_([0-9]+)_',os.path.basename(tarpath)).group(1)
	year = int(datestr[:2])
	if datestr[0]=='9': year+=1900
	else: year+=2000
	results['year'] = year
	results['month'] = int(datestr[2:])

	print('Start:',os.path.basename(tarpath))
	with tarfile.open(tarpath,mode='r') as tf:
		for entry in tf:
			if entry.isfile():
				if entry.name.endswith('.pdf'):
					#print('PDF: '+entry.name+', '+entrymetadata['categories'])
					results['pdf']+=1
				else:
					# Extract gzip
					#print('GZP: '+entry.name+', '+entrymetadata['categories'])
					try:
						with tarfile.open(fileobj=tf.extractfile(entry),mode='r:gz') as ntf:
							names = [nentry.name for nentry in ntf if nentry.isfile()]
					except tarfile.ReadError:
						with gzip.open(tf.extractfile(entry),mode='r') as gf:
							names = [gf.name]

					if any(texre.search(name) for name in names):
						results['tex']+=1
					elif all(psre.search(name) for name in names):
						results['ps']+=1
					elif all(htmlre.search(name) for name in names):
						results['html']+=1
					elif all(withdrawre.search(name) for name in names):
						results['withdraw']+=1
					elif all(textre.search(name) for name in names):
						results['text']+=1
					else:
						results['other']+=1
						print(entry.name,names)

					for base in (os.path.basename(name) for name in names):
						parts = base.split('.')
						for i in range(1,len(parts)):
							extensions['.'+'.'.join(parts[i:])] += 1
						if len(parts)==1:
							extensions[parts[0]] += 1

	data = pandas.DataFrame([results])
	data.set_index(['year','month'],inplace=True)

	extensions_frame = pandas.DataFrame.from_dict(extensions, orient='index').rename(columns={0:'count'})

	return data,extensions_frame


if __name__ == '__main__':

	from gaml.utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	from gaml.utilities.argparseactions import ArgumentParser,PathAction,DirectoryAction
	from gaml.metadata import MetadataAction
	from gaml.utilities import parallel

	from collections import defaultdict
	import random

	parser = ArgumentParser()
	parser.add_argument('path',action=PathAction, mustexist=True, allowed=['file','dir'])
	parser.add_argument('metadata',action=MetadataAction)
	parser.add_argument('-p','--processes',default=1,type=int)
	parser.add_argument('-r','--resultsdir',action=DirectoryAction,mustexist=False,mkdirs=True,default=os.path.abspath('.'))
	args = parser.parse_args()

	if args.path_type is 'dir':
		tarlist = [args.path + os.sep + n for n in os.listdir(args.path) if n.endswith('.tar')]
	elif args.path_type is 'file':
		tarlist = [args.path]

	random.shuffle(tarlist) ## Just in case?

	results = pandas.DataFrame(columns=['year','month','tex']).set_index(['year','month'])
	extensions = pandas.DataFrame(columns=['count'])

	with parallel.Pool(processes=args.processes) as pool:
		for r,e in pool.imap_unordered(collect_from_tar, tarlist):
			results = results.add(r,fill_value=0)
			extensions = extensions.add(e,fill_value=0)

	results.fillna(value=0,inplace=True)
	results = results.astype(int)

	results = results[results.sum().sort_values(ascending=True).index]

	#print(results)

	results_noindex = results.reset_index()
	results_year = results.groupby(['year'])[list(set(results.columns) - set(['year','month']))].sum().reset_index().set_index('year')
	results_year = results_year[results_year.sum().sort_values(ascending=True).index]
	#print(results_year)

	extensions.sort_values(by='count',ascending=False,inplace=True)
	#print(extensions)

	resultspath = os.path.join(args.resultsdir,'arXivresults.csv')
	results.to_csv(resultspath)
	resultsyearpath = os.path.join(args.resultsdir,'arXivresultsyear.csv')
	results_year.to_csv(resultsyearpath)
	extensionspath = os.path.join(args.resultsdir,'arXivextensions.csv')
	extensions.to_csv(extensionspath)

	plt.switch_backend('agg')
	#print(results_year.index.shape)
	#print(results_year.shape)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.stackplot(results_year.index,results_year.values.T)
	#ax.set_yscale('symlog',linthreshy=1)
	figpath = os.path.join(args.resultsdir,'arXivstats.png')
	fig.savefig(figpath)

	stopwatch.report(prefix='')

