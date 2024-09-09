import tarfile
import gaml.utilities.mygzip as gzip
import os
import re

## yield from generator syntax useful here

def is_tarfile(filename=None,fileobj=None):
	if fileobj is None:
		fileobj = open(filename, 'rb')
	try:
		t = tarfile.open(fileobj=fileobj)
		n = t.next()
		t.close()
		return bool(n)
	except tarfile.TarError:
		return False

def pathjoin(*paths):
	fullpath = ''
	for path in paths:
		fullpath = os.path.join(fullpath,os.path.splitext(path)[0])
	return fullpath + os.path.splitext(paths[-1])[1]


def crawlFilesystem(start,depth=0):
	print(('\t'*depth)+'crawlFilesystem ('+str(depth)+')')
	if os.path.isdir(start):
		crawlDir(start)
	else:
		with open(start,'rb') as f:
			if is_tarfile(fileobj=f):
				crawlTar(fileobj=f,depth=depth+1)
			elif gzip.is_gzip(f):
				crawlGzip(fileobj=f,depth=depth+1)
			else:
				print(('\t'*depth)+start)


def crawlDir(start,depth=0):
	print(('\t'*depth)+'crawlDir ('+str(depth)+')')
	for (dirpath, dirnames, filenames) in os.walk(start):
		for filepath in [os.path.join(dirpath,filename) for filename in filenames]:
			with open(filepath,'rb') as f:
				if filepath.endswith('.tar'):
					crawlTar(fileobj=f,depth=depth+1)
				elif filepath.endswith('.gz'):
					crawlGzip(fileobj=f,depth=depth+1)
				else:
					print(('\t'*depth)+filepath)


def crawlTar(filename=None,fileobj=None,depth=0):
	print(('\t'*depth)+'crawlTar ('+str(depth)+')')
	if fileobj is None:
		fileobj = open(filename, 'rb')

	with tarfile.open(fileobj=fileobj,mode='r') as tf:
		for entry in tf:
			print(('\t'*depth)+entry.name)
			if entry.isfile():
				if entry.name.endswith('.gz'):
					crawlGzip(fileobj=tf.extractfile(entry),depth=depth+1)
				elif entry.name.endswith('.tar'):
					crawlTar(fileobj=tf.extractfile(entry),depth=depth+1)
				else:
					print(('\t'*depth)+entry.name)


def crawlGzip(filename=None,fileobj=None,depth=0):
	print('\t'*depth+'crawlGzip ('+str(depth)+')')
	if fileobj is None:
		fileobj = open(filename, 'rb')

	with gzip.open(fileobj,mode='rb') as f:
		if f.name.endswith('.tar'):
			crawlTar(fileobj=f,depth=depth+1)
		else:
			print(('\t'*depth)+f.name)


#with tarfile.open(args.path) as tf:
#	for entry in tf:  # list each entry one by one
#		#print('Entry type:',type(entry))
#		print('Entry:',entry.name,entry.isdir(),entry.isfile())
#		#print([s for s in dir(entry) if not s.startswith('_')])
#		if entry.isfile():
#			fileobj = tf.extractfile(entry)
#			try:
#				with gzip.open(fileobj,mode='rb') as f:
#					try:
#						with tarfile.open(fileobj=f) as ntf:
#							for nentry in ntf:
#								print('\t',nentry.name)
#					except:
#						print('\t','Not tar file ('+f.name+').')
#					#print('\t',f.name,type(f))
#					#for i in [s for s in dir(f) if not s.startswith('_')]:
#					#	if not callable(getattr(f,i)):
#					#		print('\t'+i+':',getattr(f,i))
#			except:
#				print('\t','Not gzip file ('+entry.name+').')
#		#print('fileobj type:',type(fileobj))
#		#print([s for s in dir(fileobj) if not s.startswith('_')])
#		#text = fileobj.read()

#crawlFilesystem(args.path)



if __name__ == '__main__':

	from orbitapy.utilities import StopWatch
	stopwatch = StopWatch(memory=False)

	from gaml.utilities.argparseactions import ArgumentParser,PathAction
	from gaml.metadata import MetadataAction

	from collections import defaultdict

	parser = ArgumentParser()
	parser.add_argument('path',action=PathAction, mustexist=True, allowed=['file','dir'])
	parser.add_argument('metadata',action=MetadataAction)
	args = parser.parse_args()

	results = defaultdict(int)
	extensions = defaultdict(int)

	texre = re.compile('(('+re.escape('.tex')+')|('+re.escape('.latex')+')|('+re.escape('.ltx')+'))'+'$',re.IGNORECASE)
	psre = re.compile(re.escape('.ps')+'$',re.IGNORECASE)
	htmlre = re.compile(re.escape('.html')+'$',re.IGNORECASE)
	withdrawre = re.compile(re.escape('withdraw'),re.IGNORECASE)
	textre = re.compile('^'+re.escape('text')+'$',re.IGNORECASE)

	## HTML
	# .tex.bak?
	# .tex.cry
	# Lots of files lacking extensions which are probably Latex.

	if args.path_type is 'dir':
		tarlist = [args.path + os.sep + n for n in os.listdir(args.path) if n.endswith('.tar')]
	elif args.path_type is 'file':
		tarlist = [args.path]

	for tarpath in tarlist:
		print('Start:',os.path.basename(tarpath))
		with tarfile.open(tarpath,mode='r') as tf:
			for entry in tf:
				if entry.isfile():
					entrymetadata = args.metadata.get(os.path.basename(os.path.splitext(entry.name)[0]))
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
		print(results)

	import operator
	report = ''
	for key,value in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
		report += key + ':' + str(value) + ','
	print(report + 'Total:'+str(sum(results.values())))

	for key,value in sorted(extensions.items(), key=operator.itemgetter(1), reverse=True):
		if value==10: break
		print('{:30} {}'.format(key,value))

	stopwatch.report(prefix='')

