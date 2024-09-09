if __name__ == '__main__':

	from utilities import StopWatch
	stopwatch = StopWatch(memory=True)

	from tensorflow.keras.models import load_model
	from utilities.kerasutils import Projection
	from tensorflow.keras.preprocessing.sequence import pad_sequences

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas

	from utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from preprocessing.manifest import Manifest
	from metadata.oaipmh import arXivID_from_path
	from utilities.fileutilities import iter_files
	import os
	import re
	import itertools
	#from lxml import etree as let
	from utilities.gxml import fastxmliter

	from sklearn.externals import joblib

	from annotations.bratutils import Standoff
	from annotations.wordembeddings import WordEmbeddings
	from preprocessing import latexmlpy as latexml
	from metadata.oaipmh import MetadataAction

	import sqlite3
	import annotations.database as database

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('source',action=DirectoryAction, help='Source directory for XML documents, which will either be searched recursively, or must contain a manifest file if --idlist is specified.')
	parser.add_argument('metadata',action=MetadataAction, help='arXiv metadata file (pickled).')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('entities',action=DirectoryAction, mustexist=True,help='Entities model.')
	parser.add_argument('-r','--relations',action=DirectoryAction, mustexist=True,help='Relations model.')
	parser.add_argument('-i','--idlist',action=FileAction,mustexist=True,help='List of arXiv papers to divide up, from CSV file. A single column will be used as a key (defaults to \'id\').')
	parser.add_argument('-c','--column',default='id',help='Column to use as key for id list. Defaults to \'id\'.')
	parser.add_argument('-n','--nrows',type=int,default=None,help='Number of rows of dataset to read in. Defaults to all.')
	parser.add_argument('-o','--output',action=DirectoryAction, mustexist=False, mkdirs=True,help='Directory in which to output predictions.')
	parser.add_argument('-d','--database', action=FileAction, mustexist=False, help='If a filepath is provided, a database of the predictions will be created and stored at this location.')
	parser.add_argument('-I','--initialise',action='store_true',help='Flag to indicate that the database should be initialised.')
	args = parser.parse_args()

	## Read in models and data
	embeddings = WordEmbeddings.open(args.embeddings)

	entity_model_name = os.path.basename(args.entities)
	if args.relations: relation_model_name = os.path.basename(args.relations)

	entity_labels = joblib.load(os.path.join(args.entities,entity_model_name+'_labels.joblib'))
	if args.relations: relation_labels = joblib.load(os.path.join(args.relations,relation_model_name+'_labels.joblib'))
	#relation_entity_labels = joblib.load(os.path.join(args.relations,relation_model_name+'_entity_labels.joblib'))

	custom_objects={i.__name__:i for i in [Projection]}
	entity_model = load_model(os.path.join(args.entities,entity_model_name+'.h5'),custom_objects=custom_objects)
	if args.relations: relation_model = load_model(os.path.join(args.relations,relation_model_name+'.h5'),custom_objects=custom_objects)

	def minmaxmean(x):
		return numpy.concatenate([x.min(axis=0),x.max(axis=0),x.mean(axis=0)])

	## Infer values and reconstruct required data
	numpy.random.seed(42)
	padding_vector = numpy.random.normal(size=(embeddings.dim,))

	if args.idlist:
		manifest = Manifest(args.source)
		ids = pandas.read_csv(args.idlist,nrows=args.nrows)[args.column]

	whitespacere = re.compile(r'\s+|$')
	def tokenize(standoff):
		current_index = 0
		for match in whitespacere.finditer(standoff.text):
			start,end = current_index,match.start()
			token = standoff.text[start:end]
			if token:
				yield token,start,end
			current_index = match.end()

	def summarize_ent_conf(confs):
		c = numpy.vstack(confs)
		return f'Confidence: {c.max(axis=1).mean():.2f}' + (f' ([{", ".join(f"{i:.2f}" for i in c.max(axis=1))}])' if len(confs)>1 else '')

	if args.output:
		os.makedirs(args.output, exist_ok=True)

	if args.database:
		connection = sqlite3.connect(args.database, timeout=60)
		if args.initialise:
			cursor = connection.cursor()
			database.init(cursor)
		connection.commit()

	papercount = 0
	processedcount = 0
	for arXiv,path in (((a,manifest[a]) for a in ids) if args.idlist else ((arXivID_from_path(p),p) for p in iter_files(args.source,suffix='.xml'))):
		papercount += 1
		if path:
			text = ''
			for event,p in fastxmliter(path, events=("end",), tag='abstract'):
				text += latexml.tostring(p)[0] + '\n'
			text = text.strip()
			if not text:
				print('Could not find text for ' + path)
				continue

			#text, spans = latexml.tostring(root)
			ann = Standoff.create(text)
			entity_predictions = {}

			## Cycle through entities
			tokens = list(tokenize(ann)) # (token,start,finish)?
			x = numpy.array([[embeddings.getindex(t) for t,s,e in tokens]])

			pred = entity_model.predict(x)[0]
			pred_labels = entity_labels.inverse_transform(pred)

			ent_type,ent_start,ent_end,ent_conf = None,None,None,None
			for i,(token,start,finish) in enumerate(tokens):
				pred_label = pred_labels[i]

				## Deal with prediction
				if pred_label.endswith('_begin'): # Start new entity
					if ent_type is not None:
						ent = ann.entity(ent_type,ent_start,ent_end)
						ann.note(ent,summarize_ent_conf(ent_conf))
						entity_predictions[ent] = numpy.vstack(ent_conf).mean(axis=0)
					ent_type = pred_label[:-6] # Remove '_begin' from label
					ent_start,ent_end = start,finish
					ent_conf = [pred[i]]
				elif pred_label.endswith('_inside'):
					if ent_type is None: # No begin label was given, treat as "begin"
						ent_type = pred_label[:-7] # Remove '_inside' from label
						ent_start,ent_end = start,finish
						ent_conf = [pred[i]]
					else: # ent_type is not None
						if ent_type != pred_label[:-7]: # Transitioned to new type
							ent = ann.entity(ent_type,ent_start,ent_end) # Finish previous entity
							ann.note(ent,summarize_ent_conf(ent_conf))
							entity_predictions[ent] = numpy.vstack(ent_conf).mean(axis=0)
							ent_type = pred_label[:-7] # Remove '_inside' from label
							ent_start,ent_end = start,finish
							ent_conf = [pred[i]]
						else: # Same type as before
							ent_end = finish
							ent_conf.append(pred[i])
				else: # 'outside' label by default
					if ent_type is not None: # There is an entity which needs finishing
						ent = ann.entity(ent_type,ent_start,ent_end)
						ann.note(ent,summarize_ent_conf(ent_conf))
						entity_predictions[ent] = numpy.vstack(ent_conf).mean(axis=0)
						ent_type,ent_conf = None,None

			if args.relations:
				entity_pairs = []
				x_in = []
				for start,end in itertools.combinations(ann.entities,2):
					#tokens = numpy.stack([padding_vector] + [embeddings[t] for t in ann.text[start.end:end.start].split()] + [padding_vector])
					#ent1 = numpy.stack(embeddings[t] for t in start.text.split())
					#ent2 = numpy.stack(embeddings[t] for t in end.text.split())
					#x_in = numpy.concatenate([minmaxmean(tokens),minmaxmean(ent1),minmaxmean(ent2),relation_entity_labels.transform([start.type,end.type]).reshape(-1)])

					entity_pairs.append((start,end))
					x_in.append([embeddings.getindex(t) for t in ann.text[start.start:end.end].split()])

				if len(x_in)>0:
					x_in = pad_sequences(x_in, padding='post')

					predictions = relation_model.predict(x_in)
					prediction_labels = relation_labels.inverse_transform(pred)

					## Deal with predictions
					for (start,end),pred,label in zip(entity_pairs, predictions, prediction_labels):
						if label != 'none':
							rel = ann.relation(label,start,end)
							ann.note(rel,f'Confidence: {pred.max():.2f}')


			print(f'{arXiv:>18}: {len(ann.entities):4d} entities, {len(ann.relations):4d} relations')
			processedcount += 1

			if args.output:
				ann.save(os.path.join(args.output,arXiv.replace('/','')+'.ann'))

			if args.database:
				cursor = connection.cursor()
				database.add(ann,arXiv,args.metadata.get(arXiv,'date'),cursor)
				connection.commit()

	if args.database:
		connection.close()

	print(f'Processed {processedcount} papers out of {papercount}.')
	stopwatch.report()
