if __name__ == '__main__':

	from tensorflow.keras.models import load_model

	import matplotlib.pyplot as plt
	plt.switch_backend('agg')

	import numpy
	import pandas

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	from gaml.preprocessing.manifest import ManifestAction
	import os
	import re
	import itertools
	from lxml import etree as let

	from sklearn.externals import joblib

	from gaml.annotations.bratutils import Standoff
	from gaml.annotations.brattowindow import StandoffLabels
	from gaml.annotations.wordembeddings import WordEmbeddings
	from gaml.preprocessing import latexmlpy as latexml
	from gaml.metadata.oaipmh import MetadataAction

	import sqlite3
	import gaml.annotations.database as database

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('source',action=ManifestAction, help='Source directory for XML documents, with manifest file.')
	parser.add_argument('metadata',action=MetadataAction, help='arXiv metadata file (pickled).')
	parser.add_argument('data',action=FileAction,mustexist=True,help='List of arXiv papers to divide up, from CSV file. A single column will be used as a key (defaults to \'id\').')
	parser.add_argument('-c','--column',default='id',help='Column to use as key for data. Defaults to \'id\'.')
	parser.add_argument('-e','--element', help='Element tag to take from document. By default, first element instance will be processed. If no elements of this type exist, or flag is unspecified, whole document will be processed.')
	parser.add_argument('-N','--elemnumber', type=int,default=0, help='Element index (in document order) to take from the list of elements specified. Ignored if --element not given. Defaults to 0.')
	parser.add_argument('-n','--nrows',type=int,default=None,help='Number of rows of dataset to read in. Defaults to all.')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('entities',action=DirectoryAction, mustexist=True,help='Entities model.')
	parser.add_argument('relations',action=DirectoryAction, mustexist=True,help='Relations model.')
	parser.add_argument('output',action=DirectoryAction, mustexist=False, mkdirs=True,help='Directory in which to output predictions.')
	parser.add_argument('-d','--database', action='store_true', help='Flag to indicate that the predictions should be processed and stores in a database.')
	args = parser.parse_args()

	## Read in models and data
	embeddings = WordEmbeddings.open(args.embeddings)

	entity_model_name = os.path.basename(args.entities)
	relation_model_name = os.path.basename(args.relations)

	entity_labels = joblib.load(os.path.join(entity_model_name,entity_model_name+'_labels.joblib'))
	relation_labels = joblib.load(os.path.join(relation_model_name,relation_model_name+'_labels.joblib'))
	relation_entity_labels = joblib.load(os.path.join(relation_model_name,relation_model_name+'_entity_labels.joblib'))

	entity_model = load_model(os.path.join(entity_model_name,entity_model_name+'.h5'))
	relation_model = load_model(os.path.join(relation_model_name,relation_model_name+'.h5'))

	def minmaxmean(x):
		return numpy.concatenate([x.min(axis=0),x.max(axis=0),x.mean(axis=0)])

	## Infer values and reconstruct required data
	numpy.random.seed(42)
	padding_vector = numpy.random.normal(size=(embeddings.dim,))

	window_size = (int(entity_model.input.shape[1]) - int(entity_model.output.shape[1])) / embeddings.dim
	assert window_size%2 == 1
	window_size = int(window_size) # Just to be sure it was an integer to start with...
	window_pad = window_size//2

	data = pandas.read_csv(args.data,nrows=args.nrows)[args.column]

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

	os.makedirs(args.output, exist_ok=True)

	if args.database:
		connection = sqlite3.connect(os.path.join(args.output,'predictions.db'))
		connection.execute('PRAGMA foreign_keys = 1')
		cursor = connection.cursor()
		database.init(cursor)

	for arXiv in data:
		path = args.source[arXiv]

		if path:
			root = let.parse(path).getroot()

			if args.element:
				elems = root.findall('.//'+args.element)
				if elems:
					root = elems[args.elemnumber]
				else:
					print('Could not find element with tag \'' + args.element + '\' for ' + path)
					continue

			text, spans = latexml.tostring(root)
			ann = Standoff.create(text)
			entity_predictions = {}

			## Cycle through entities
			tokens = list(tokenize(ann)) # (token,start,finish)?
			document_matrix = numpy.vstack([padding_vector]*window_pad + [embeddings[t] for t,s,e in tokens] + [padding_vector]*window_pad)

			previous_pred = entity_labels.transform([StandoffLabels.outside_label])
			ent_type,ent_start,ent_end,ent_conf = None,None,None,None
			for i,(token,start,finish) in enumerate(tokens):
				j = i + window_pad
				window = document_matrix[(j-window_pad):(j+window_pad+1)].flatten()

				pred = entity_model.predict(numpy.concatenate((window.reshape(1,-1),previous_pred),axis=1))
				pred_label = entity_labels.inverse_transform(pred)[0]
				previous_pred = pred

				## Deal with prediction
				if pred_label.endswith('_begin'): # Start new entity
					if ent_type is not None:
						ent = ann.entity(ent_type,ent_start,ent_end)
						ann.note(ent,summarize_ent_conf(ent_conf))
						entity_predictions[ent] = numpy.vstack(ent_conf).mean(axis=0)
					ent_type = pred_label[:-6] # Remove '_begin' from label
					ent_start,ent_end = start,finish
					ent_conf = [pred]
				elif pred_label.endswith('_inside'):
					if ent_type is None: # No begin label was given, treat as "begin"
						ent_type = pred_label[:-7] # Remove '_inside' from label
						ent_start,ent_end = start,finish
						ent_conf = [pred]
					else: # ent_type is not None
						if ent_type != pred_label[:-7]: # Transitioned to new type
							ent = ann.entity(ent_type,ent_start,ent_end) # Finish previous entity
							ann.note(ent,summarize_ent_conf(ent_conf))
							entity_predictions[ent] = numpy.vstack(ent_conf).mean(axis=0)
							ent_type = pred_label[:-7] # Remove '_inside' from label
							ent_start,ent_end = start,finish
							ent_conf = [pred]
						else: # Same type as before
							ent_end = finish
							ent_conf.append(pred)
				else: # 'outside' label by default
					if ent_type is not None: # There is an entity which needs finishing
						ent = ann.entity(ent_type,ent_start,ent_end)
						ann.note(ent,summarize_ent_conf(ent_conf))
						entity_predictions[ent] = numpy.vstack(ent_conf).mean(axis=0)
						ent_type,ent_conf = None,None

			for start,end in itertools.combinations(ann.entities,2):
				tokens = numpy.stack([padding_vector] + [embeddings[t] for t in ann.text[start.end:end.start].split()] + [padding_vector])
				ent1 = numpy.stack(embeddings[t] for t in start.text.split())
				ent2 = numpy.stack(embeddings[t] for t in end.text.split())

				x_in = numpy.concatenate([minmaxmean(tokens),minmaxmean(ent1),minmaxmean(ent2),relation_entity_labels.transform([start.type,end.type]).reshape(-1)])

				pred = relation_model.predict(x_in.reshape(1,-1))
				pred_label = relation_labels.inverse_transform(pred)[0]

				## Deal with prediction
				if pred_label != 'none':
					rel = ann.relation(pred_label,start,end)
					ann.note(rel,f'Confidence: {pred.max():.2f}')


			print(f'{arXiv:>18}: {len(ann.entities):4d} entities, {len(ann.relations):4d} relations')

			ann.save(os.path.join(args.output,arXiv.replace('/','')+'.ann'))

			if args.database: database.add(ann,arXiv,args.metadata.get(arXiv,'date'),cursor)

	if args.database:
		connection.commit()
		connection.close()
