if __name__ == '__main__':

	import torch

	from gaml.annotations.models import load_ann_model
	#from . import IndexWindowMemoryEntityModel, IndexSplitSpansRelationModel
	#from gaml.annotations.models.indexwindowmemoryentitymodel import IndexWindowMemoryEntityModel
	#from gaml.annotations.models.testcharentitymodel import IndexCharsWindowMemoryEntityModel
	#from gaml.annotations.models.windowattributemodel import WindowAttributeModel
	#from gaml.annotations.models.indexsplitspansrelationmodel import IndexSplitSpansRelationModel
	#from gaml.annotations.models.rulesrelationmodel import RulesRelationModel

	from gaml.utilities.argparseactions import ArgumentParser,FileAction,DirectoryAction
	import os
	import re
	import itertools
	from lxml import etree as let

	from gaml.annotations.bratutils import Standoff
	from gaml.preprocessing import latexmlpy as latexml
	from gaml.metadata.oaipmh import MetadataAction,arXivID_from_path
	from gaml.utilities.fileutilities import changeext,iter_files
	from gaml.annotations.bratnormalisation import find_repititions,add_implied_relations,remove_stopwords

	import sqlite3
	import gaml.annotations.database as database

	parser = ArgumentParser(description='Train Keras ANN to predict entities in astrophysical text.')
	parser.add_argument('source',action=DirectoryAction, help='Directory to search (recursively) for XML files to process.')
	parser.add_argument('-s','--subdir', help='Sub-directory to be appended to \'source\' in which search will be performed.')
	parser.add_argument('metadata',action=MetadataAction, help='arXiv metadata file (pickled).')
	parser.add_argument('-e','--element', help='Element tag to take from document. By default, first element instance will be processed. If no elements of this type exist, or flag is unspecified, whole document will be processed.')
	parser.add_argument('-N','--elemnumber', type=int,default=0, help='Element index (in document order) to take from the list of elements specified. Ignored if --element not given. Defaults to 0.')
	parser.add_argument('entities',action=FileAction, mustexist=True, help='Entities model.')
	parser.add_argument('--attributes',action=FileAction, mustexist=True, help='Attributes model. Optional.')
	parser.add_argument('relations',action=FileAction, mustexist=True, help='Relations model.')
	parser.add_argument('-D','--database', action=FileAction, mustexist=True, help='Flag to indicate that the predictions should be processed and stores in a database.')
	parser.add_argument('-A','--standoff', action=DirectoryAction, mustexist=False, mkdirs=True, help='Flag to indicate that the predictions should be stored in standoff format in output directory.')
	parser.add_argument('-R','--no-post', dest='post', action='store_false', help='Flag to indicate that the raw model outputs should be used, without postprocessing.')
	args = parser.parse_args()

	if args.subdir:
		source = os.path.join(args.source, args.subdir)
	else:
		source = args.source
	print(f'Start {source}')

	## Read in models and data
	#entity_model = IndexWindowMemoryEntityModel.load_pretrained(args.entities)
	#entity_model = IndexCharsWindowMemoryEntityModel.load_pretrained(args.entities, map_location=torch.device('cpu'))
	#if args.attributes: attribute_model = WindowAttributeModel.load_pretrained(args.attributes, map_location=torch.device('cpu'))
	#relation_model = IndexSplitSpansRelationModel.load_pretrained(args.relations)
	#relation_model = RulesRelationModel()

	entity_model = load_ann_model(args.entities)
	if args.attributes: attribute_model = load_ann_model(args.attributes)
	relation_model = load_ann_model(args.relations)

	## If CUDA is available, move models onto GPU
	if torch.cuda.is_available():
		entity_model = entity_model.cuda()
		if args.attributes: attribute_model = attribute_model.cuda()
		relation_model = relation_model.cuda()

	# Create path iter
	paths = iter_files(source, recursive=True, suffix='.xml')

	batch_size = 128

	if args.database:
		connection = sqlite3.connect(args.database, timeout=180) # Timeout 2 minutes

		# Mark source directory as STARTED
		cursor = connection.cursor()
		database.start_predictions(source, cursor)
		cursor.close()
		connection.commit()

	def batch(iterable, n):
		l = len(iterable)
		for ndx in range(0, l, n):
			yield iterable[ndx:min(ndx + n, l)]
	def grouper(n, iterable):
		it = iter(iterable)
		while True:
			chunk = tuple(itertools.islice(it, n))
			if not chunk:
				return
			yield chunk

	check_repetition_types = ('ParameterName','ParameterSymbol','ObjectName')

	with entity_model.evaluation(), relation_model.evaluation(): # Small speed up by using no_grad

		for paths in grouper(batch_size, paths):

			batch_anns = []
			for path in paths:
				## Text section needs replacing with the one that combines all available elements
				root = let.parse(path).getroot()

				if args.element:
					elems = root.findall('.//'+args.element)
					if elems:
						root = elems[args.elemnumber]
					else:
						print(f"Could not find element with tag '{args.element}' for {path}")
						continue

				text,_ = latexml.tostring(root)
				if text:
					print(f'Do prediction for {path}')
					batch_anns.append((path, arXivID_from_path(path), Standoff.create(text)))

			batch_anns = entity_model.predict(batch_anns, batch_size=batch_size, inplace=True)

			if args.post: # Find repetitions and remove nonsense from entity predictions
				for path,arXiv,ann in batch_anns:
					remove_stopwords(ann,inplace=True)
					for e in [i for i in ann.entities if i.type in ('ParameterSymbol','ParameterName','ObjectName')]:
						if re.search('[A-Za-z]',e.text) is None:
							ann.entities.remove(e)
					for e in [i for i in ann.entities if i.type in ('MeasuredValue','Constraint')]:
						if re.search('[0-9]',e.text) is None:
							ann.entities.remove(e)
					find_repititions(ann,inplace=True,types=check_repetition_types)

			## If attribute model provided, run it here
			if args.attributes:
				batch_anns = attribute_model.predict(batch_anns, batch_size=batch_size, inplace=True)

			batch_anns = relation_model.predict(batch_anns, batch_size=batch_size, inplace=True)

			if args.post: # Make sure implied relations are present
				for path,arXiv,ann in batch_anns:
					add_implied_relations(ann,inplace=True)

			## Convert Constraint entities to the correct type (based on presence of Attributes)
			if args.attributes:
				for path,arXiv,ann in batch_anns:
					for ent in [att.subject for att in ann.attributes if att.type in ('UpperBound','LowerBound')]:
						ent.type = 'Constraint'

			for path,arXiv,ann in batch_anns:
				print(f'{arXiv:>18}: {len(ann.entities):4d} entities, {len(ann.relations):4d} relations, {len(ann.attributes):4d} attributes')

			if args.standoff:
				for path,arXiv,ann in batch_anns:
					destination = changeext(path.replace(args.source,args.standoff), '.ann')
					if not os.path.exists(os.path.dirname(destination)):
						os.makedirs(os.path.dirname(destination))
					ann.save(destination)
			if args.database:
				cursor = connection.cursor()
				for path,arXiv,ann in batch_anns:
					database.add(ann, arXiv, args.metadata.get(arXiv,'date'), cursor)
				cursor.close()
				connection.commit()

	if args.database:
		# Mark source directory as COMPLETED
		cursor = connection.cursor()
		database.complete_predictions(source, cursor)
		cursor.close()
		connection.commit()

		connection.close()

	print(f'Finished {source}')
