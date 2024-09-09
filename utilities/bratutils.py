def read_ann(path):
	return Annotation(path)

class Annotation:
	def __init__(self,path):
		with open(path,'r') as f:
			lines = [l for l in f.read().split('\n') if l]

		self.entities = {}
		self.relations = {}
		for l in lines:
			tokens = l.split('\t')

			if tokens[0][0] == 'T':
				details = tokens[1].split(' ')
				tag = details[0]
				#span = ' '.join(details[1:])
				span = [tuple([int(s) for s in i.split(' ') if s]) for i in ' '.join(details[1:]).split(';')]
				text = tokens[2]
				self.entities[tokens[0]] = {'tag':tag,'span':span,'text':text}
			elif tokens[0][0] == 'R':
				details = tokens[1].split(' ')
				tag = details[0]
				arg1 = details[1].split(':')[1]
				arg2 = details[2].split(':')[1]
				self.relations[tokens[0]] = {'tag':tag,'arg1':arg1,'arg2':arg2}

	def __getitem__(self, sid):
		if sid[0] == 'T':
			return self.entities[sid]
		elif sid[0] == 'R':
			return self.relations[sid]

	def get_relations(self,tag=None,arg1=None,arg2=None):
		return [val for val in self.relations.values() if (val['tag'] == tag if tag else True) and (val['arg1'] == arg1 if arg1 else True) and (val['arg2'] == arg2 if arg2 else True)]

	def get_entities(self,tag=None):
		return [val for val in self.entities.values() if (val['tag'] == tag if tag else True)]

	def relation_dicts(self,tag=None,arg1=None,arg2=None):
		for relation in self.get_relations(tag=tag,arg1=arg1,arg2=arg2):
			yield {
					self.entities[relation['arg1']]['tag']: self.entities[relation['arg1']]['text'],
					self.entities[relation['arg2']]['tag']: self.entities[relation['arg2']]['text']
				}

