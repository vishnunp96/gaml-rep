if __name__ == "__main__":

	import os
	import re

	from utilities.argparseactions import ArgumentParser,FileAction
	from preprocessing import latexmlpy as latexml
	from preprocessing.manifest import ManifestAction

	from lxml import etree as let

	import json
	from collections import Counter

	parser = ArgumentParser(description='Perform simple methodology keyword search on OmegaM papers.')
	parser.add_argument('source',action=ManifestAction, help='Source directory for XML documents, with manifest file.')
	parser.add_argument('data',action=FileAction,mustexist=True,help='Source file with OmegaM paper ids, JSON file.')
	args = parser.parse_args()


	patterns = {
		'CMB': ["Cosmic Microwave Background", "CMB", "primary anistrophies"],
		'LSS': ["galaxy cluster", "cluster of galaxies", r"Ly[ \- ]?\\alpha forest", r"Ly[ \- ]?\\alpha", "quasar absorption", "quasar"],
		'Pec.Vels.': ["peculiar velocity", "peculiar velocities"],
		'SN': ["SN", "supernova", "supernovae", r"\bIa\b"],
		'Lensing': ["lensing", "\blens\b", "gravitational lens", "gravitational lensing", "lensing shear"],
		'BBN': ["nucleosynthesis", "big bang"],
		#'Clusters': ["abundance", "mass", "galaxy", r"Sunyaev[ \- ]?Zeldovich"],
		'Clusters': [r"cluster[^\.]+abundance", "cluster[^\.]+mass", "gala(xy|ctic)[^\.]+abundance", "gala(xy|ctic)[^\.]+mass", r"Sunyaev[ \- ]?Zeldovich"],
		'BAO': ["Baryonic Acoustic Oscillation", "BAO"],
		'ISW': ["Integrated Sachs Wolfe", "Sachs Wolfe", "ISW"],
		'z dist.': ["z distortion", "redshift distortion"],
		'Other': ["Tully Fisher", "galaxy age", r"galaxy colou?r", "globular cluster", "cepheid", "brightness fluctuation", "reverberation map", "radio source", "Gamma Ray Burst", "GRB"]
		}

	with open(args.data,'r') as f:
		data = json.load(f)

	print({k:v for k,v in data.items() if k!='data'})

	def get_score(text, pattern_list):
		score = 0
		for p in pattern_list:
			score += len([j for i in re.findall(p, text) for j in i.split()])
		return score

	for e_idx,entry in enumerate(data['data']):

		arXivID = entry['arxiv_id']
		path = args.source[arXivID]

		root = let.parse(path).getroot()

		elems = root.findall('abstract')
		if elems:
			abstract = elems[0]
		else:
			abstract = None
			print('Could not find element with tag \'abstract\' for ' + path)

		category = 'Unknown'

		if abstract is not None:
			text, _ = latexml.tostring(root)
			scores = {k:get_score(text,ps) for k,ps in patterns.items()}
			category = max(scores.keys(), key=lambda k: scores[k]/len(patterns[k]))

		entry['methodology'] = category

		print(e_idx, arXivID, category)

	print(Counter([e['methodology'] for e in data['data']]))

	oldname,ext = os.path.splitext(args.data)
	newfilename = oldname + '_categorized' + ext
	print(f'Saving to {newfilename}')
	with open(newfilename, 'w') as f:
		json.dump(data, f)
