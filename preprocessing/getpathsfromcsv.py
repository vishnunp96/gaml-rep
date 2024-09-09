import pandas

from gaml.preprocessing.manifest import Manifest

from gaml.utilities.argparseactions import ArgumentParser,FileAction

parser = ArgumentParser(description='Get filepaths from manifest using csv.')
parser.add_argument('manifest',action=FileAction, mustexist=True, help='Path to manifest file.')
parser.add_argument('csv',action=FileAction, mustexist=True, help='Path to csv file.')
parser.add_argument('output',action=FileAction, mustexist=False, help='Path for output file.')
args = parser.parse_args()


manifest = Manifest('/cluster/project2/gaml/data/arXivXML',args.manifest)

data = pandas.read_csv(args.csv)

with open(args.output,'w') as f:
	for i in data[data['parameter'] == 'H_0']['arXiv'].dropna().unique():
		path = manifest[i]
		if path:
			f.write(path + '\n')
		else:
			print('Could not find path for ' + i)
