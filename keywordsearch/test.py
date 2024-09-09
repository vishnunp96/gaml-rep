from gaml.utilities.argparseactions import ArgumentParser,FileAction

def make_pair(l,type=None):
	if type:
		l = [type(i) for i in l]
	if len(l) == 2:
		return tuple(l)
	elif len(l) == 1:
		return (l[0],l[0])
	raise ValueError(f'Must take 1 or 2 length argument, not {len(l)}.')

parser = ArgumentParser(description="Plot histogram of values in file.")
parser.add_argument('sourcepath',action=FileAction, mustexist=True, help='Values to plot.')
parser.add_argument('-H','--histogram',action=FileAction,nargs='?',const=True,help='Plot histogram of values.')
parser.add_argument('-P','--pretty',action='store_true',help='Plot scatter/histogram of values.')
parser.add_argument('-n','--nbins',type=int,nargs='+',default=[100,100],help='Plot scatter/histogram of values.')
args = parser.parse_args()

args.nbins = make_pair(args.nbins,type=int)

print(args)
