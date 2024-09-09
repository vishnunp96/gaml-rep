from utilities.argparseactions import ArgumentParser,FileAction
from metadata import MetadataAction
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy
import os

parser = ArgumentParser(description='Description.')
parser.add_argument('metadata',action=MetadataAction,help='Path to ArXiv metadata.')
parser.add_argument('output',action=FileAction,help='Path at which to save figure.')
args = parser.parse_args()

astro = []
other = []
start = 9999
end = 1
for entry in args.metadata.entries():
	entrydate = entry['date'].year
	if 'astro-ph' in entry['categories']:
		astro.append(entrydate)
	else:
		other.append(entrydate)
	start = min(start, entrydate)
	end = max(end, entrydate)

bins = list(range(start, end+1))

astro_cumsum = numpy.insert(numpy.cumsum(numpy.histogram(astro, bins=bins)[0]), 0, 0)
other_cumsum = numpy.insert(numpy.cumsum(numpy.histogram(other, bins=bins)[0]), 0, 0)
astro_peranum = numpy.insert(numpy.histogram(astro, bins=bins)[0], 0, 0)
other_peranum = numpy.insert(numpy.histogram(other, bins=bins)[0], 0, 0)

def make_plot(astro_vals, other_vals, note, ylabel):
	fontsize = 12

	plt.figure(figsize=(4.8,4.8))
	plt.stackplot(bins, astro_vals, other_vals, labels=['astro-ph','other'])
	plt.xlim(1992, end)
	plt.ylim(0, (astro_vals+other_vals).max())
	plt.xlabel('Year', fontsize=fontsize)
	plt.ylabel(ylabel, fontsize=fontsize)

	def human_number(x):
		#mag = 0
		#while x>1000:
		#	x /= 1000
		#	mag += 1
		#return f'{x:.0f}' + ['', 'k', 'M', 'B'][mag]
		if x==0:
			return '0'
		elif x==1000000:
			return '1M'
		elif x>=1000000:
			return f'{x/1000000:.1f}M'
		else:
			return f'{x/1000:.0f}k'

	# after plotting the data, format the labels
	current_values = plt.gca().get_yticks()
	# using format string '{:.0f}' here but you can choose others
	plt.gca().set_yticklabels([human_number(x) for x in current_values])

	parts = os.path.splitext(args.output)
	output = parts[0]+note+parts[1]

	plt.legend(loc='upper left')
	plt.savefig(output, bbox_inches='tight', dpi=300)

make_plot(astro_cumsum, other_cumsum, 'cumsum', 'Number of Artices in the arXiv')
make_plot(astro_peranum, other_peranum, 'peranum', 'Yearly Submissions to the arXiv')
