import pickle
import os
import argparse

from directoryactions import FileAction

parser = argparse.ArgumentParser(description="Some queries for arXiv data stored in a Whoosh index.")
parser.add_argument("picklefilepath",action=FileAction,mustexist=True,help='Path of metadata pickled file.')
args = parser.parse_args()

with open(args.picklefilepath, 'rb') as f:
    # The protocol version used is detected automatically, so we do not have to specify it.
    data = pickle.load(f)

print(str(data['astro-ph/0001001']['title']))
print(str(data['astro-ph/0001002']['title']))
print(str(data['astro-ph/0001003']['title']))
print(str(data['astro-ph/0105004']['title']))
#print(str(data['']['title']))

astrocount = 0
for key in data:
	if 'astro' in data[key]['categories']:
		astrocount += 1

print('Astro papers count: ' + str(astrocount))
