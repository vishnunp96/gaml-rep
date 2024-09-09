if __name__ == "__main__":

	import pandas
	import re
	import sys
	from preprocessing.manifest import ManifestAction
	import preprocessing.latexmlpy as latexml
	from utilities.argparseactions import ArgumentParser,FileAction
	from utilities import StopWatch
	from utilities.gxml import fastxmliter
	from utilities.terminalutils import printacross,printwrapped,color_text,ColorCode
	#from keywordsearch.rulesBasedSearch import measurementre

	stopwatch = StopWatch()

	parser = ArgumentParser(description="Use keyword search to find values of provided keywords in a corpus of LateXML documents.")
	parser.add_argument('source',action=ManifestAction, help='Directory containing articles and manifest.')
	parser.add_argument('output',action=FileAction, mustexist=False, help='File in which output will be stored or read from (to resume classification).')
	parser.add_argument('-s','--scores',action=FileAction, mustexist=True, help='File of scores from which to sample articles for classification.')
	parser.add_argument('-n','--number', type=int,default=100, help='Number of articles of each (silver) class to sample. Defaults to 100.')
	parser.add_argument('-m','--minscore',type=int, default=1, help='Minimum score for an article to be considered a positive sample. Defaults to 2.')
	parser.add_argument('-S','--summary',action='store_true', help='Only output summary, then close.')
	args = parser.parse_args()

	# Select n entries from score list
	# Loop over entries
	# Print abstract of each entry
	# Request classification from user
	# Save classification and id (write to file as soon as recieved, to prevent losing work)
	# Try and implement a function for resuming a lost session?
	##### Create table of ids and labels, with -1s in label column
	##### Upon reloading, any entry which does not contain -1 has already been classified
	##### This would mean that we have to re-save whole table every time, but that won't be costly

	manifest = args.source

	if args.output_exists:
		data = pandas.read_csv(args.output)
	elif args.scores:
		scores = pandas.read_csv(args.scores)

		pos_ids = scores[scores['score'] >= args.minscore]#['id']
		neg_ids = scores[scores['score'] == 0]#['id']
		pos_sample = pos_ids.sample(n=args.number)
		neg_sample = neg_ids.sample(n=args.number)
		data = pandas.DataFrame(data=pandas.concat([pos_sample,neg_sample]))
		data['label'] = -1
		data = data.sample(frac=1)
	else:
		sys.exit('Must provide existing output file, or scores file to sample from.')

	#print(data)

	if not args.summary:

		counter = 0
		for i in data.index:
			counter += 1

			if data.at[i,'label'] != -1:
				continue

			print('\033[2J\033[H',end='')
			printacross('=',maxwidth=70)
			printacross('=', begin=f'{data.at[i,"id"]:24}({counter}/{data.shape[0]}) ', maxwidth=70)
			print()

			for event,p in fastxmliter(manifest[data.at[i,'id']], events=("end",), tag='abstract'):
				text,spans = latexml.tostring(p)

				mathspans = [(s[0],s[1]) for s in spans if s[3]=='text' and 'Math' in s[2]]
				text = color_text(text,ColorCode.red,spans=mathspans)

				text = text.strip()
				text = re.sub('\ +\.','.',text)
				text = re.sub('\ +\,',',',text)
				text = re.sub('\ +\:',':',text)
				text = re.sub('\ +\)',')',text)
				text = re.sub('\(\ +','(',text)
				text = re.sub('(?<=[A-Za-z])\s*\-\s*(?=[A-Za-z])','-',text)
				text = re.sub('\xe2','-',text)
				text = text.encode('utf8').decode('ascii',errors='ignore')

				#text = re.sub(measurementre,'\x1b[0;31;40m\g<0>\x1b[0m',text)

				printwrapped(text, width=60)
				print()

			printacross('=',maxwidth=70)

			while True:
				result = input('Article reports new measurement (y/n): ').lower().strip()
				if result == 'y' or result =='n' or result == 's' or result == 'exit': break

			if result == 'y':
				data.at[i,'label'] = 1
			elif result == 'n':
				data.at[i,'label'] = 0
			elif result == 'exit':
				break

			data.to_csv(args.output,index=False)


		print('\033[2J\033[H',end='') # Clear console

	print(data)

	def accuracy(labelled):
		return sum((labelled['score'] >= args.minscore) == (labelled['label'] == 1)) / labelled.shape[0]

	def precision(labelled):
		ground_pos_data = labelled[labelled['score'] >= args.minscore]
		return sum(ground_pos_data['label'] == 1) / ground_pos_data.shape[0]

	labelled_data = data[data['label'] != -1]

	print(f'Total accuracy:    {accuracy(labelled_data)}')
	print(f'Silver precision:  {precision(labelled_data)}')

	print(f'Uncategoried samples: {data["label"].value_counts().to_dict().get(-1)}')
