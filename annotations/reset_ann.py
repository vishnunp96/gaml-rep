if __name__ == "__main__":

	from utilities.argparseactions import ArgumentParser,IterFilesAction
	from utilities.fileutilities import iter_files

	parser = ArgumentParser()
	parser.add_argument('-s','--source',action=IterFilesAction,suffix='.ann',default=iter_files('.',suffix='.ann'),help="File or directory to reset. Defaults to current directory.")
	parser.add_argument('-f','--force',action='store_true',help="Force reset without query.")
	args = parser.parse_args()

	args.source = list(args.source)

	def query_user():
		print(f'Are you sure you want to delete contents of all ({len(args.source)}) .ann files? [y/n]',end=' ')
		if input().lower() == 'y':
			return True
		else:
			return False

	if args.force or query_user():
		for i in args.source:
			open(i,'w').close()
