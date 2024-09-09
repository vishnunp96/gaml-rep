import sys
from utilities.systemutilities import console
from utilities import stringhandler
from preprocessing.findTexDocuments import prettify

for index, cmd in enumerate(sys.argv[1:]):
	print('Command ' + str(index) + ': ' + cmd)
	success, stdout, stderr = console(cmd)
	print('Completed:', success)
	print('Output:', stdout)
	print('Error:', stderr)
	print('End command ' + str(index))
