import subprocess
import sys
import os

#filepath = sys.argv[1]
#filename = os.path.basename(filepath)
#directorypath = os.path.dirname(os.path.abspath(filepath))
#
#targetname = sys.argv[2]
#
#
#os.chdir(directorypath)
#os.system("detex -e , " + filename + " | python /home/tom/Documents/s3Test/detexTokenize.py " + " > " + targetname)


sourcepath = os.path.abspath(sys.argv[1])
targetpath = os.path.abspath(sys.argv[2])

encodings = ['UTF-8']
def readFile(filepath):
	for enc in encodings:
		try:
			with open(filepath, encoding=enc) as file:
				text = file.read()
			return text
		except Exception:
			# This encoding failed. Try another.
			pass
	# All encodings failed
	print("Could not read: " + filepath)
	return ''

def processDirectory(directory):

	#print("Directory: " + directory)

	contents = [directory + os.sep + n for n in os.listdir(directory)]

	doctexfiles = []

	for c in contents:
		if os.path.isdir(c):
			processDirectory(c)
		else: # c is a file
			processFile(c)


def processFile(filepath):
	
	#print("File: " + filepath)

	if filepath.endswith(".tex"):
		#print("filepath: " + filepath)
		if "\\begin{document}" in readFile(filepath):
			
			origin = os.path.dirname(filepath)
			sourcefilename = os.path.basename(filepath)

			destination = os.path.dirname(filepath.replace(sourcepath,targetpath))
			targetfilepath = destination + os.sep + os.path.splitext(sourcefilename)[0] + ".txt"

			#print("origin: " + origin)
			#print("sourcefilename: " + sourcefilename)
			#print("destination: " + destination)
			#print("targetfilepath: " + targetfilepath)

			if not os.path.exists(destination):
				os.makedirs(destination)

			#print("Moving to: " + origin)
			os.chdir(origin)
			os.system("detex -e , " + sourcefilename + " | python /home/tom/Documents/s3Test/detexTokenize.py " + " > " + targetfilepath)


processDirectory(sourcepath)










