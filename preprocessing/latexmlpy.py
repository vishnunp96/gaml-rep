from utilities.systemutilities import console
from utilities.terminalutils import eprint
import utilities.stringhandler as stringhandler
from preprocessing.texthandler import normalise, isEmpty
from preprocessing.mathtokenise import regularize_math
import re
from lxml import etree as let
from utilities.gxml import fastxmliter

global latexmlignoretags
global latexmlremovetags
global latexmlnoprettify
global latexmlfollownewline
global latexmlprettifyspecial

def format(element, depth=0):
	""" Format ElementTree into a human-readable string, preserving element structure and details. """

	if depth != 0:
		display = '\n'
	else:
		display = ''

	display += '\t'*depth + element.tag + ' ' + repr(element.attrib)
	#if element.text:
	display += '\n' + '\t'*(depth+1) + 'Text (' + element.tag + '): ' + re.sub('\n','\n'+'\t'*(depth+2),repr(element.text))
	for child in element:
		display += format(child, depth + 1)
	#if element.tail:
	display += '\n' + '\t'*(depth+1) + 'Tail (' + element.tag + '): ' + re.sub('\n','\n'+'\t'*(depth+2),repr(element.tail))

	return display

def overlappingspans(span,possibles):
	"""
		Return partial and full overlaps between \'span\' and \'possibles\'.
		This includes instances where \'possibles\' do not cover full reach of \'span\'.
	"""

	#print(f'Span: {span}\nPossibles: {possibles}')

	index = 0
	while index < len(possibles) and span[0] >= possibles[index][1]:
		index += 1

	#print(f'First index: {index}')

	if index < len(possibles) and span[1] >= possibles[index][0]:
		overlap = [possibles[index]]
	else:
		return []

	while span[1] > possibles[index][1]:
		index += 1
		if index >= len(possibles): break
		overlap.append(possibles[index])

	#print(f'Overlap: {overlap}')

	return overlap

def get_local_overlap(span,possibles):
	"""
		Return overlapping spans where character positions are relative
		to the text within the element, not the overall text.
	"""
	local_overlap = []
	for overlap in overlappingspans(span,possibles):
		local_overlap.append((max(0,span[0]-overlap[0]), min(span[1]-overlap[0],overlap[1]-overlap[0]), overlap[2], overlap[3]))
	return local_overlap

def get_local_overlap_from_multiple(spans,possibles):
	result = []
	for span in spans:
		result += get_local_overlap(span,possibles)
	return result

def appendspan(spans,buffer_char_count,nextlength,path,location):
	if spans:
		start = spans[-1][1] + buffer_char_count
		spans.append((start, start + nextlength, path, location))
		#print(f'Span appended: {spans[-1]}')
	else:
		start = 0 + buffer_char_count
		spans.append((start,start+nextlength,path,location))
		#print(f'New span list: {spans[-1]}')
	return spans

def tostring(element):
	text,spans = __recursive_tostring(element)
	spans = [s for s in spans if s[2] != 'NOTAPATH']
	return text,spans

def __recursive_tostring(element, spans=None, pref_sep=False):
	"""
	Format LateXML ElementTree object into organised text for machine reading.
	Input is expected to be a 'prettified' tree, and hence output will be 
	tokenized and sentence split. Also returns list of spans in text, along
	with path of element, and span location (text/tail) in element.
	"""

	if spans is None:
		spans = []

	path = element.getroottree().getpath(element)

	if element.tag == 'tabular':
		response = '\n[Table Here]\n'
		appendspan(spans,0,len(response),path,'')
		return response,spans

	text = ''
	separator = ' '
	#seplen = 1

	if element.tag not in latexmlignoretags:
		if not isEmpty(element.text):
			add_sep = pref_sep
			buffer_chars = (separator if add_sep else '')
			text += buffer_chars + element.text
			appendspan(spans,len(buffer_chars),len(element.text),path,'text')
		for child in element:

			## If there is text and the last character isn't a space, add a space
			## Or if there is no text and we were told to add a space, add one
			#(text and not text[-1].isspace()) or (not text and pref_sep)

			childtext, _ = __recursive_tostring(child,spans=spans,pref_sep=(text and not text[-1].isspace()) or (not text and pref_sep))
			if not isEmpty(childtext):
				text += childtext
	if not isEmpty(element.tail):

		## If there is text and the last character isn't a space, add a space
		## Or if there is no text and we were told to add a space, add one
		#(text and not text[-1].isspace()) or (not text and pref_sep)

		add_sep = (text and not text[-1].isspace()) or (not text and pref_sep)
		buffer_chars = (separator if add_sep else '')
		text += buffer_chars + element.tail
		appendspan(spans,len(buffer_chars),len(element.tail),path,'tail')

	##text = '\n'.join([l.strip() for l in text.split('\n')])

	if element.tag in latexmlfollownewline:
		if spans:
			text += '\n\n'
			appendspan(spans,0,2,'NOTAPATH','')

	return text, spans

def positer(s,char):
	next = s.find(char)
	while next != -1:
		yield next
		next = s.find(char,next+1)

def spaniter(s,delimiter):
	breaks = [i for i in positer(s,delimiter)]
	if breaks:
		yield 0,breaks[0]
		for i in range(0,len(breaks)-1):
			yield breaks[i]+1,breaks[i+1]
		yield breaks[-1]+1,len(s)
	else:
		yield 0,len(s)

def tolines(element, include_empty = False):

	text,spans = tostring(element)

	index = 0

	#print()
	#print('tolines:')
	#print(f'Spans: {spans}')
	#print('Text: ' + text)

	for start,end in spaniter(text,'\n'):
		line = text[start:end]

		#print(f'Line ({start},{end}): {line}')

		if line: ## Just in case the line is of zero length
			while start >= spans[index][1]:
				index += 1

			linespans = [spans[index]]

			while end > spans[index][1]:
				index += 1
				linespans.append(spans[index])

			if not isEmpty(line) or include_empty:
				yield line,(start,end),linespans

def elemiter(filepath, tag):
	for event,elem in fastxmliter(filepath, events=("end",), tag=tag):
		yield elem

def tostringiter(filepath, tag):
	for elem in elemiter(filepath, tag):
		yield tostring(elem)

def tolinesiter(filepath, tag):
	for elem in elemiter(filepath, tag):
		for elem_tolines in tolines(elem):
			yield elem_tolines

def prettify(element):
	""" Refactor LateXML ElementTree object such that texts and tails of elements are normalised and tokenized. """

	if element.tag in latexmlprettifyspecial:
		latexmlprettifyspecial[element.tag](element)
		return
	else:
		makepretty = (element.tag not in latexmlnoprettify)

		if makepretty:
			if element.text:
				element.text = normalise(element.text)

		for child in element:
			prettify(child)
			if child.tail and makepretty:
				child.tail = normalise(child.tail)

		return

def prettify_single_line(element):
	""" Refactor LateXML ElementTree element such that texts and tails of element and all subelements are given on a single line. """

	if element.text:
		element.text = re.sub('\s+', ' ', element.text.strip())

	for child in element:
		prettify_single_line(child) ## Should this recurse to prettifyxml...?
		if child.tail:
			child.tail = re.sub('\s+', ' ', child.tail.strip())

def prettify_math(element):
	""" Refactor LateXML ElementTree Math element using the mathtokenise.regularize_math function. """
	element.text = regularize_math(element.attrib['tex'])

newlinere = re.compile(r'^\n')
def remove_leading_newlines(element):
	## Should this remove *all* leading newlines, or just the first one?
	if element.text:
		element.text = newlinere.sub('',element.text,count=0)
	for child in element:
		remove_leading_newlines(child)
		if child.tail:
			child.tail = newlinere.sub('',child.tail,count=0)

def opentex(texfile, verbose=False, timeout=300):
	"""
	Open a .tex file as a LateXML ElementTree.
	Returns the root element of the resulting tree, or None if parse failed.
	Defaults to timeout after 5 minutes.
	"""

	success, output, error = console('latexml --quiet --nocomments --noparse --includestyles ' + texfile + ' | latexmlpost --format=xml -', timeout=timeout)
	## Option here for using --documentid=id for including arXiv ID in xml document

	try:
		xmlstring = stringhandler.decode(output, return_encoding=False, fallback_errors='replace')

		#print(stringhandler.decode(error))

		xmlstring = re.sub(' xmlns="[^"]+"', '', xmlstring)
		xmlstring = re.sub('xml:', '', xmlstring)

		xmlbytes = xmlstring.encode(encoding='utf-8',errors='replace')

		#root = let.fromstring(xmlstring)
		root = let.XML(xmlbytes)

		for tag in latexmlremovetags:
			removetag(root, tag)

		remove_leading_newlines(root)

		prettify(root)

		return root

	except (KeyboardInterrupt, SystemExit):
		raise
	except Exception as e:
		## Something has gone wrong...
		if verbose: eprint(e)
		pass

	## If all else fails...
	if verbose: print(stringhandler.decode(error))
	return None

def openxml(filepath,getroot=True):
	if getroot:
		return let.parse(filepath).getroot()
	else: # Else return ElementTree object
		return let.parse(filepath)


def savexml(element, filepath):
	""" Save an ElementTree element to a given filepath. """
	tree = let.ElementTree(element)
	tree.write(filepath)

def removetag(element, tag):
	""" Remove all elements with a given tag from an ElementTree structure. """
	for child in element:
		if child.tag == tag:
			element.remove(child)
		else:
			removetag(child, tag)

def sections(root):
	""" Returns list of sections/abstract in a LateXML ElementTree structure.  """

	contents = [root.find('abstract')]
	contents += [s for s in root.findall('.//section')]

	return contents



### Global variables (need to be here to allow for references to functions defined above)
latexmlignoretags = ['bibtag']
latexmlremovetags = ['XMath']
latexmlnoprettify = ['bibtag', 'tabular', 'table']
latexmlfollownewline = ['title', 'p', 'bibitem', 'tabular', 'table']
latexmlprettifyspecial = {'bibblock': prettify_single_line, 'Math': prettify_math}



if __name__ == "__main__":

	from utilities.argparseactions import ArgumentParser,FileAction
	from utilities.fileutilities import changeext

	parser = ArgumentParser(description='Convert a .tex file into an ElementTree structure, saving and/or displaying the result in various formats.')
	parser.add_argument('texfile',action=FileAction, mustexist=True, help='Path to tex file to parse.')
	parser.add_argument('-x','--xmlfile',action=FileAction, mustexist=False, default=('texfile', lambda s: changeext(s,'.xml')), help='Path in which to save xml output.')
	parser.add_argument('-d','--xmldispfile',action=FileAction, mustexist=False, help='Path in which to save neatened xml output (not suitable for reconstruction into text format).')
	parser.add_argument('-t','--txtfile',action=FileAction, mustexist=False, help='Path in which to save text form of output.')
	parser.add_argument('-v','--verbose',action="store_true", help='Output human-readable tree and text format.')
	args = parser.parse_args()

	root = opentex(args.texfile)
	xmlstr = tostring(root)

	if args.verbose or args.xmldispfile:
		display = format(root)

	if args.verbose:
		print(display)
		print('\n')
		print(xmlstr)

	tree = let.ElementTree(root)
	tree.write(args.xmlfile)

	if args.txtfile:
		with open(args.txtfile, 'w') as f:
			f.write(xmlstr)
	if args.xmldispfile:
		with open(args.xmldispfile, 'w') as f:
			f.write(display)

