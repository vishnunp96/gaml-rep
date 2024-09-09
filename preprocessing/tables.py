import pandas
import gaml.preprocessing.latexmlpy as latexml
import numpy
import copy
from gaml.utilities.extracollections import List2D

class Table(pandas.DataFrame):
	def __init__(self, data=None, index=None, columns=None, dtype=None, caption=None, copy=False):
		super(Table,self).__init__(data=data,index=index,columns=columns,dtype=dtype,copy=copy)
		self.caption = caption

	def __repr__(self):
		data_repr = super(Table,self).__repr__()
		if self.caption:
			return 'Caption: ' + self.caption + '\n' + data_repr
		else:
			return data_repr

	def __str__(self):
		data_str = super(Table,self).__str__()
		if self.caption:
			return 'Caption: ' + self.caption + '\n' + data_str
		else:
			return data_str

border_keys = {'l':1, 't':2, 'b':3, 'r':4, ' ':0}
def sort_border_key(c):
	return border_keys[c]

def process_table(table_elem, verbose=False):
	rows,borders,caption = preprocess_table_elem(table_elem,verbose=verbose)
	table = format_table_parts(rows,borders,caption)
	return table

class ExtendingList(list):
	def __init__(self,*args,factory=int,**kwargs):
		super(ExtendingList, self).__init__(*args,**kwargs)
		self.factory = factory
	def __setitem__(self,index,value):
		assert index >= 0
		if index >= len(self):
			self.extend([self.factory()]*(1+index-len(self)))
		super(ExtendingList, self).__setitem__(index,value)
	def __getitem__(self,index):
		assert index >= 0
		if index >= len(self):
			self.extend([self.factory()]*(1+index-len(self)))
		return super(ExtendingList, self).__getitem__(index)

class ParsingError(Exception): pass
class TranscriptionError(Exception): pass
class NormalisationError(Exception): pass
class FormattingError(Exception): pass

def preprocess_table_elem(table_elem, verbose=False):

	if any(table_elem.findall('.//ERROR')):
		raise ParsingError()

	caption_elem = table_elem.find('.//caption')
	caption = latexml.tostring(caption_elem)[0] if caption_elem is not None else ''

	try:
		rows = []
		row_borders = []

		for_below = ExtendingList(factory=lambda: [0,''])

		for tr in table_elem.findall('.//tr'): # Each row
			columns = []
			column_borders = []

			i = 0 # Column index
			for td in tr.findall('.//td'): # Each column in this row

				val = latexml.tostring(td)[0].encode('ascii',errors='replace').decode('ascii')
				colspan = eval(td.attrib.get('colspan',default='1'))
				rowspan = eval(td.attrib.get('rowspan',default='1'))
				border = td.attrib.get('border',default='')

				## Deal with columns from rowspans above
				while for_below[i][0] > 0:
					columns.append(None)
					above_border = for_below[i][1]
					if for_below[i][0] > 1: # This is not the bottom of the cell
						column_borders.append(''.join([i*above_border.count(i) for i in 'lr']))
					else: # This is the bottom of the cell
						column_borders.append(''.join([i*above_border.count(i) for i in 'lbr']))
					i += 1

				if rowspan > 1:
					if colspan>1:
						for j in range(i,i+colspan):
							for_below[j][0] += rowspan
							if j==i:
								for_below[j][1] = border.replace('l','')
							elif j==(i+colspan-1):
								for_below[j][1] = border.replace('r','')
							else:
								for_below[j][1] = ''.join([i*border.count(i) for i in 'tb'])
					else: # colspan==1
						for_below[i][0] += rowspan
						for_below[i][1] = border
					border = border.replace('b','') # Do not put the bottom lines on this cell, as it adjoins another

				columns.append(val if val else None)
				#if colspan>1 and rowspan>1:
				#	raise ValueError('Extends over columns and rows?')

				if colspan > 1:
					columns.extend([None]*(colspan-1))

					column_borders.append(''.join([i*border.count(i) for i in 'ltb']))
					column_borders.extend((colspan-2)*[''.join([i*border.count(i) for i in 'tb'])])
					column_borders.append(''.join([i*border.count(i) for i in 'tbr']))
				else:
					column_borders.append(''.join(sorted(border,key=sort_border_key)).strip())

				i += colspan
			for_below = ExtendingList([[max(0,count-1),b] for count,b in for_below],factory=lambda: [0,''])

			#if any(columns):
			rows.append(columns)
			row_borders.append(column_borders)
			#else:
				# Any behaviour needed if we're ignoring a row?
				#pass

		rows = List2D(rows)
		borders = List2D(row_borders)
	except Exception:
		raise TranscriptionError()

	if verbose:
		print(rows)
		print(borders)

	## Fill in short rows
	# Should be obselete now that we have the colspan/rowspan feature
	#rows,borders = fill_short_rows(rows,borders)

	try:
		borders = normalise_borders(borders)

		## Remove empty columns
		for j in range(rows.shape[1]-1,-1,-1): # For each column
			if not any(rows[:,j]):
				del rows[:,j]
				## Deal with borders
				if j>0:
					for i,bij in enumerate(borders[:,j]):
						borders[i,j-1] += 'r'*bij.count('r')
				del borders[:,j]

		## Remove empty rows
		for i in range(rows.shape[0]-1,-1,-1): # For each row
			if not any(rows[i,:]):
				del rows[i]
				## Deal with borders
				if i>0:
					for j,bij in enumerate(borders[i,:]):
						borders[i-1,j] += 'b'*bij.count('b')
				del borders[i]

		## Better way than just repeating this?
		borders = normalise_borders(borders)
	except Exception:
		print(caption)
		print(rows)
		print(borders)
		raise NormalisationError()

	return rows,borders,caption

def fill_short_rows(rows,borders):
	if any(rows) and len(set(len(i) for i in rows.iterrows()))!=1:
		maxlen = max(len(i) for i in rows.iterrows())
		for i in range(len(rows.data)):
			if len(rows.data[i])!=maxlen:
				rows.data[i].extend([None]*(maxlen-len(rows.data[i])))
				borders.data[i].extend([borders.data[i][-1]]*(maxlen-len(borders.data[i])))
		rows = List2D(rows.data)
		borders = List2D(borders.data)
		#print(rows)
		#print(borders)
	return rows,borders

def format_table_parts(rows,borders,caption):
	try:
		if any(rows):
			table = tryall([simple_pattern,single_horizontal,line_indexcolumn,horizontal_subdivisions,default_table],rows,borders,caption)
			if table is not None:
				table.fillna(value=numpy.nan,inplace=True)
				return table
		return default_table(rows,borders,caption)
	except Exception:
		print(caption)
		print(rows)
		print(borders)
		raise FormattingError()

def no_borders(borders):
	return not any(borders)
##### These don't really have any meaning anymore
#def all_single_vertical(borders):
#	return all(i=='r' for i in borders[:,:-1]) and not any(borders[:,-1])
#def all_single_horizontal(borders):
#	return all(i=='b' for i in borders[:-1,:]) and not any(borders[-1,:])

## This any use?
#def no_double_lines(borders):
#	return not any('rr' in i or 'bb' in i for i in borders)

def horizontals(borders):
	return [min(i.count('b') for i in row) for row in borders.iterrows()]
def verticals(borders):
	return [min(i.count('r') for i in column) for column in borders.itercolumns()]

def kinds_horizontal(borders):
	sets = [set(i.count('b') for i in row) for row in borders.iterrows()]
	return [len(s-{0}) if s=={0} else len(s) for s in sets]
	#return [len(set([i.count('b') for i in row])-{0}) for row in borders.iterrows()]
def kinds_vertical(borders):
	sets = [set(i.count('r') for i in col) for col in borders.itercolumns()]
	return [len(s-{0}) if s=={0} else len(s) for s in sets]
	#return [len(set([i.count('r') for i in column])-{0}) for column in borders.itercolumns()]

def has_floating(borders): # Returns True if there are borders which do not span entire table (horizontally or vertically)
	return any(i>1 for i in kinds_horizontal(borders)) or any(i>1 for i in kinds_vertical(borders))

def concat_first_n_rows(rows,n):
	columns = rows.data[0]
	if n>1:
		for (row,col),val in rows[1:n,:].nditer():
			columns[col] = (columns[col] if columns[col] else '') + ('\n' if columns[col] and val else '') + (val if val else '')
	return columns


## Default behaviour
def default_table(rows,borders,caption):
	return None
	#return Table(rows.data,caption=caption)

## If no border present, or only single vertical/horizontal lines, assume first row is column headings, and rest is data
def simple_pattern(rows,borders,caption):
	# Cannot have all_single_x anymore.
	#if no_borders(borders) or all_single_vertical(borders) or all_single_horizontal(borders):
	if no_borders(borders):
		return Table(rows[1:,:].data,columns=rows.data[0],caption=caption)

## If there is a single horizontal column near the top, and no verticals
def single_horizontal(rows,borders,caption):
	if not has_floating(borders):
		hs = horizontals(borders)
		if sum(i>0 for i in hs)==1 and sum(verticals(borders))==0:
			hindex = next(i for i,b in enumerate(hs) if b>0) + 1
			column_names = concat_first_n_rows(rows,hindex)
			return Table(rows[hindex:,:].data,columns=column_names,caption=caption)

## If there is demarkation for index and columns, and no other lines present
## This should probably be extended to account for multi-line indices or headings
def line_indexcolumn(rows,borders,caption):
	if not has_floating(borders):
		## The logic here needs improving
		if all('b' in i for i in borders[0,:]) and all('r' in i for i in borders[:,0]) and no_borders(borders[1:,1:]):
			return Table(rows[1:,1:].data,columns=rows[0,1:].data[0],index=pandas.Index(rows[1:,0].flatten(),name=rows[0,0]),caption=caption)

## If multiple horizontal lines are present, assume first denotes headings, and the rest denote subdivisions
## If a subdivision begins with a single-entry line, assume this is the name of the subdivision
def horizontal_subdivisions(rows,borders,caption):
	if not has_floating(borders):
		hs = horizontals(borders)
		if sum(bool(i) for i in hs)>1 and sum(verticals(borders))==0:
			h_indexes = [i+1 for i,b in enumerate(hs) if b>0] # Indexes of the first row in each subdivision
			hindex = h_indexes[0] # This corresponds to the index of the first row after the column headings
			column_names = concat_first_n_rows(rows,hindex)

			# Only first entry in h_index rows has a value, then assume it is group title
			if all(bool(rows[i,0]) and sum(bool(j) for j in rows.data[i][1:])==0 for i in h_indexes):
				index = []
				for i,h in enumerate(h_indexes):
					index += [next(s for s in rows.data[h] if s)]*((h_indexes[i+1] if (i+1)<len(h_indexes) else rows.shape[0])-h-1) # Minus one because we will soon remove this header row

				rows = copy.deepcopy(rows) # So the original is safe for user, if needed
				for i in reversed(h_indexes):
					del rows[i]
			else:
				## Find another way to determine subdivision headings?
				index = None

			table = Table(rows[hindex:,:].data,columns=column_names,index=index,caption=caption)
			table.dropna(axis='columns',how='all',inplace=True)
			return table

## How do we deal with empty rows?

def tryall(funcs,*args,**kwargs):
	for f in funcs:
		result = f(*args,**kwargs)
		if result is not None:
			return result
	return None

def iter_array(a):
	it = numpy.nditer(a,flags=['multi_index'])
	while not it.finished:
		yield it.multi_index, it[0].item()
		it.iternext()

def normalise_borders(borders): # borders should be List2D
	''' Have each entry be aware of all of its borders (i.e. those originally belonging to separate cells). '''

	if len(borders)==0: return borders
	normalised = copy.deepcopy(borders)
	h,w = borders.shape
	for (row,column),border in borders.nditer():
		t = border.count('t')
		l = border.count('l')
		if row>0 and t>0:
			idx = (row-1,column)
			normalised[idx] += 'b'*t
		if column>0 and l>0:
			idx = (row,column-1)
			normalised[idx] += 'r'*l
	for index,border in normalised.nditer():
		row,column = index
		normalised[index] = normalised[index].replace('t','')
		normalised[index] = normalised[index].replace('l','')
		if row==h-1: normalised[index] = normalised[index].replace('b','')
		if column==w-1: normalised[index] = normalised[index].replace('r','')
		normalised[index] = ''.join(sorted(normalised[index],key=sort_border_key))

	if w > 1:
		rcount = min(i.count('r') for i in normalised[:,:-1])
		if rcount>0:
			normalised.apply(lambda s: s.replace('r','',rcount))
	if h > 1:
		bcount = min(i.count('b') for i in normalised[:-1,:])
		if bcount>0:
			normalised.apply(lambda s: s.replace('b','',bcount))

	return normalised

if __name__ == '__main__':

	from gaml.utilities.argparseactions import ArgumentParser,IterFilesAction
	from gaml.metadata.oaipmh import arXivID_from_path
	from collections import defaultdict

	parser = ArgumentParser(description="Run through tables.")
	parser.add_argument("sources",action=IterFilesAction, recursive=True, suffix='.xml', help='Path to xml source(s).')
	parser.add_argument("-S","--summary",action='store_true',help='Provide summary of successes, rather than iterate through tables.')
	args = parser.parse_args()


	#for path in args.sources:
	#	for elem in latexml.elemiter(path,'table'):
	#		try:
	#			if any(elem.findall('.//td[@colspan]')):
	#				print(latexml.format(elem).encode('ascii',errors='replace').decode('ascii'))
	#				print(path)
	#				rows,borders,caption = preprocess_table_elem(elem,True)
	#				print(caption)
	#				print(pandas.DataFrame(borders.data))
	#				print(pandas.DataFrame(rows.data))
	#				table = format_table_parts(rows,borders,caption)
	#				print(table)
	#				input()
	#		except UnicodeEncodeError as e:
	#			pass

	if not args.summary:
		for path in args.sources:
			for elem in latexml.elemiter(path,'table'):
				try:
					print(latexml.format(elem).encode('ascii',errors='replace').decode('ascii'))
					print(path)
					rows,borders,caption = preprocess_table_elem(elem,True)
					print(caption)
					print(pandas.DataFrame(borders.data))
					table = format_table_parts(rows,borders,caption)
					print(table)

					#rows,borders,caption = preprocess_table_elem(elem)
					#if has_floating(borders):
					#	print(latexml.format(elem).encode('ascii',errors='replace').decode('ascii'))
					#	rows,borders,caption = preprocess_table_elem(elem,True)
					#	print(path)
					#	print(caption)
					#	print(pandas.DataFrame(borders.data))
					#	print(pandas.DataFrame(rows.data))
					#	input()
				except UnicodeEncodeError as e:
					print(e)
					pass
				input()
	else:
		tablesuccesses = 0
		tablefailures = 0
		papersuccesses = set()
		paperfailures = set()
		errors = defaultdict(int)
		for path in args.sources:
			arXiv = arXivID_from_path(path)
			for elem in latexml.elemiter(path,'table'):
				table = None
				try:
					rows,borders,caption = preprocess_table_elem(elem)
					table = format_table_parts(rows,borders,caption)
				except (ParsingError,TranscriptionError,NormalisationError,FormattingError) as e:
					errors[type(e)] += 1
					#print(latexml.format(elem).encode('ascii',errors='replace').decode('ascii'))
					#print(path)
				if table is None:
					tablefailures += 1
					paperfailures.add(arXiv)
				else:
					tablesuccesses += 1
					papersuccesses.add(arXiv)

		allpapers = papersuccesses | paperfailures
		mixedpapers = papersuccesses & paperfailures

		print(f'Successes: {tablesuccesses} from {len(papersuccesses)} papers, with {tablefailures} failures from {len(paperfailures)} papers.')
		print(f'{len(allpapers)} total papers. {len(mixedpapers)} mixed results.')
		print(f'Tables: {100*tablesuccesses/(tablesuccesses+tablefailures):.1f}%, Papers: {100*len(papersuccesses-mixedpapers)/(len(allpapers)):.1f}% ({100*(len(papersuccesses)/len(allpapers)):.1f}% with mixed)')

		print(errors)
