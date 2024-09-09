import sqlite3
import pandas
import re

import numpy
from scipy.special import erfinv

from parsing import parse_symbol,parse_measurement
from annotations.bratutils import Standoff

### Need code to create tables
def init(cursor):

	## Needed for some strange reason
	cursor.execute('PRAGMA foreign_keys = 1')
	cursor.execute('PRAGMA busy_timeout = 120000') # 2 min

	# Table for storing successful completion of individual processing runs
	cursor.execute('''
		CREATE TABLE prediction_runs
			(
				source TEXT PRIMARY KEY NOT NULL,
				status TEXT NOT NULL CHECK(status IN ('STARTED','COMPLETED'))
			)''')

	# ParameterName table
	cursor.execute('''
		CREATE TABLE parameter_names
			(
				name TEXT PRIMARY KEY NOT NULL
			)''')

	# ParameterSymbol table
	cursor.execute('''
		CREATE TABLE parameter_symbols
			(
				symbol TEXT PRIMARY KEY NOT NULL,
				symbol_norm TEXT
			)''')

	# Name relation table
	cursor.execute('''
		CREATE TABLE names
			(
				name_id INTEGER PRIMARY KEY NOT NULL,
				name TEXT NOT NULL,
				symbol TEXT NOT NULL,
				FOREIGN KEY (name) REFERENCES parameter_names(name),
				FOREIGN KEY (symbol) REFERENCES parameter_symbols(symbol),
				UNIQUE(name,symbol)
			)''')

	# ParameterName paper occurences
	cursor.execute('''
		CREATE TABLE parameter_name_occurences
			(
				entity_id INTEGER PRIMARY KEY NOT NULL,
				name TEXT NOT NULL,
				start INTEGER NOT NULL,
				end INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (name) REFERENCES parameter_names(name),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(name,start,end,arxiv_id)
			)''')

	# ParameterSymbol paper occurences
	cursor.execute('''
		CREATE TABLE parameter_symbol_occurences
			(
				entity_id INTEGER PRIMARY KEY NOT NULL,
				symbol TEXT NOT NULL,
				start INTEGER NOT NULL,
				end INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (symbol) REFERENCES parameter_symbols(symbol),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(symbol,start,end,arxiv_id)
			)''')

	# Name relation paper occurences (needed to distinguish repeated uses of same symbol)
	cursor.execute('''
		CREATE TABLE name_occurences
			(
				relation_id INTEGER PRIMARY KEY NOT NULL,
				name_id INTEGER NOT NULL,
				start_id INTEGER NOT NULL,
				end_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (name_id) REFERENCES names(name_id),
				FOREIGN KEY (start_id) REFERENCES parameter_name_occurences(entity_id),
				FOREIGN KEY (end_id) REFERENCES parameter_symbol_occurences(entity_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(name_id,start_id,end_id,arxiv_id)
			)''')

	# MeasuredValue and Constraint table (entity ID/name & symbol, string, ConfidenceLimit, SeparatedUncertainty)
	cursor.execute('''
		CREATE TABLE measurements
			(
				value_id INTEGER PRIMARY KEY NOT NULL,
				value TEXT NOT NULL,
				start INTEGER NOT NULL,
				end INTEGER NOT NULL,
				bound TEXT NOT NULL CHECK(bound IN ('C','U','L')),
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(start,end,arxiv_id)
			)''')
			# confidence TEXT
			# separated_uncertainty TEXT

	# ConfidenceLimit entity table
	cursor.execute('''
		CREATE TABLE confidence_limits
			(
				entity_id INTEGER PRIMARY KEY NOT NULL,
				confidence TEXT NOT NULL,
				confidence_norm REAL NOT NULL CHECK(confidence_norm > 0),
				start INTEGER NOT NULL,
				end INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(confidence,start,end,arxiv_id)
			)''')
	# Confidence relation tables
	cursor.execute('''
		CREATE TABLE value_confidences
			(
				value_id INTEGER NOT NULL,
				confidence_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (value_id) REFERENCES measurements(value_id),
				FOREIGN KEY (confidence_id) REFERENCES confidence_limits(entity_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(value_id,confidence_id)
			)''')

	# Measurement relation tables
	cursor.execute('''
		CREATE TABLE symbol_measurements
			(
				symbol TEXT NOT NULL,
				symbol_id INTEGER NOT NULL,
				value_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (symbol) REFERENCES parameter_symbols(symbol),
				FOREIGN KEY (symbol_id) REFERENCES parameter_symbol_occurences(entity_id),
				FOREIGN KEY (value_id) REFERENCES measurements(value_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(symbol,symbol_id,value_id)
			)''')
	cursor.execute('''
		CREATE TABLE name_measurements
			(
				name TEXT NOT NULL,
				name_id INTEGER NOT NULL,
				value_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (name) REFERENCES parameter_names(name),
				FOREIGN KEY (name_id) REFERENCES parameter_name_occurences(entity_id),
				FOREIGN KEY (value_id) REFERENCES measurements(value_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(name,name_id,value_id)
			)''')

	# Objects table (Object ID, name)
	cursor.execute('''
		CREATE TABLE objects
			(
				object TEXT PRIMARY KEY NOT NULL
			)''')
	# Object paper occurences
	cursor.execute('''
		CREATE TABLE object_occurences
			(
				entity_id INTEGER PRIMARY KEY NOT NULL,
				object TEXT NOT NULL,
				start INTEGER NOT NULL,
				end INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (object) REFERENCES objects(object),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(object,start,end,arxiv_id)
			)''')

	# Property name/symbol relation tables
	cursor.execute('''
		CREATE TABLE object_property_names
			(
				property_name_id INTEGER PRIMARY KEY NOT NULL,
				object TEXT NOT NULL,
				name TEXT NOT NULL,
				FOREIGN KEY (object) REFERENCES objects(object),
				FOREIGN KEY (name) REFERENCES parameter_names(name),
				UNIQUE(object,name)
			)''')
	cursor.execute('''
		CREATE TABLE object_property_symbols
			(
				property_symbol_id INTEGER PRIMARY KEY NOT NULL,
				object TEXT NOT NULL,
				symbol TEXT NOT NULL,
				FOREIGN KEY (object) REFERENCES objects(object),
				FOREIGN KEY (symbol) REFERENCES parameter_symbols(symbol),
				UNIQUE(object,symbol)
			)''')

	# Property name/symbol occurences relation tables
	cursor.execute('''
		CREATE TABLE property_name_occurences
			(
				property_name_id INTEGER NOT NULL,
				start_id INTEGER NOT NULL,
				end_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (property_name_id) REFERENCES object_property_names(property_name_id),
				FOREIGN KEY (start_id) REFERENCES object_occurences(entity_id),
				FOREIGN KEY (end_id) REFERENCES parameter_name_occurences(entity_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(property_name_id,start_id,end_id,arxiv_id)
			)''')
	cursor.execute('''
		CREATE TABLE property_symbol_occurences
			(
				property_symbol_id INTEGER NOT NULL,
				start_id INTEGER NOT NULL,
				end_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (property_symbol_id) REFERENCES object_property_symbols(property_symbol_id),
				FOREIGN KEY (start_id) REFERENCES object_occurences(entity_id),
				FOREIGN KEY (end_id) REFERENCES parameter_symbol_occurences(entity_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(property_symbol_id,start_id,end_id,arxiv_id)
			)''')

	# Property measurement relation tables
	cursor.execute('''
		CREATE TABLE property_measurements
			(
				object TEXT NOT NULL,
				object_id INTEGER NOT NULL,
				value_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (object) REFERENCES objects(object),
				FOREIGN KEY (object_id) REFERENCES object_occurences(entity_id),
				FOREIGN KEY (value_id) REFERENCES measurements(value_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(object_id,value_id)
			)''')

	# Papers table (paper ID, date)
	cursor.execute('''
		CREATE TABLE papers
			(
				arxiv_id TEXT PRIMARY KEY NOT NULL,
				abstract TEXT NOT NULL,
				date TEXT NOT NULL
			)''')

	# Definitions table (Entity ID, string)
	## Table of equations
	cursor.execute('''
		CREATE TABLE equations
			(
				equation TEXT PRIMARY KEY NOT NULL
			)''')
	# Equation paper occurences
	cursor.execute('''
		CREATE TABLE equation_occurences
			(
				entity_id INTEGER PRIMARY KEY NOT NULL,
				equation TEXT NOT NULL,
				start INTEGER NOT NULL,
				end INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (equation) REFERENCES equations(equation),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(equation,start,end,arxiv_id)
			)''')
	## Tables for symbol/name equation relations
	cursor.execute('''
		CREATE TABLE symbol_definitions
			(
				symbol_definition_id INTEGER PRIMARY KEY NOT NULL,
				symbol TEXT NOT NULL,
				equation TEXT NOT NULL,
				FOREIGN KEY (symbol) REFERENCES parameter_symbols(symbol),
				FOREIGN KEY (equation) REFERENCES equations(equation),
				UNIQUE(symbol,equation)
			)''')
	cursor.execute('''
		CREATE TABLE name_definitions
			(
				name_definition_id INTEGER PRIMARY KEY NOT NULL,
				name TEXT NOT NULL,
				equation TEXT NOT NULL,
				FOREIGN KEY (name) REFERENCES parameter_names(name),
				FOREIGN KEY (equation) REFERENCES equations(equation),
				UNIQUE(name,equation)
			)''')
	## Occurences of definition relations
	cursor.execute('''
		CREATE TABLE symbol_definition_occurences
			(
				symbol_definition_id INTEGER NOT NULL,
				start_id INTEGER NOT NULL,
				end_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (symbol_definition_id) REFERENCES symbol_definitions(symbol_definition_id),
				FOREIGN KEY (start_id) REFERENCES parameter_symbol_occurences(entity_id),
				FOREIGN KEY (end_id) REFERENCES equation_occurences(entity_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(symbol_definition_id,start_id,end_id,arxiv_id)
			)''')
	cursor.execute('''
		CREATE TABLE name_definition_occurences
			(
				name_definition_id INTEGER NOT NULL,
				start_id INTEGER NOT NULL,
				end_id INTEGER NOT NULL,
				arxiv_id TEXT NOT NULL,
				FOREIGN KEY (name_definition_id) REFERENCES name_definitions(name_definition_id),
				FOREIGN KEY (start_id) REFERENCES parameter_name_occurences(entity_id),
				FOREIGN KEY (end_id) REFERENCES equation_occurences(entity_id),
				FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
				UNIQUE(name_definition_id,start_id,end_id,arxiv_id)
			)''')

	create_or_update_views(cursor)

def create_or_update_views(cursor):
	#### Useful views
	# Get all name occurences in full (name, symbol, and arxiv_id)
	cursor.execute('DROP VIEW IF EXISTS all_names')
	cursor.execute('''
		CREATE VIEW all_names
			(
				name_id,
				name,
				symbol,
				symbol_norm,
				arxiv_id
			)
		AS
			SELECT N.name_id,N.name,N.symbol,S.symbol_norm,O.arxiv_id
			FROM names N
				LEFT OUTER JOIN name_occurences O ON N.name_id = O.name_id
				LEFT OUTER JOIN parameter_symbols S ON N.symbol = S.symbol
		''')

	# Get symbol_measurements with arxiv_id
	cursor.execute('DROP VIEW IF EXISTS symbol_measurements_papers')
	cursor.execute('''
		CREATE VIEW symbol_measurements_papers
			(
				symbol,
				symbol_norm,
				value_id,
				arxiv_id
			)
		AS
			SELECT M.symbol,S.symbol_norm,M.value_id,V.arxiv_id
			FROM symbol_measurements M
				INNER JOIN measured_values V ON M.value_id = V.value_id
				INNER JOIN parameter_symbols S ON M.symbol = S.symbol
		''')

	# Get an expanded symbol_measurements to include symbol and value text, along with id numbers
	cursor.execute('DROP VIEW IF EXISTS symbol_measurements_values')
	cursor.execute('''
		CREATE VIEW symbol_measurements_values
			(
				value_id,
				symbol_id,
				value,
				bound,
				symbol,
				symbol_norm,
				arxiv_id
			)
		AS
			SELECT V.value_id,M.symbol_id,V.value,V.bound,S.symbol,S.symbol_norm,V.arxiv_id
			FROM symbol_measurements M
				INNER JOIN measurements V ON M.value_id = V.value_id
				INNER JOIN parameter_symbols S ON M.symbol = S.symbol
		''')

	# Get an expanded name_measurements to include name and value text, along with id numbers
	cursor.execute('DROP VIEW IF EXISTS name_measurements_values')
	cursor.execute('''
		CREATE VIEW name_measurements_values
			(
				value_id,
				name_id,
				value,
				bound,
				name,
				arxiv_id
			)
		AS
			SELECT V.value_id,M.name_id,V.value,V.bound,M.name,V.arxiv_id
			FROM name_measurements M
				INNER JOIN measurements V ON M.value_id = V.value_id
		''')

	# Get all measurements (name and symbol) in one view
	cursor.execute('DROP VIEW IF EXISTS all_measurements')
	cursor.execute('''
		CREATE VIEW all_measurements
			(
				value_id,
				name_id,
				symbol_id,
				value,
				bound,
				name,
				symbol,
				symbol_norm,
				arxiv_id
			)
		AS
			SELECT M.value_id,NS.name_id,NS.symbol_id,M.value,M.bound,NS.name,NS.symbol,NS.symbol_norm,M.arxiv_id
			FROM measurements M
			LEFT OUTER JOIN (
				SELECT N.value_id,N.name_id,S.symbol_id,N.value,N.bound,N.name,S.symbol,S.symbol_norm,N.arxiv_id
				FROM name_measurements_values N
					LEFT OUTER JOIN symbol_measurements_values S ON N.value_id = S.value_id
				UNION
				SELECT S.value_id,N.name_id,S.symbol_id,S.value,S.bound,N.name,S.symbol,S.symbol_norm,S.arxiv_id
				FROM symbol_measurements_values S
					LEFT OUTER JOIN name_measurements_values N ON S.value_id = N.value_id
				) NS
			ON M.value_id = NS.value_id
		''')

	# Get a view of the confidence limits associated with each measurement
	cursor.execute('DROP VIEW IF EXISTS measurement_confidences')
	cursor.execute('''
		CREATE VIEW measurement_confidences
			(
				value_id,
				confidence_id,
				confidence_text,
				confidence,
				arxiv_id
			)
		AS
			SELECT V.value_id,V.confidence_id,C.confidence,C.confidence_norm,C.arxiv_id
			FROM confidence_limits C
				INNER JOIN value_confidences V ON C.entity_id = V.confidence_id
		''')

	# Get all measurements, with confidences
	cursor.execute('DROP VIEW IF EXISTS all_measurements_confidences')
	cursor.execute('''
		CREATE VIEW all_measurements_confidences
			(
				value_id,
				name_id,
				symbol_id,
				value,
				bound,
				name,
				symbol,
				symbol_norm,
				confidence_text,
				confidence,
				arxiv_id
			)
		AS
			SELECT M.value_id,M.name_id,M.symbol_id,M.value,M.bound,M.name,M.symbol,M.symbol_norm,C.confidence_text,IFNULL(C.confidence,1),M.arxiv_id
			FROM all_measurements M
			LEFT OUTER JOIN measurement_confidences C ON M.value_id = C.value_id
		''')

def convert_symbol_for_database(symbol):
	parsed = parse_symbol(symbol)
	return str(parsed) if parsed is not None else symbol
	#return str(parse_symbol(symbol))

clre = re.compile(r'''
		((?P<number>[0-9]+(\.[0-9]+)?) | (?P<letters>one | two | three | four | five))
		\s*
		((?P<percent> per \s* cent | \\?\%) | (\u002D|\u2013|\u2014|\u2212)? \s* (?P<sigma>\\?sigma))
	''',flags=re.VERBOSE|re.I)
letters_lookup = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10}
def convert_confidence_for_database(confidence):

	def parse(match):
		if match.group('number'):
			number = float(match.group('number'))
		else:
			number = letters_lookup[match.group('letters').lower()]

		if match.group('percent'):
			if number > 99.999999:
				number = 4 ## Not ideal, but better than Inf
			else:
				number = erfinv(number/100)*numpy.sqrt(2)

		return number

	return min((parse(m) for m in clre.finditer(confidence)),key=lambda x: abs(x-1), default=1) # Why min here? What is this supposed to solve?

def refresh_database(cursor):

	# Inlcude any updates to the database views
	create_or_update_views(cursor)

	# Refresh normalised symbols to most up-to-date versions
	symbols = query(cursor, 'SELECT symbol FROM parameter_symbols')['symbol']
	cursor.executemany('UPDATE parameter_symbols SET symbol_norm = ? WHERE symbol = ?', ((convert_symbol_for_database(s),s) for s in symbols))

	# Refresh normalised confidence limits to most up-to-date versions
	### THIS IS NOT OPTIMAL - RETHINK CONFIDENCE STORAGE STRATEGY?
	confidence_limits = query(cursor, 'SELECT confidence FROM confidence_limits')['confidence']
	cursor.executemany('UPDATE confidence_limits SET confidence_norm = ? WHERE confidence = ?', ((convert_confidence_for_database(c),c) for c in confidence_limits))


def start_predictions(source, cursor):
	cursor.execute('INSERT OR REPLACE INTO prediction_runs (source,status) VALUES (?,?)', (source,'STARTED'))

def complete_predictions(source, cursor):
	cursor.execute('''
		UPDATE prediction_runs
		SET status = ?
		WHERE source = ?
			''', ('COMPLETED',source))


def add(ann, arXivID, date, cursor):

	cursor.execute('INSERT INTO papers (arxiv_id,abstract,date) VALUES (?,?,?)', (arXivID,ann.text,date))

	entity_ids = {}
	for e in ann.entities:
		try:
			if e.type == 'ParameterSymbol':
				cursor.execute('INSERT OR IGNORE INTO parameter_symbols (symbol,symbol_norm) VALUES (?,?)', (e.text,convert_symbol_for_database(e.text)))
				cursor.execute('INSERT INTO parameter_symbol_occurences (symbol,start,end,arxiv_id) VALUES (?,?,?,?)', (e.text,e.start,e.end,arXivID))
				entity_ids[id(e)] = cursor.lastrowid
			elif e.type == 'ParameterName':
				cursor.execute('INSERT OR IGNORE INTO parameter_names (name) VALUES (?)', (e.text,))
				cursor.execute('INSERT INTO parameter_name_occurences (name,start,end,arxiv_id) VALUES (?,?,?,?)', (e.text,e.start,e.end,arXivID))
				entity_ids[id(e)] = cursor.lastrowid
			elif e.type == 'MeasuredValue' or e.type == 'Constraint':
				bound = next(iter(ann.get_attributes(e)),None)
				if bound and bound.type in ('UpperBound','LowerBound'): bound = bound.type[0]
				else: bound = 'C'
				cursor.execute('INSERT INTO measurements (value,start,end,bound,arxiv_id) VALUES (?,?,?,?,?)', (e.text,e.start,e.end,bound,arXivID))
				entity_ids[id(e)] = cursor.lastrowid
			elif e.type == 'ConfidenceLimit':
				cursor.execute('INSERT INTO confidence_limits (confidence,confidence_norm,start,end,arxiv_id) VALUES (?,?,?,?,?)', (e.text,convert_confidence_for_database(e.text),e.start,e.end,arXivID))
				entity_ids[id(e)] = cursor.lastrowid
			elif e.type == 'ObjectName':
				cursor.execute('INSERT OR IGNORE INTO objects (object) VALUES (?)', (e.text,))
				cursor.execute('INSERT INTO object_occurences (object,start,end,arxiv_id) VALUES (?,?,?,?)', (e.text,e.start,e.end,arXivID))
				entity_ids[id(e)] = cursor.lastrowid
			elif e.type == 'Definition':
				cursor.execute('INSERT OR IGNORE INTO equations (equation) VALUES (?)', (e.text,))
				cursor.execute('INSERT INTO equation_occurences (equation,start,end,arxiv_id) VALUES (?,?,?,?)', (e.text,e.start,e.end,arXivID))
				entity_ids[id(e)] = cursor.lastrowid
		except sqlite3.IntegrityError:
			pass # We've violated an SQL constraint somewhere... (probably Constraint CHECK?)
			raise

	for r in ann.relations:
		try:
			if r.type == 'Measurement' and id(r.arg1) in entity_ids and id(r.arg2) in entity_ids:
				if r.arg1.type == 'ParameterSymbol':
					cursor.execute('''INSERT INTO symbol_measurements (symbol,symbol_id,value_id,arxiv_id) VALUES (?,?,?,?)''',
							(r.arg1.text,entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID))
				elif r.arg1.type == 'ParameterName':
					cursor.execute('''INSERT INTO name_measurements (name,name_id,value_id,arxiv_id) VALUES (?,?,?,?)''',
							(r.arg1.text,entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID))
			elif r.type == 'Confidence':
				cursor.execute('''INSERT INTO value_confidences (value_id,confidence_id,arxiv_id) VALUES (?,?,?)''',
						(entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID))
			elif r.type == 'Name':
				if r.arg1.type == 'ParameterName' and r.arg2.type == 'ParameterSymbol':
					name,symbol = r.arg1,r.arg2
				else: # In case the relation has been reversed somehow?
					name,symbol = r.arg2,r.arg1
				cursor.execute('INSERT OR IGNORE INTO names (name,symbol) VALUES (?,?)', (name.text,symbol.text))
				cursor.execute('''INSERT INTO name_occurences (name_id,start_id,end_id,arxiv_id)
								SELECT name_id, ?, ?, ? FROM names WHERE name = ? and symbol = ?''',
							(entity_ids[id(name)],entity_ids[id(symbol)],arXivID,name.text,symbol.text))
			elif r.type == 'Property':
				if r.arg2.type == 'ParameterName':
					cursor.execute('INSERT OR IGNORE INTO object_property_names (object,name) VALUES (?,?)', (r.arg1.text,r.arg2.text))
					cursor.execute('''INSERT INTO property_name_occurences (property_name_id,start_id,end_id,arxiv_id)
									SELECT property_name_id, ?, ?, ? FROM object_property_names WHERE object = ? and name = ?''',
								(entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID,r.arg1.text,r.arg2.text))
				elif r.arg2.type == 'ParameterSymbol':
					cursor.execute('INSERT OR IGNORE INTO object_property_symbols (object,symbol) VALUES (?,?)', (r.arg1.text,r.arg2.text))
					cursor.execute('''INSERT INTO property_symbol_occurences (property_symbol_id,start_id,end_id,arxiv_id)
									SELECT property_symbol_id, ?, ?, ? FROM object_property_symbols WHERE object = ? and symbol = ?''',
								(entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID,r.arg1.text,r.arg2.text))
				elif r.arg2.type == 'MeasuredValue' or r.arg2.type == 'Constraint':
					cursor.execute('INSERT INTO property_measurements (object,object_id,value_id,arxiv_id) VALUES (?,?,?,?)',
							(r.arg1.text,entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID))
			elif r.type == 'Defined':
				if r.arg1.type == 'ParameterSymbol':
					cursor.execute('INSERT OR IGNORE INTO symbol_definitions (symbol,equation) VALUES (?,?)', (r.arg1.text,r.arg2.text))
					cursor.execute('''INSERT OR IGNORE INTO symbol_definition_occurences (symbol_definition_id,start_id,end_id,arxiv_id)
									SELECT symbol_definition_id, ?, ?, ? FROM symbol_definitions WHERE symbol = ? and equation = ?''',
								(entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID,r.arg1.text,r.arg2.text))
				elif r.arg1.type == 'ParameterName':
					cursor.execute('INSERT OR IGNORE INTO name_definitions (name,equation) VALUES (?,?)', (r.arg1.text,r.arg2.text))
					cursor.execute('''INSERT OR IGNORE INTO name_definition_occurences (name_definition_id,start_id,end_id,arxiv_id)
									SELECT name_definition_id, ?, ?, ? FROM name_definitions WHERE name = ? and equation = ?''',
								(entity_ids[id(r.arg1)],entity_ids[id(r.arg2)],arXivID,r.arg1.text,r.arg2.text))
		except sqlite3.IntegrityError:
			pass # We've violated an SQL constraint somewhere...
			raise

def get(arXivID, cursor, parse_notes=False):
	cursor.execute('SELECT abstract FROM papers WHERE arxiv_id = ?', (arXivID,))
	abstract = cursor.fetchone()[0]
	ann = Standoff.create(abstract)

	objects = query(cursor,'SELECT entity_id,start,end FROM object_occurences WHERE arxiv_id = ?',(arXivID,))
	names = query(cursor,'SELECT entity_id,start,end FROM parameter_name_occurences WHERE arxiv_id = ?',(arXivID,))
	symbols = query(cursor,'SELECT entity_id,start,end FROM parameter_symbol_occurences WHERE arxiv_id = ?',(arXivID,))
	confidence_limits = query(cursor,'SELECT entity_id,start,end FROM confidence_limits WHERE arxiv_id = ?',(arXivID,))
	definitions = query(cursor,'SELECT entity_id,start,end FROM equation_occurences WHERE arxiv_id = ?',(arXivID,))

	measurements = query(cursor,'SELECT value_id,start,end,bound FROM measurements WHERE arxiv_id = ?',(arXivID,))

	entity_ids = dict()

	for type,table in [('ObjectName',objects),('ParameterName',names),('ParameterSymbol',symbols),('ConfidenceLimit',confidence_limits),('Definition',definitions)]:
		for i,row in table.iterrows():
			ent = ann.entity(type,row['start'],row['end'])
			entity_ids[(type,row['entity_id'])] = ent

			if parse_notes:
				if type == 'ConfidenceLimit':
					ann.note(ent, '{:.2f}'.format(convert_confidence_for_database(ent.text)))
				elif type == 'ParameterSymbol':
					ann.note(ent, convert_symbol_for_database(ent.text))

	for i,row in measurements.iterrows():
		if row['bound'] in ('U','L'):
			type = 'Constraint'
		else:
			type = 'MeasuredValue'
		ent = ann.entity(type,row['start'],row['end'])
		entity_ids[('Measurement',row['value_id'])] = ent
		if row['bound'] == 'U':
			ann.attribute('UpperBound',ent)
		elif row['bound'] == 'L':
			ann.attribute('LowerBound',ent)

		if parse_notes:
			ann.note(ent, str(parse_measurement(ent.text)))

	names = query(cursor,'SELECT start_id,end_id FROM name_occurences WHERE arxiv_id = ?',(arXivID,))
	confidences = query(cursor,'SELECT value_id AS start_id,confidence_id AS end_id FROM value_confidences WHERE arxiv_id = ?',(arXivID,))
	name_measurements = query(cursor,'SELECT name_id AS start_id,value_id AS end_id FROM name_measurements WHERE arxiv_id = ?',(arXivID,))
	symbol_measurements = query(cursor,'SELECT symbol_id AS start_id,value_id AS end_id FROM symbol_measurements WHERE arxiv_id = ?',(arXivID,))
	property_name = query(cursor,'SELECT start_id,end_id FROM property_name_occurences WHERE arxiv_id = ?',(arXivID,))
	property_symbol = query(cursor,'SELECT start_id,end_id FROM property_symbol_occurences WHERE arxiv_id = ?',(arXivID,))
	property_measurements = query(cursor,'SELECT object_id AS start_id,value_id AS end_id FROM property_measurements WHERE arxiv_id = ?',(arXivID,))
	definition_symbol = query(cursor,'SELECT start_id,end_id FROM symbol_definition_occurences WHERE arxiv_id = ?',(arXivID,))
	definition_name = query(cursor,'SELECT start_id,end_id FROM name_definition_occurences WHERE arxiv_id = ?',(arXivID,))

	relations = [
			('Property','ObjectName','ParameterName',property_name),
			('Property','ObjectName','ParameterSymbol',property_symbol),
			('Property','ObjectName','Measurement',property_measurements),
			('Name','ParameterName','ParameterSymbol',names),
			('Measurement','ParameterName','Measurement',name_measurements),
			('Measurement','ParameterSymbol','Measurement',symbol_measurements),
			('Confidence','Measurement','ConfidenceLimit',confidences),
			('Defined','ParameterName','Definition',definition_name),
			('Defined','ParameterSymbol','Definition',definition_symbol),
		]

	for type,arg1,arg2,table in relations:
		for i,row in table.iterrows():
			ann.relation(type,entity_ids[(arg1,row['start_id'])],entity_ids[(arg2,row['end_id'])])

	return ann


def query(cursor, query, args=()):
	cursor.execute(query, args)
	return pandas.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])


## Answer from: https://stackoverflow.com/a/10856450/11002708
from io import StringIO
def loadconnect(path):
	# Read database to tempfile
	con = sqlite3.connect(path)
	tempfile = StringIO()
	for line in con.iterdump():
		tempfile.write('%s' % line)
	con.close()
	tempfile.seek(0)

	# Create a database in memory and import from tempfile
	mem_conn = sqlite3.connect(":memory:")
	#mem_conn.cursor().execute('PRAGMA foreign_keys = 1')
	#mem_conn.commit()
	mem_conn.cursor().executescript(tempfile.read())
	mem_conn.commit()

	return mem_conn

if __name__ == '__main__':

	import os

	from utilities.argparseactions import ArgumentParser,FileAction,IterFilesAction
	from metadata.oaipmh import MetadataAction,arXivID_from_path

	def run_init(args):

		if args.overwrite:
			if os.path.exists(args.database):
				response = input(f'Do you wish to overwrite the database at {args.database}? [YES/NO] ')
				if response == 'YES':
					os.remove(args.database)
					print(f'Remove database file at {args.database}. Reinitialising.')
				else:
					print('Did not overwrite database.')
					return
		elif os.path.exists(args.database):
			print(f'File {args.database} already in use. You may use the --overwrite flag to reinitialise the file (all existing data will be lost).')
			return

		connection = sqlite3.connect(args.database)
		cursor = connection.cursor()
		init(cursor)
		cursor.close()
		connection.commit()
		connection.close()

		print(f'Created database at {args.database}')

	def run_refresh(args):

		if not os.path.exists(args.database):
			print(f'No database found at {args.database}.')
			return

		connection = sqlite3.connect(args.database)
		cursor = connection.cursor()
		refresh_database(cursor)
		cursor.close()
		connection.commit()
		connection.close()

		print(f'Refreshed database at {args.database}')

	def run_populate(args):
		#anns = [(arXivID_from_path(a),Standoff.open(a)) for a in args.source]
		#anns = [(arXivID,a) for arXivID,a in anns if a]

		anns = []
		for a in args.source:
			try:
				arXivID = arXivID_from_path(a)
				ann = Standoff.open(a)
				anns.append((arXivID,ann))
			except ValueError:
				print(f'Could not open {a}')

		try:
			os.remove(args.database)
		except OSError:
			pass

		connection = sqlite3.connect(args.database)
		cursor = connection.cursor()
		init(cursor)

		for arXivID,ann in anns:
			add(ann,arXivID,args.metadata.get(arXivID,'date').date(),cursor)

		#from annotations.bratnormalisation import similarity_include,sort
		#failcount = 0
		#for arXivID,ann in anns:
		#	recovered = get(arXivID,cursor)
		#	s = similarity_include(['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Confidence','Measurement','Name','Property','UpperBound','LowerBound','Definition','Defined'],[ann,recovered])
		#	if s != 1 and (len(ann)>0 or len(recovered)>0):
		#		failcount += 1
		#		print(f'Malformed ({s}): {arXivID}')
		#		print(sort(ann))
		#		print('\n\n')
		#		print(sort(recovered))
		#		print('\n\n')
		#print(f'Fail count: {failcount}')

		connection.commit()
		connection.close()

		print(f'Created and populated database at {args.database}')

	parser = ArgumentParser(description='Create, refresh, or create and populate a database for .ann files.')
	subparsers = parser.add_subparsers()
	parser_init = subparsers.add_parser('init',help='Initialise a new database to be written to by another program.')
	parser_init.add_argument('database', action=FileAction, mustexist=False, help='Path at which to create database.')
	parser_init.add_argument('--overwrite', action='store_true', help='Flag to indicate that an existing database at the specified path should be wiped and reinitialised (will ask for conformation).')
	parser_init.set_defaults(main=run_init)
	parser_refresh = subparsers.add_parser('refresh',help='Update an existing database to have most recent versions of VIEWS and any dependent calculated data.')
	parser_refresh.add_argument('database', action=FileAction, mustexist=True, help='Database file to update.')
	parser_refresh.set_defaults(main=run_refresh)
	parser_pop = subparsers.add_parser('populate',help='Create a database and copy in data from .ann files.')
	parser_pop.add_argument('source',action=IterFilesAction, recursive=True, suffix='.ann', help='Source directory for .ann files, searched recursively.')
	parser_pop.add_argument('metadata',action=MetadataAction, help='arXiv metadata file (pickled).')
	parser_pop.add_argument('database',action=FileAction,mustexist=False,help='Path at which to create database.')
	parser_pop.set_defaults(main=run_populate)

	args = parser.parse_args()
	args.main(args)
