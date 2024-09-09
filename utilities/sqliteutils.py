import io
import zlib
import sqlite3
import numpy
import pandas

def query(cursor, query_str, args=()):
	cursor.execute(query_str, args)
	return pandas.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])

def query_db(databasepath, query_str, args=(),
		timeout=5.0, detect_types=0, isolation_level=None,
		check_same_thread=True,
		cached_statements=100, uri=False):

	with sqlite3.connect(databasepath, timeout=timeout, detect_types=detect_types,
			isolation_level=isolation_level, check_same_thread=check_same_thread,
			cached_statements=cached_statements, uri=uri) as connection:
		cursor = connection.cursor()
		result = query(cursor, query_str, args)
		cursor.close()

	return result

## Following taken from https://stackoverflow.com/a/46358247/11002708 (asterio gonzalez)
def __encode_array(arr):
	# From http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
	out = io.BytesIO()
	numpy.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(zlib.compress(out.read()))

def __decode_array(text):
	t = io.BytesIO(text)
	t.seek(0)
	out = io.BytesIO(zlib.decompress(t.read()))
	return numpy.load(out)

def register_numpy_array_type():
	sqlite3.register_adapter(numpy.ndarray, __encode_array)
	sqlite3.register_converter('array', __decode_array)
