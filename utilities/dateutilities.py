from datetime import datetime, timedelta
import dateutil.parser

def datetime_range(start, end, delta, inclusive=True, tick_delta=timedelta(seconds=1)):
	finish = end-tick_delta

	current = start
	next = start + delta

	while next-tick_delta < finish:
		yield (current, next-tick_delta)
		current = next
		next += delta

	if inclusive:
		yield (current,end)
	else:
		yield (current,end-tick_delta)


def roundtime(dt, delta, starttime=datetime.min,rounding=round):
	deltaSeconds = delta.total_seconds()
	difference = (dt.replace(tzinfo=None) - starttime)
	seconds = difference.total_seconds()
	return starttime + timedelta(seconds=rounding(seconds/deltaSeconds)*deltaSeconds)

def roundtimedelta(td, delta, rounding=round):
	deltaSeconds = delta.total_seconds()
	tdSeconds = td.total_seconds()
	return timedelta(seconds=rounding(tdSeconds/deltaSeconds)*deltaSeconds)

def getdatetime(datestring, format=None):
	try:
		if format:
			return datetime.strptime(datestring, format)
		else:
			return dateutil.parser.parse(datestring)
	except ValueError:
		return None

def months_range(start,end,inclusive=True,follow_on=True):
	for month in range(start.month,13):
		yield datetime(start.year,month,1)
	for year in range(start.year+1,end.year):
		for month in range(1,13):
			yield datetime(year,month,1)
	if end.year != start.year:
		for month in range(1,end.month):
			yield datetime(end.year,month,1)
	if inclusive or follow_on:
		yield datetime(end.year,end.month,1)
		if follow_on:
			if end.month < 12:
				yield datetime(end.year,end.month+1,1)
			else:
				yield datetime(end.year+1,1,1)

if __name__ == '__main__':

	for i in datetime_range(datetime(2016, 9, 1, 0), datetime(2016, 9, 3, 0), timedelta(days=1), inclusive=False):
		print(i)
