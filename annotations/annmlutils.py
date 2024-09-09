from annotations.bratnormalisation import open_clean,add_implied_relations
#from annotations.bratutils import Standoff
from annotations.brattowindow import StandoffLabels

def open_anns(paths,types=None,use_labelled=False):
	''' Read in data from iterable of paths. Number of Standoff(Labels) objects may not equal number of paths. '''
	anns = []
	for path in paths:
		try:
			#anns.append(StandoffLabels.open(path,types=types,include_bio=True))
			standoff = open_clean(path,check_repetitions=('ParameterName','ParameterSymbol','ObjectName'))
			standoff = add_implied_relations(standoff,inplace=True)
			# Convert Constraints to MeasuredValues (we can recover Constraints using Attributes)
			for c in [e for e in standoff.entities if e.type=='Constraint']:
				c.type = 'MeasuredValue'
			if standoff:
				if use_labelled:
					anns.append(StandoffLabels(standoff,types=types,include_bio=True))
				else:
					anns.append(standoff)
		except KeyError:
			print(f'Error opening {path}')
	return anns

