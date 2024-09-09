from lxml import etree as let

#def fastxmliter(filepath, events=("end",), tag=None):
#	"""
#	http://lxml.de/parsing.html#modifying-the-tree
#	Based on Liza Daly's fast_iter
#	http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
#	See also http://effbot.org/zone/element-iterparse.htm
#	"""
#	context = let.iterparse(filepath,events=events,tag=tag)
#	for event, elem in context:
#		yield event,elem
#		elem.clear()
#		# Also eliminate now-empty references from the root node to elem
#		for ancestor in elem.xpath('ancestor-or-self::*'):
#			while ancestor.getprevious() is not None:
#				del ancestor.getparent()[0]
#	del context


def fastxmliter(filepath, events=("end",), tag=None):
	"""
	Based on Liza Daly's fast_iter
	http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
	"""
	context = let.iterparse(filepath,events=events,tag=tag)
	for event, elem in context:
		yield event,elem
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
	del context

