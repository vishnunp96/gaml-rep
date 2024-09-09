module LateXMLJL

import LightXML

isEmpty(node::LightXML.XMLNode) = isempty(strip(LightXML.content(node)))
isEmpty(str::String) = isempty(strip(str))

function find_elements(e, tag)
	if tag == LightXML.name(e)
		return [e]
	end
	elements = []
	for c in LightXML.child_nodes(e)
		elements = vcat(elements,find_elements(c,tag))
	end
	return elements
end

### XML functions
function get_node_text(node)
	text = LightXML.is_textnode(node) ? LightXML.content(node) : ""
	for i in LightXML.child_nodes(node)
		childtext = get_node_text(i)
		if !isEmpty(childtext)
			text = text * " " * get_node_text(i)
		end
	end
	return text
end

function readxmltext(path)
	doc = LightXML.parse_file(path)
	text = get_node_text(LightXML.root(doc))
	LightXML.free(doc)
	return text
end

latexmlignoretags = ["bibtag"]
latexmlfollownewline = ["title", "p", "bibitem", "tabular", "table"]

### Format LateXML ElementTree object into organised text for machine reading.
### Input is expected to be a 'prettified' tree, and hence output will be
### tokenized and sentence split. Also returns list of spans in text, along
### with path of element, and span location (text/tail) in element.
function tostring(element::Union{LightXML.XMLNode,LightXML.XMLElement}, pref_sep::Bool=false)

	if LightXML.name(element) == "tabular"
		response = "\n[TableHere]\n"
		return response
	end

	text = ""

	if !(LightXML.name(element) in latexmlignoretags)
		if LightXML.is_textnode(element) && !isEmpty(element)
			buffer_chars = pref_sep ? " " : ""
			text *= buffer_chars * LightXML.content(element)
		end
		for child in LightXML.child_nodes(element)

			## If there is text and the last character isn't a space, add a space
			## Or if there is no text and we were told to add a space, add one
			#(text and not text[-1].isspace()) or (not text and pref_sep)

			childtext = tostring(child,((!isempty(text) && !isspace(text[end])) || (isempty(text) && pref_sep)))
			if !isEmpty(childtext)
				text *= childtext
			end
		end
	end

	if LightXML.name(element) in latexmlfollownewline
		text *= "\n\n"
	end

	return text
end

function tostring(path::String)
	doc = LightXML.parse_file(path)
	xroot = LightXML.root(doc)
	text = tostring(xroot)
	LightXML.free(doc)
	return text
end

function abstract_string(path)
	doc = LightXML.parse_file(path)
	xroot = LightXML.root(doc)
	text = join([tostring(elem) for elem in find_elements(xroot,"abstract")], "\n\n")
	LightXML.free(doc)
	return text
end

end # module
