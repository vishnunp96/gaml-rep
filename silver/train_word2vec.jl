using Word2Vec
using ArgParse
using LightXML
using DataStructures
using JSON
using ProgressMeter

isEmpty(node::LightXML.XMLNode) = isempty(strip(content(node)))
isEmpty(str::String) = isempty(strip(str))

function find_elements(e, tag)
	if tag == name(e)
		return [e]
	end
	elements = []
	for c in child_nodes(e)
		elements = vcat(elements,find_elements(c,tag))
	end
	return elements
end

### XML functions
function get_node_text(node)
	text = is_textnode(node) ? content(node) : ""
	for i in child_nodes(node)
		childtext = get_node_text(i)
		if !isEmpty(childtext)
			text = text * " " * get_node_text(i)
		end
	end
	return text
end

function readxmltext(path)
	doc = parse_file(path)
	text = get_node_text(root(doc))
	free(doc)
	return text
end

latexmlignoretags = ["bibtag"]
latexmlfollownewline = ["title", "p", "bibitem", "tabular", "table"]


### Format LateXML ElementTree object into organised text for machine reading.
### Input is expected to be a 'prettified' tree, and hence output will be
### tokenized and sentence split. Also returns list of spans in text, along
### with path of element, and span location (text/tail) in element.
function tostring(element::Union{LightXML.XMLNode,LightXML.XMLElement}, pref_sep::Bool=false)

	if name(element) == "tabular"
		response = "\n[TableHere]\n"
		return response
	end

	text = ""

	if !(name(element) in latexmlignoretags)
		if is_textnode(element) && !isEmpty(element)
			buffer_chars = pref_sep ? " " : ""
			text *= buffer_chars * content(element)
		end
		for child in child_nodes(element)

			## If there is text and the last character isn't a space, add a space
			## Or if there is no text and we were told to add a space, add one
			#(text and not text[-1].isspace()) or (not text and pref_sep)

			childtext = tostring(child,((!isempty(text) && !isspace(text[end])) || (isempty(text) && pref_sep)))
			if !isEmpty(childtext)
				text *= childtext
			end
		end
	end

	if name(element) in latexmlfollownewline
		text *= "\n\n"
	end

	return text
end

function abstract_string(path)
	doc = parse_file(path)
	xroot = root(doc)
	text = join([tostring(elem) for elem in find_elements(xroot,"abstract")], "\n\n")
	free(doc)
	return text
end

function tostring(path::String)
	doc = parse_file(path)
	text = tostring(root(doc))
	free(doc)
	return text
end

function main()

	function parse_commandline()
		s = ArgParseSettings()

		@add_arg_table s begin
			"-s", "--source"
				help = "Source directory containing manifest.json from which to construct corpus."
			"corpus"
				help = "Corpus text file. If source provied the resulting corpus will be stored here, otherwise it will be read."
				required = true
			"output"
				help = "Output file for storing word vectors."
				required = true
			"-p", "--processes"
				help = "Number of processes to use when training word2vec."
				arg_type = Int
				default = 1
		end

		return parse_args(s)
	end

	args = parse_commandline()

	if args["source"] != nothing
		source = abspath(args["source"])
		manifest = JSON.parsefile(joinpath(source,"manifest.json"),dicttype=Dict{String,String})

		println("Opened files.")

		corpus = open(args["corpus"],"w")
		progress = Progress(length(manifest), dt=1, desc="Compiling corpus: ")
		for path in values(manifest)
			write(corpus,"\n" * tostring(joinpath(source,path)))
			next!(progress)
		end
		close(corpus)

		println("Constructed corpus.")
	end

	word2vec(args["corpus"], args["output"], verbose=true, threads=args["processes"])

end

main()
