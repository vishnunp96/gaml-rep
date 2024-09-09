addprocs(16)

using Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle, sigmoid
using Base.Iterators: repeated
using Flux

using JSON
using BSON: @save
using DataFrames
using CSV

using ArgParse

@everywhere using LightXML
@everywhere using DataStructures

@everywhere isEmpty(node::LightXML.XMLNode) = isempty(strip(content(node)))
@everywhere isEmpty(str::String) = isempty(strip(str))

@everywhere function find_elements(e, tag)
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
@everywhere function get_node_text(node)
	text = is_textnode(node) ? content(node) : ""
	for i in child_nodes(node)
		childtext = get_node_text(i)
		if !isEmpty(childtext)
			text = text * " " * get_node_text(i)
		end
	end
	return text
end

@everywhere function readxmltext(path)
	doc = parse_file(path)
	text = get_node_text(root(doc))
	free(doc)
	return text
end

@everywhere latexmlignoretags = ["bibtag"]
@everywhere latexmlfollownewline = ["title", "p", "bibitem", "tabular", "table"]


### Format LateXML ElementTree object into organised text for machine reading.
### Input is expected to be a 'prettified' tree, and hence output will be
### tokenized and sentence split. Also returns list of spans in text, along
### with path of element, and span location (text/tail) in element.
@everywhere function tostring(element::Union{LightXML.XMLNode,LightXML.XMLElement}, pref_sep::Bool=false)

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

@everywhere function abstract_string(path)
	doc = parse_file(path)
	xroot = root(doc)
	text = join([tostring(elem) for elem in find_elements(xroot,"abstract")], "\n\n")
	free(doc)
	return text
end

### Alter this to determine how much of document is read
@everywhere read_latexml(path) = abstract_string(path)

function pmapreduce(mapfunc, reducefunc, iters...)
	@parallel reducefunc for arg in collect(zip(iters...))
		mapfunc(arg...)
	end
end

@everywhere mergedict(d1,d2) = Dict(merge(counter(d1),counter(d2)))

@everywhere function get_wordcounts(path)
	counts = Dict{String,Int64}()
	for word in split(read_latexml(path))
		counts[word] = get(counts, word, 0) + 1
	end
	return counts
end

# Construct vocabulary in parallel
function make_vocab(paths)
	counts = pmapreduce(get_wordcounts,mergedict,paths)
	counts = filter((k,v) -> v > 1, counts)
	vocabulary = Dict{String,Int64}("OOV" => 1)
	vocab_count = 1
	function addword(word)
		vocab_count += 1
		vocabulary[word] = vocab_count
	end
	for (k,v) in counts
		addword(k)
	end
	return vocabulary
end

# Helper function for creating document vectors
@everywhere function make_docvec(path, vocabulary)
	#getid(word) = get(vocabulary,word,vocabulary["OOV"])
	oov = vocabulary["OOV"]
	docvec = Int[]
	for word in split(read_latexml(path))
		push!(docvec, get(vocabulary,word,oov))
	end
	if length(docvec) == 0
		println("Zero length text: $path")
	end
	return docvec
end

# Helper function for making datasets
function make_dataset(paths, target, vocabulary)
	dataset_x = Vector{Vector{Int}}()
	for path in paths
		docvec = make_docvec(path,vocabulary)
		if length(docvec) > 0
			push!(dataset_x,docvec)
		end
	end
	return (dataset_x, fill(target, length(dataset_x)))
end


function main()

	function parse_commandline()
		s = ArgParseSettings()

		@add_arg_table s begin
			"scores"
				help = "Location of dataset."
				required = true
			"source"
				help = "Source directory containing manifest.json."
				required = true
			"-s", "--minscore"
				help = "Minimum score for data to be considered a positive sample."
				arg_type = Int
				default = 3
			"-m", "--model"
				help = "Path at which to store model."
				default = "silvermodel.bson"
			"-y", "--predictions"
				help = "Path at which to store model predictions."
				default = "model_predictions.csv"
			"-l", "--lossdata"
				help = "Location of loss data csv file. Defaults to 'lossdata.csv'."
				default = "lossdata.csv"
		end

		return parse_args(s)
	end

	args = parse_commandline()


	#### Load data
	scores = CSV.read(args["scores"],types=[String,Int])

	source = abspath(args["source"])
	manifest = JSON.parsefile(joinpath(source,"manifest.json"),dicttype=Dict{String,String})

	println("Opened files.")

	positive_paths = map(id -> joinpath(source,manifest[id]) , scores[scores[:score] .>= args["minscore"], :id])
	negative_paths = map(id -> joinpath(source,manifest[id]) , scores[scores[:score] .== 0, :id])

	println("Compiled paths.")

	###### Make datasets
	# Make positive dataset
	#pos_data_x,pos_data_y = make_dataset(positive_paths,vocabulary)
	#pos_num = length(pos_data_x)
	#train_num = floor(Int,0.9*pos_num)

	#pos_train_x,pos_train_y = pos_data_x[1:train_num],pos_data_y[1:train_num]
	#pos_test_x,pos_test_y = pos_data_x[train_num+1:end],pos_data_y[train_num+1:end]
	#neg_test_x,neg_test_y = make_dataset(rand(negative_paths, length(pos_test_x)),vocabulary)

	class_num = 500
	train_num = max(floor(Int,0.9*class_num), 1)
	test_num = max(class_num - train_num, 1)

	pos_paths = rand(positive_paths, class_num)
	neg_paths = rand(negative_paths, class_num)

	trainrange = 1:train_num
	testrange = min(train_num+1,class_num):class_num

	vocabulary = make_vocab(vcat(pos_paths[trainrange],neg_paths[trainrange]))
	println("Created vocabulary.")

	pos_train_x,pos_train_y = make_dataset(pos_paths[trainrange],1.0,vocabulary)
	pos_test_x,pos_test_y = make_dataset(pos_paths[testrange],1.0,vocabulary)

	neg_train_x,neg_train_y = make_dataset(neg_paths[trainrange],0.0,vocabulary)
	neg_test_x,neg_test_y = make_dataset(neg_paths[testrange],0.0,vocabulary)

	println("Created datasets.")
	println("Vocab size = $(length(vocabulary)), +ve samples = $(length(pos_paths)), -ve samples = $(length(neg_paths))")
	println("Training size = $(length(trainrange)), testing size = $(length(testrange))")

	@show length(pos_train_x)
	@show length(neg_train_x)
	@show length(pos_test_x)
	@show length(neg_test_x)

	@show pos_paths[1]
	@show length(pos_train_x[1])
	@show neg_paths[1]
	@show length(neg_train_x[1])

	#### Vocabulary

	#vocabulary = JSON.parsefile(args["vocabulary"],dicttype=Dict{String,Int})
	#vocabulary = make_vocab(vcat(pos_paths[1:90],neg_paths[1:90]))
	#println("Loaded vocabulary.")

	#### Make model

	vocab_size = length(vocabulary)
	embedding_dim = 32
	input_size = 3 * embedding_dim
	output_size = 1

	embedding = param(randn(embedding_dim,vocab_size)/1000)
	#W = param(rand(output_size,input_size))
	#b = param(rand(output_size))
	#embedding = param(Flux.glorot_uniform(embedding_dim,vocab_size))
	W = param(Flux.glorot_uniform(output_size,input_size))
	b = param(zeros(output_size))

	model = Chain(
			x -> embedding[:,x],
			x -> cat(1,minimum(x,2),maximum(x,2),mean(x,2)),
			x -> W*x .+ b,
			#x -> sigmoid.(x),
		)

	loss(x,y::Number) = Flux.logitbinarycrossentropy(sum(model(x)),y)
	loss(x,y::Array) = sum(loss(xi,yi) for (xi,yi) in zip(x,y))/size(y,1)

	println("Created model.")

	#### Train model

	println("Setup training.")

	lossdata = open(args["lossdata"],"w")
	write(lossdata,"epoch,pos_train,neg_train,pos_test,neg_test\n")
	write(lossdata,"0,$(Flux.data(loss(pos_train_x,pos_train_y))),$(Flux.data(loss(neg_train_x,neg_train_y))),$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y)))\n")
	flush(lossdata)

	eta = 0.1
	for epoch in 1:100

		println("Epoch $epoch")

		#neg_train_x,neg_train_y = make_dataset(rand(negative_paths, train_num),vocabulary)
		#println("Made epoch $epoch negative samples.")

		for (x,y) in zip(vcat(pos_train_x,neg_train_x), vcat(pos_train_y,neg_train_y))
		#for (x,y) in zip(pos_train_x, pos_train_y)
			l = loss(x,y)
			back!(l)

			#@show l
			#@show sum(embedding.grad.^2)
			#@show sum(embedding.data.^2)
			#@show sum(W.grad.^2)
			#@show sum(W.data.^2)
			#@show sum(b.grad.^2)
			#@show sum(b.data.^2)

			for p in (embedding,W,b)
				p.data .-= eta .* p.grad
				p.grad .= 0
			end
		end

		write(lossdata,"$epoch,$(Flux.data(loss(pos_train_x,pos_train_y))),$(Flux.data(loss(neg_train_x,neg_train_y))),$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y)))\n")
		flush(lossdata)
	end

	close(lossdata)

	println("Finished training.")

	#### Save model
	@save args["model"] model

	println("Saved model.")

	#### Save model predictions
	#open(args["predictions"],"w") do f
	#	write(f,"arXiv,prediction\n")
	#	for id in scores[:id]
	#		write(f,id*","*string(model(make_docvec(joinpath(source,manifest[id]),vocabulary)))*"\n")
	#	end
	#end
	#println("Saved model predictions.")

end ## End main

main()
