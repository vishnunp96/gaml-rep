using Distributed
#addprocs(16)

using Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle, sigmoid
using Base.Iterators: repeated
using Flux

using JSON
using BSON: @save
using DataFrames
using CSV

@everywhere using Word2Vec

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

#=
function pmapreduce(mapfunc, reducefunc, iters...)
	@parallel reducefunc for arg in collect(zip(iters...))
		mapfunc(arg...)
	end
end
=#

# Helper function for creating document vectors
@everywhere function make_docmat(path, vocabulary)
	#oov = vocabulary["OOV"]
	#getid(word) = get(vocabulary,word,vocabulary["OOV"])
	getid(word) = get_vector(vocabulary, word)
	vecs = []
	for word in split(read_latexml(path))
		if in_vocabulary(vocabulary, word)
			push!(vecs, getid(word))
		#else
		#	println("\'$word\' not in vocabulary.")
		end
	end
	if length(vecs) == 0
		println("Zero length text: $path")
	end
	return hcat(vecs...)
end

# Helper function for making datasets
function make_dataset(paths, target, vocabulary)
	dataset_x = Vector{Array{Float64}}()
	for path in paths
		docmat = make_docmat(path,vocabulary)
		if length(docmat) > 0
			push!(dataset_x,docmat)
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
			"wordvectors"
				help = "File containing word vectors to be used."
				required = true
			"-s", "--minscore"
				help = "Minimum score for data to be considered a positive sample. Defaults to 1."
				arg_type = Int
				default = 1
			"-l", "--lossdata"
				help = "Location of loss data csv file. Defaults to 'lossdata.csv'."
				default = "lossdata.csv"
			"-n", "--numsamples"
				help = "Number of positive samples to take from dataset. An input less than 1 uses all available data. Defaults to -1 (all)."
				arg_type = Int
				default = -1
			"-t", "--testfrac"
				help = "Fraction of sampled positive data to use as testing samples. Defaults to 0.1."
				arg_type = Float64
				default = 0.1
			"-m", "--model"
				help = "Path at which to store model."
			"-y", "--predictions"
				help = "Path at which to store model predictions."
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

	vocabulary = wordvectors(args["wordvectors"])
	println("Loaded vocabulary.")

	###### Make datasets
	# Make positive dataset
	pos_num = args["numsamples"] > 0 ? min(args["numsamples"], length(positive_paths)) : length(positive_paths)

	test_num = floor(Int,args["testfrac"]*pos_num)
	train_num = pos_num - test_num

	pos_data_x, pos_data_y = make_dataset(rand(positive_paths, pos_num), 1.0, vocabulary)

	pos_test_x, pos_test_y = pos_data_x[1:test_num], pos_data_y[1:test_num]
	pos_train_x, pos_train_y = pos_data_x[test_num+1:end], pos_data_y[test_num+1:end]

	neg_test_x, neg_test_y = make_dataset(rand(negative_paths, test_num), 0.0, vocabulary)

	println("Created datasets.")

	@show length(pos_train_x)
	@show length(pos_test_x)
	@show length(neg_test_x)

	#### Make model

	vocab_dim = size(vocabulary)[1]
	input_size = 3 * vocab_dim
	output_size = 1

	#projection = param(eye(vocab_dim)/100 .+ randn(vocab_dim,vocab_dim)/1000)
	projection = param(eye(vocab_dim)/100 .+ randn(vocab_dim,vocab_dim)/1000)
	#W = param(rand(output_size,input_size))
	#b = param(rand(output_size))
	#projection = param(Flux.glorot_uniform(vocab_dim,vocab_dim))
	W = param(Flux.glorot_uniform(output_size,input_size))
	b = param(zeros(output_size))

	#=
	model = Chain(
			#x -> projection * x,
			x -> tanh.(projection * x),
			x -> cat(1,minimum(x,2),maximum(x,2),mean(x,2)),
			x -> W*x .+ b,
			#x -> sigmoid.(x),
		)
	=#

	model = Chain(
			x -> [tanh.(projection * i) for i in x],
			x -> [cat(1,minimum(i,2),maximum(i,2),mean(i,2)) for i in x],
			x -> hcat(x...),
			x -> W*x .+ b,
			#x -> vcat(x...)
		)

	#loss(x,y::Number) = Flux.logitbinarycrossentropy(sum(model(x)),y) + 0.05*(sum(W.^2)+sum(b.^2)+sum(projection.^2))/2
	#loss(x,y::Array) = sum(loss(xi,yi) for (xi,yi) in zip(x,y))/size(y,1)

	loss(x::Number,y::Number) = Flux.logitbinarycrossentropy(x,y) + 0.05*(sum(W.^2)+sum(b.^2)+sum(projection.^2))/2
	loss(x::Array,y::Array) = sum(loss(xi,yi) for (xi,yi) in zip(model(x),y))/size(y,1)

	accuracy(x::Array,y::Array) = mean(y .== round.(sigmoid.(Flux.data(model(x)))))

	println("Created model.")

	#i1 = [tanh.(projection * i) for i in pos_test_x]
	#@show i1
	#i2 = [cat(1,minimum(i,2),maximum(i,2),mean(i,2)) for i in i1]
	#@show i2
	#i3 = hcat(i2...)
	#@show i3
	#i4 = W*i3 .+ b
	#@show i4
	@show accuracy(pos_test_x,pos_test_y)
	@show loss(pos_test_x,pos_test_y)

	#### Train model

	println("Setup training.")

	lossdata = open(args["lossdata"],"w")
	write(lossdata,"epoch,pos_train_loss,neg_train_loss,pos_test_loss,neg_test_loss,pos_train_acc,neg_train_acc,pos_test_acc,neg_test_acc\n")
	#write(lossdata,"0,$(Flux.data(loss(pos_train_x,pos_train_y))),$(Flux.data(loss(neg_train_x,neg_train_y))),$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y)))\n")
	#write(lossdata,"0,$(Flux.data(loss(pos_train_x,pos_train_y))),,$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y)))\n")
	write(lossdata,"0,$(Flux.data(loss(pos_train_x,pos_train_y))),,$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y))),$(Flux.data(accuracy(pos_train_x,pos_train_y))),,$(Flux.data(accuracy(pos_test_x,pos_test_y))),$(Flux.data(accuracy(neg_test_x,neg_test_y)))\n")
	flush(lossdata)

	#=
	eta = 0.001
	for epoch in 1:100

		println("Epoch $epoch")

		#neg_train_x,neg_train_y = make_dataset(neg_paths[trainrange],0.0,vocabulary)
		neg_train_x, neg_train_y = make_dataset(rand(negative_paths, train_num), 0.0, vocabulary)
		println("Made epoch $epoch negative samples.")

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

			for p in (projection,W,b)
				p.data .-= eta .* p.grad
				p.grad .= 0
			end
		end

		write(lossdata,"$epoch,$(Flux.data(loss(pos_train_x,pos_train_y))),$(Flux.data(loss(neg_train_x,neg_train_y))),$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y)))\n")
		flush(lossdata)
	end
	=#

	#=
	opt = ADAM([projection, W, b], 0.001)
	for epoch in 1:100

		println("Epoch $epoch")

		neg_train_x, neg_train_y = make_dataset(rand(negative_paths, train_num), 0.0, vocabulary)
		println("Made epoch $epoch negative samples.")

		epoch_data = zip(vcat(pos_train_x,neg_train_x), vcat(pos_train_y,neg_train_y))

		#Flux.train!(loss, epoch_data, opt, cb = throttle(evalcb, 10))
		Flux.train!(loss, epoch_data, opt)

		write(lossdata,"$epoch,$(Flux.data(loss(pos_train_x,pos_train_y))),$(Flux.data(loss(neg_train_x,neg_train_y))),$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y)))\n")
		flush(lossdata)
	end
	=#

	opt = ADAM([projection, W, b], 0.0005)
	batch_size = 32

	function trainloss(d...)
		xs = [xi for (xi,yi) in d]
		ys = [yi for (xi,yi) in d]
		return loss(xs,ys)
	end

	for epoch in 1:100

		println("Epoch $epoch")

		# Just in case
		for p in (projection,W,b)
			p.grad .= 0
		end

		neg_train_x, neg_train_y = make_dataset(rand(negative_paths, train_num), 0.0, vocabulary)
		println("Made epoch $epoch negative samples.")

		epoch_data = collect(zip(vcat(pos_train_x,neg_train_x), vcat(pos_train_y,neg_train_y)))

		Flux.train!(trainloss, Base.Iterators.partition(shuffle(epoch_data),batch_size), opt)

		#### Save model
		if args["model"] != nothing
			@save args["model"] model
			println("Saved model.")
		end

		write(lossdata,"$epoch,$(Flux.data(loss(pos_train_x,pos_train_y))),$(Flux.data(loss(neg_train_x,neg_train_y))),$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y))),$(Flux.data(accuracy(pos_train_x,pos_train_y))),$(Flux.data(accuracy(neg_train_x,neg_train_y))),$(Flux.data(accuracy(pos_test_x,pos_test_y))),$(Flux.data(accuracy(neg_test_x,neg_test_y)))\n")
		flush(lossdata)
	end

	close(lossdata)

	println("Finished training.")


	#### Save model predictions
	if args["predictions"] != nothing
		open(args["predictions"],"w") do f
			write(f,"arXiv,prediction\n")
			for id in scores[:id]
				path = joinpath(source,manifest[id])
				docmat = make_docmat(path,vocabulary)
				if length(docmat) > 0
					#prediction = Flux.data(sum(sigmoid.(model(docmat))))
					prediction = sum(sigmoid.(Flux.data(model(docmat))))
					write(f,id*","*string(prediction)*"\n")
				end
			end
		end
		println("Saved model predictions.")
	end

end ## End main

main()
