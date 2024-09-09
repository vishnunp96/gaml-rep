using Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated

using JSON
using BSON: @save
using DataFrames
using CSV

using LightXML

using ArgParse

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
		"-v", "--mincount"
			help = "Minimum frequency in corpus for token to be included in vocabulary."
			arg_type = Int
			default = 1
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

### XML functions

function get_elem_text(elem)
	text = content(elem)
	for i in child_elements(elem)
		text = text * " " * get_elem_text(i)
	end
	return text
end

function readxmltext(path)
	doc = parse_file(path)
	lines = get_elem_text(root(doc))
	free(doc)
	return lines
end

#### Create vocabulary

# Data structure and functions
vocabulary = Dict{String,Int}("OOV"=>1)
function getid(word)
	if haskey(vocabulary,word)
		return vocabulary[word]
	else
		return vocabulary["OOV"]
	end
end

# Count word occurances for finding OOV tokens
wordcounts = Dict{String,Int}()
for path in map(id -> joinpath(source,manifest[id]), scores[:,:id])
	for word in split(readxmltext(path))
		if haskey(wordcounts,word)
			wordcounts[word] += 1
		else
			wordcounts[word] = 1
		end
	end
end

# Construct final vocabulary
for (token,count) in wordcounts
	if count >= args["mincount"]
		vocabulary[word] = length(vocabulary) + 1
	end
end

# Helper function for creating document vectors
function make_docvec(path)
	docvec = Int[]
	for word in split(readsmltext(path))
		push!(docvec, getid(word))
	end
	return docvec
end

println("Defined functions and created vocabulary.")

# Save vocabulary somehow?

#### Make model

vocab_size = length(vocabulary)
embedding_dim = 32
input_size = 3 * embedding_dim
output_size = 1

embedding = param(rand(embedding_dim,vocab_size))
W = param(rand(output_size,input_size))
b = param(rand(output_size))

model = Chain(
		x -> embedding[:,x],
		x -> cat(1,minimum(x,2),maximum(x,2),mean(x,2)),
		x -> W*x .+ b
	)


loss(x,y) = Flux.mse(model(x),y)
accuracy(x, y) = mean(argmax(model(x)) .== argmax(y))

println("Created model.")

#### Train model

# Make positive dataset
pos_data = []
for path in positive_paths
	push!(pos_data,(make_docvec(path),1))
end
pos_num = length(pos_data)

optimiser = ADAM(params(model))

println("Setup training.")

for epoch in 1:10

	println("Epoch 1")

	neg_data = []
	for path in rand(negative_paths, pos_num)
		push!(neg_data,(make_docvec(path),0))
	end

	Flux.train!(loss, [pos_data; neg_data], optimiser, cb = throttle(() -> @show(loss(X, Y)), 10))
end

println("Finished training.")

#### Test model?
# Report test accuracy

#accuracy(X, Y)

#### Save model
@save "silvermodel.bson" model

println("Saved model.")

#### Save model predictions
pred = DataFrame(id = scores[:id], prediction = map(i => model(make_docvec(joinpath(source,manifest[i]))), scores[:,:id]))

CSV.write("model_predictions.csv", pred)

println("Saved model predictions.")
