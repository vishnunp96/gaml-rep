using Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
using Flux

using JSON
using BSON: @load
using DataFrames
using CSV

using LightXML

using ArgParse

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"model"
			help = "Flux model saved in BSON format."
			required = true
		"scores"
			help = "Location of dataset."
			required = true
		"source"
			help = "Source directory containing manifest.json."
			required = true
		"vocabulary"
			help = "Vocabulary file (dict String->Int in json format)."
			required = true
	end

	return parse_args(s)
end

args = parse_commandline()

#### Load data
scores = CSV.read(args["scores"],types=[String,Int])

source = abspath(args["source"])
manifest = JSON.parsefile(joinpath(source,"manifest.json"),dicttype=Dict{String,String})

println("Opened files.")

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

#### Open vocabulary

# Data structure and functions
vocabulary = JSON.parsefile(args["vocabulary"],dicttype=Dict{String,Int})
function getid(word)
	if haskey(vocabulary,word)
		return vocabulary[word]
	else
		return vocabulary["OOV"]
	end
end

# Helper function for creating document vectors
function make_docvec(path)
	docvec = Int[]
	for word in split(readxmltext(path))
		push!(docvec, getid(word))
	end
	return docvec
end

println("Loaded vocabulary and defined functions.")

#### Load model
@load "silvermodel.bson" model

println("Model loaded.")

#### Save model predictions
pred = DataFrame(id = scores[:id], prediction = map(i -> model(make_docvec(joinpath(source,manifest[i]))), scores[:,:id]))

CSV.write("model_predictions.csv", pred)

println("Saved model predictions.")
