using Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle, sigmoid
using Flux

import JSON
import BSON
import BSON: @save, @load
using CSV

using Word2Vec

using ArgParse

using LinearAlgebra
using Statistics
using Random

using LateXMLJL: abstract_string
#using GAML

### Alter this to determine how much of document is read
read_latexml(path) = abstract_string(path)

#=
function pmapreduce(mapfunc, reducefunc, iters...)
	@parallel reducefunc for arg in collect(zip(iters...))
		mapfunc(arg...)
	end
end
=#

# Helper function for making datasets from XML files
function make_dataset_build(paths, target, vocabulary)
	dataset_x = Vector{Array{Float64,2}}()
	oov = find_oov(vocabulary)
	for path in paths
		push!(dataset_x,get_matrix(vocabulary,read_latexml(path),oov))
	end
	return (dataset_x, fill(target, length(dataset_x)))
end

# Make dataset from preprocessed BSON files
function make_dataset_load(paths, target)
	dataset_x = Vector{Array{Float64,2}}()
	for path in paths
		push!(dataset_x, BSON.load(path)[:data])
	end
	return (dataset_x, fill(target, length(dataset_x)))
end

function main()

	function parse_commandline()
		s = ArgParseSettings()

		@add_arg_table s begin
			"identifiers"
				help = "Location of dataset."
				required = true
			"source"
				help = "Source directory containing manifest.json."
				required = true
			#=
			"wordvectors"
				help = "File containing word vectors to be used."
				required = true
			=#
			"model"
				help = "Path at which to store model."
				required = true
			"results"
				help = "Path at which to store model predictions."
				required = true
		end

		return parse_args(s)
	end

	args = parse_commandline()

	source = abspath(args["source"])
	manifest = JSON.parsefile(joinpath(source,"manifest.json"),dicttype=Dict{String,String})

	@load args["model"] model
	println("Loaded model.")

	identifiers = open(args["identifiers"])
	results = open(args["results"],"w")

	write(results,"arXiv,prediction\n")
	failcount = 0
	successcount = 0
	for id in eachline(identifiers)
		if haskey(manifest, id)
			path = joinpath(source,manifest[id])
			data,t = make_dataset_load([path],0)
			prediction = sigmoid(sum(Flux.data(model(data))))
			write(results,"$id,$prediction\n")
			successcount += 1
		else
			println("Could not find $id in manifest.")
			failcount += 1
		end
	end
	println("Saved model predictions.")
	println("$successcount predictions, $failcount articles could not be found.")

	close(identifiers)
	close(results)

end

main()
