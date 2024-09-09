using Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle, sigmoid
using Flux

import JSON
using BSON: @save
using CSV

using Word2Vec

using ArgParse

using LinearAlgebra
using Statistics
using Random

using LateXMLJL: abstract_string

### Alter this to determine how much of document is read
read_latexml(path) = abstract_string(path)

#=
function pmapreduce(mapfunc, reducefunc, iters...)
	@parallel reducefunc for arg in collect(zip(iters...))
		mapfunc(arg...)
	end
end
=#

# Helper function for making datasets
function make_dataset(paths, target, vocabulary)
	dataset_x = Vector{Array{Float64,2}}()
	oov = find_oov(vocabulary)
	for path in paths
		push!(dataset_x,get_matrix(vocabulary,read_latexml(path),oov))
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

