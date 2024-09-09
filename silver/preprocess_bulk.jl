using Distributed

using ArgParse

@everywhere using Word2Vec
@everywhere using LateXMLJL: abstract_string

import JSON
@everywhere import BSON: bson
import CSV

@everywhere using ProgressMeter

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
			"output"
				help = "Filepath for storing output data in BSON format."
				required = true
			"wordvectors"
				help = "File containing word vectors to be used."
				required = true
			#=
			"processes"
				help = "Number of processes to utilise in pre-processing."
				arg_type = Int
			=#
		end

		return parse_args(s)
	end

	args = parse_commandline()

	source = abspath(args["source"])

	#### Load data
	scores = CSV.read(args["scores"],types=[String,Int])
	manifest = JSON.parsefile(joinpath(source,"manifest.json"),dicttype=Dict{String,String})
	println("Opened files.")

	#paths = map(id -> joinpath(source,manifest[id]), scores[:id])
	#println("Compiled paths.")

	vocabulary = wordvectors(args["wordvectors"])
	oov = find_oov(vocabulary)
	println("Loaded vocabulary.")

	println("Begin with ",nworkers()," workers.")

	data = @sync @distributed vcat for id in scores[:id]
		(id, get_matrix(vocabulary,abstract_string(joinpath(source,manifest[id])),oov))
	end

	data_dict = Dict(Symbol(id) => matrix for (id,matrix) in data)

	bson(args["output"], data_dict)

end # End main

main()
