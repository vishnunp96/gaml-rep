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
			"destination"
				help = "Target directory."
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

	# Find paths
	source = abspath(args["source"])
	destination = abspath(args["destination"])

	#### Load data
	scores = CSV.read(args["scores"],types=[String,Int])
	manifest = JSON.parsefile(joinpath(source,"manifest.json"),dicttype=Dict{String,String})
	println("Opened files.")

	paths = map(id -> joinpath(source,manifest[id]), scores[:id])
	println("Compiled paths.")

	vocabulary = wordvectors(args["wordvectors"])
	oov = find_oov(vocabulary)
	println("Loaded vocabulary.")

	#=
	progress = Progress(length(paths), dt=1, desc="Processing files: ")
	update!(progress,0)
	for path in paths
		docmat = get_matrix(vocabulary,abstract_string(path),oov)
		dest = splitext(replace(path, source=>destination))[1] * ".bson"
		if !isdir(dirname(dest))
			mkpath(dirname(dest))
		end
		bson(dest, data = docmat)
		next!(progress)
	end
	=#

	println("Begin with ",nworkers()," workers.")

	#=
	#progress = Progress(length(paths), dt=1, desc="Processing files: ")
	progress = Progress(length(paths))
	progress_channel = RemoteChannel(()->Channel{Bool}(length(paths)), 1)

	@sync begin
		# this task prints the progress bar
		@async while take!(progress_channel)
			yield()
			println("Take.")
			next!(progress)
			println("Taken.")
		end

		# this task does the computation
		@async begin
			@distributed for path in paths
				yield()
				docmat = get_matrix(vocabulary,abstract_string(path),oov)
				dest = splitext(replace(path, source=>destination))[1] * ".bson"
				if !isdir(dirname(dest))
					mkpath(dirname(dest))
				end
				bson(dest, data = docmat)

				#println("Saving " * dest)
				put!(progress_channel, true)
				#println("Saved " * dest)
			end
			put!(progress_channel, false) # this tells the printing task to finish
		end
	end
	=#
	#=
	progress = Progress(length(paths))
	progress_channel = RemoteChannel(()->Channel{Bool}(length(paths)), 1)

	# this task does the computation
	@sync begin
		@async begin
			@distributed for path in paths
				yield()
				docmat = get_matrix(vocabulary,abstract_string(path),oov)
				dest = splitext(replace(path, source=>destination))[1] * ".bson"
				if !isdir(dirname(dest))
					mkpath(dirname(dest))
				end
				bson(dest, data = docmat)

				#println("Saving " * dest)
				put!(progress_channel, true)
				#println("Saved " * dest)
			end
			put!(progress_channel, false) # this tells the printing task to finish
		end

		while take!(progress_channel)
			println("Take.")
			next!(progress)
			println("Taken.")
			yield()
		end
	end
	=#

	@sync @distributed for path in paths
		docmat = get_matrix(vocabulary,abstract_string(path),oov)
		dest = splitext(replace(path, source=>destination))[1] * ".bson"
		if !isdir(dirname(dest))
			mkpath(dirname(dest))
		end
		bson(dest, data = docmat)
	end

end # End main

main()
