using Distributed # Use julia -p NUM for processes
#addprocs(16)

using Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle, sigmoid
using Flux

import JSON
import BSON
import BSON: @save
using CSV

using Word2Vec
using GAML

using ArgParse
using ProgressMeter

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
	#dataset_x = Vector{Array{Float64,2}}()
	dataset_x::Array{Array{Float64,2},1} = @sync @distributed vcat for path in paths
		#push!(dataset_x, BSON.load(path)[:data])
		[BSON.load(path)[:data]]
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
			"-e", "--epochs"
				help = "Number of training epochs."
				arg_type = Int
				default = 100
			"-m", "--model"
				help = "Path at which to store model."
			"-y", "--predictions"
				help = "Path at which to store model predictions."
			"-x", "--exclude"
				help = "File containing ids to exclude from available data."
		end

		return parse_args(s)
	end

	args = parse_commandline()

	#### Load data
	scores = CSV.read(args["scores"],types=[String,Int])

	if args["exclude"] != nothing
		ids = [i for i in eachline(args["exclude"])]
		scores = filter(row -> !(row.id in ids), scores)
	end


	source = abspath(args["source"])
	manifest = JSON.parsefile(joinpath(source,"manifest.json"),dicttype=Dict{String,String})

	println("Opened files.")

	positive_paths = map(id -> joinpath(source,manifest[id]) , scores[scores[:score] .>= args["minscore"], :id])
	negative_paths = map(id -> joinpath(source,manifest[id]) , scores[scores[:score] .== 0, :id])

	println("Compiled paths.")

	vocabulary = wordvectors(args["wordvectors"])
	println("Loaded vocabulary.")

	###### Make datasets
	pos_num = args["numsamples"] > 0 ? min(args["numsamples"], length(positive_paths)) : length(positive_paths)

	test_num = floor(Int,args["testfrac"]*pos_num)
	train_num = pos_num - test_num

	#pos_data_x, pos_data_y = make_dataset_build(rand(positive_paths, pos_num), 1.0, vocabulary)
	pos_data_x, pos_data_y = make_dataset_load(rand(positive_paths, pos_num), 1.0)

	pos_test_x, pos_test_y = pos_data_x[1:test_num], pos_data_y[1:test_num]
	pos_train_x, pos_train_y = pos_data_x[test_num+1:end], pos_data_y[test_num+1:end]

	#neg_test_x, neg_test_y = make_dataset_build(rand(negative_paths, test_num), 0.0, vocabulary)
	neg_test_x, neg_test_y = make_dataset_load(rand(negative_paths, test_num), 0.0)

	println("Created datasets.")

	@show length(pos_train_x)
	@show length(pos_test_x)
	@show length(neg_test_x)

	#### Make model

	vocab_dim = size(vocabulary)[1]
	projected_dim = vocab_dim
	input_size = 3 * projected_dim
	output_size = 1

	#=
	projection = param(Matrix(1.0*I,projected_dim,vocab_dim)/100 .+ randn(projected_dim,vocab_dim)/1000)
	W = param(Flux.glorot_uniform(output_size,input_size))
	b = param(zeros(output_size))

	model = Chain(
			x -> [tanh.(projection * i) for i in x],
			x -> [cat(minimum(i,dims=2),maximum(i,dims=2),mean(i,dims=2),dims=1) for i in x],
			x -> hcat(x...),
			x -> W*x .+ b,
			#x -> vcat(x...)
		)
	=#
	#=
	model = Chain(
			Projection(vocab_dim,projected_dim,tanh),
			x -> [cat(minimum(i,dims=2),maximum(i,dims=2),mean(i,dims=2),dims=1) for i in x],
			x -> hcat(x...),
			Dense(input_size, output_size),
		)
	=#
	#=
	model = Chain(
			Projection(vocab_dim,projected_dim,tanh),
			minmaxmean,
			Dense(input_size, output_size),
		)
	=#
	model = Chain(
			Projection(vocab_dim,projected_dim,tanh),
			minmaxmean,
			#Dense(input_size, input_size),
			Dense(input_size, output_size),
		)

	# Print number of parameters in model
	num_params = sum([length(i) for i in params(model)])
	@show num_params

	#loss(x::Number,y::Number) = Flux.logitbinarycrossentropy(x,y) + 0.05*(sum(W.^2)+sum(b.^2)+sum(projection.^2))/2
	loss(x::Number,y::Number) = Flux.logitbinarycrossentropy(x,y) + 0.05*sum([sum(i.^2) for i in params(model)])/2
	loss(x::Array,y::Array) = sum(loss(xi,yi) for (xi,yi) in zip(model(x),y))/size(y,1)

	accuracy(x::Array,y::Array) = mean(y .== round.(sigmoid.(Flux.data(model(x)))))

	println("Created model.")

	@show accuracy(pos_test_x,pos_test_y)
	@show loss(pos_test_x,pos_test_y)

	#### Train model

	println("Setup training.")

	lossdata = open(args["lossdata"],"w")
	write(lossdata,"epoch,pos_train_loss,neg_train_loss,pos_test_loss,neg_test_loss,pos_train_acc,neg_train_acc,pos_test_acc,neg_test_acc\n")
	write(lossdata,"0,$(Flux.data(loss(pos_train_x,pos_train_y))),,$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y))),$(Flux.data(accuracy(pos_train_x,pos_train_y))),,$(Flux.data(accuracy(pos_test_x,pos_test_y))),$(Flux.data(accuracy(neg_test_x,neg_test_y)))\n")
	flush(lossdata)

	#opt = ADAM([projection, W, b], 0.00025)
	#opt = ADAM(params(model), 0.00025)
	opt = ADAM(0.00025)
	batch_size = 32

	function trainloss(d...)
		xs = [xi for (xi,yi) in d]
		ys = [yi for (xi,yi) in d]
		return loss(xs,ys)
	end

	num_epochs = args["epochs"]

	progress = Progress(num_epochs, dt=1, desc="Training: ")
	update!(progress,0)

	for epoch in 1:num_epochs

		println("Epoch $epoch")

		# Just in case
		#for p in params(model) #(projection,W,b)
		#	p.grad .= 0
		#end

		#neg_train_x, neg_train_y = make_dataset_build(rand(negative_paths, train_num), 0.0, vocabulary)
		neg_train_x, neg_train_y = make_dataset_load(rand(negative_paths, train_num), 0.0)
		#println("Made epoch $epoch negative samples.")

		epoch_data = collect(zip(vcat(pos_train_x,neg_train_x), vcat(pos_train_y,neg_train_y)))

		#Flux.train!(trainloss, Base.Iterators.partition(shuffle(epoch_data),batch_size), opt)
		Flux.train!(trainloss, params(model), Base.Iterators.partition(shuffle(epoch_data),batch_size), opt)

		#### Save model
		if args["model"] != nothing
			@save args["model"] model opt
			#println("Saved model.")
		end

		write(lossdata,"$epoch,$(Flux.data(loss(pos_train_x,pos_train_y))),$(Flux.data(loss(neg_train_x,neg_train_y))),$(Flux.data(loss(pos_test_x,pos_test_y))),$(Flux.data(loss(neg_test_x,neg_test_y))),$(Flux.data(accuracy(pos_train_x,pos_train_y))),$(Flux.data(accuracy(neg_train_x,neg_train_y))),$(Flux.data(accuracy(pos_test_x,pos_test_y))),$(Flux.data(accuracy(neg_test_x,neg_test_y)))\n")
		flush(lossdata)

		#next!(progress;showvalue)
		update!(progress, epoch, showvalues = [(:epoch,epoch), (:num_epochs,num_epochs)])
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
