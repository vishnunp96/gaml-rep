using Flux.Tracker
using Flux

using BSON: @save

function make()

	#### Make data point
	x = [1,2,3,4,5]

	#### Make model

	vocab_size = 10
	embedding_dim = 5
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

	#### Test model
	println(model(x))

	#### Save model
	@save "mymodel.bson" model

end

#function make2()

	#### Make data point
	x = [1,2]

	#### Make model
	W = param(rand(1,2))
	b = param(rand(1))

	model(x) = W*x .+ b

	#### Test model
	println(model(x))

	#### Save model
	@save "mymodel.bson" model

#end

#make2()
