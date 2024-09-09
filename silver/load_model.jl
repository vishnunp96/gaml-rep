using Flux.Tracker
using Flux

using BSON: @load

#function reload()

	x = [1,2]

	#### Reload model
	@load "mymodel.bson" model

	#### Use loaded model
	println(model(x))

#end

#reload()
