using Distributed

function main()

	a = @sync @distributed vcat for i in 1:20
		println("$i")
		[rand(2,i)]
	end

	@show a
	@show size(a)
	@show typeof(a)

end

main()
