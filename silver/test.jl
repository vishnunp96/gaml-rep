using ArgParse

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"scores"
			help = "Location of dataset."
			required = true
	end

	return parse_args(s)
end

args = parse_commandline()

scoresfile = abspath(args["scores"])

println(args)
println(args["scores"])
println(scoresfile)

f(x) = x^2

println(f(2))

println(f(3))
