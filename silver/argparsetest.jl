using ArgParse

function main()

	function parse_commandline()
		s = ArgParseSettings()

		@add_arg_table s begin
			"source"
				help = "Source directory containing manifest.json."
				required = true
			"-y", "--predictions"
				help = "Path at which to store model predictions."
		end

		return parse_args(s)
	end

	args = parse_commandline()

	for (arg,val) in args
		println("  $arg  =>  $val")
	end

	if args["predictions"] == nothing
		println("No predictions")
	end
	if args["predictions"] != nothing
		println("Predictions")
	end

end ## End main

main()

