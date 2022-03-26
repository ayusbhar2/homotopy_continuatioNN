include("utils.jl")
import .Utils as utils

using ArgParse
using Distributions
using HomotopyContinuation
using Random


Random.seed!(1234)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "H"
            help = "number of hidden layers"
            arg_type = Int
            required = true
        "dx"
            help = "length of input vector"
            arg_type = Int
            required = true
        "dy"
            help = "length of output vector"
            arg_type = Int
            required = true
        "m"
            help = "number of examples in training data"
            arg_type = Int
            required = true
        "di"
            help = "(fixed) number of neurons in each hidden layer"
            arg_type = Int
            required = true
        "--runs", "-r"
            help = "number of trials"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end


function main()
	parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end

	# ARGS
	H = parsed_args["H"];
	dx = parsed_args["dx"];
	dy = parsed_args["dy"];
	m = parsed_args["m"];
	di = parsed_args["di"];
	runs = parsed_args["runs"];

	# configs
	a = 0; b= 1;

	# module level constants
	Unif = Uniform(a, b)	# used for constructing the Tikhonov matrices
	X = [[7 -8 3 -5 10];	# each column is a single example
		 [-7 10 6 -2 6]]

	Y = [[9 9 -8 1 10];		# each column is a single target
		 [10 3 -8 9 10]]


	# generate weight matrices
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di)
	W = reduce(*, reverse(W_list))	# weight matrices are multiplied in reverse order
	println("******** W: ", size(W))

	
	f = open("log.txt", "w") # TODO: move this to a logging function
	for run = 1:runs

		# Generate Tikhonov regularization matrices
		Λ_list = utils.generate_Tikhonov_matrices(Unif, W_list)

		# Generate U matrices
		U_list = utils.generate_U_matrices(W_list)

		# Generate V matrices
		V_list = utils.generate_V_matrices(W_list)

		
		# Generate gratiend polynomials
		p_list = Any[]
		for i = 1:length(W_list)
			∂L_Wᵢ = transpose(U_list[i]) * (W * X * transpose(X) - Y * transpose(X)) * transpose(V_list[i]) + Λ_list[i] .* W_list[i]
			# println("∂L_Wᵢ :", size(∂L_Wᵢ))

			v = vec(∂L_Wᵢ)
			# println("v: ", v)
			for p in v
				# println("p: ", p)
				push!(p_list, p)
			end
		end

		# Generate the system of polynomials
		∇L = System(p_list)	# variables are ordered alphabetically

		result = solve(∇L)

		cbb = utils.get_CBB(∇L)
		n_dm = convert(Int64, trunc(utils.get_N_DM(1, nvariables(∇L))))
		n_r = utils.get_N_R(result)
		n_c = utils.get_N_C(result)


		write(f, string("No. \t H \t dx \t dy \t m \t a \t b \t CBB \t N_DM \t N_R \t N_C\n"))
		write(f, string(run, "\t", H, "\t", dx, "\t", dy, "\t", m, "\t", a, "\t", b, "\t", cbb, "\t", n_dm, "\t", n_r, "\t", n_c, "\n"))
		close(f)
	end

end

main()

