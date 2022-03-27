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
	X = randn(dx, m)		# each column is an data point
	Y = randn(dy, m)		# each column is a target point


	# generate weight matrices
	println("generating Wᵢ matrices...")
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di)

	
	f = open("log.txt", "w") # TODO: move this to a logging function
	try
		for run = 1:runs

			# Generate Tikhonov regularization matrices
			println("generating Λᵢ matrices...")
			Λ_list = utils.generate_Tikhonov_matrices(Unif, W_list)
			# println("a total of ", length(Λ_list), " matrices generated.")

			# Generate U matrices
			println("generating Uᵢ matrices...")
			U_list = utils.generate_U_matrices(W_list)
			# println("a total of ", length(U_list), " matrices generated.")

			# Generate V matrices
			println("generating Vᵢ matrices...")
			V_list = utils.generate_V_matrices(W_list)
			# println("a total of ", length(V_list), " matrices generated.")

			# Generate gratiend polynomials
			println("generating gradient polynomials...")
			p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)	# TODO: kwargs
			println("total number of polynomials: ", length(p_list))


			# Generate the system of polynomials
			∇L = System(p_list)	# variables are ordered alphabetically
			n = nvariables(∇L)
			println("total number of variables: ", n)

			println("solving the polynomial system...")
			result = solve(∇L)

			cbb = utils.get_CBB(∇L)
			n_dm = convert(Int64, ceil(utils.get_N_DM(H, n)))
			n_r = utils.get_N_R(result)
			n_c = utils.get_N_C(result)

			println("writing output to file...")
			write(f, string("No. \t H \t dx \t dy \t m \t a \t b \t n \t CBB \t N_C \t N_DM \t N_R\n"))
			write(f, string(run, "\t", H, "\t", dx, "\t", dy, "\t", m, "\t", a, "\t", b, "\t", n, "\t", cbb, "\t", n_c, "\t", n_dm, "\t", n_r, "\n"))

		end
			
	finally
		close(f)
	end

end

main()

