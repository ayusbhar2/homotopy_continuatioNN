include("utils.jl")
import .Utils as utils

using ArgParse
using Dates
using Distributions
using HomotopyContinuation
using JSON3
using Logging
using Random


Random.seed!(1234)

s = "./logs/log.txt"
io = open(s, "w+")
simple_logger = ConsoleLogger(io, show_limited=false)
global_logger(simple_logger)


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--H"
            help = "number of hidden layers"
            arg_type = Int
            default = 1
        "--dx"
            help = "length of input vector"
            arg_type = Int
            default = 2
        "--dy"
            help = "length of output vector"
            arg_type = Int
            default = 2
        "--m"
            help = "number of examples in training data"
            arg_type = Int
            default = 5
        "--di"
            help = "(fixed) number of neurons in each hidden layer"
            arg_type = Int
            default = 1
        "--a"
        	help = "value of a in Uniform(a, b)"
            arg_type = Int
            default = 0
        "--b"
        	help = "value of b in Uniform(a, b)"
            arg_type = Int
            default = 1
        "--reg"
        	help = "regularized? y/n"
        	arg_type = String
        	default = "y"
        "--runcount"
            help = "number of trials"
            arg_type = Int
            default = 1

    end

    return parse_args(s)
end


function main()

	parsed_args = parse_commandline()
	for (arg, val) in parsed_args
		println(" $arg => $val")
	end

	# ARGS
	H = parsed_args["H"];
	dx = parsed_args["dx"];
	dy = parsed_args["dy"];
	m = parsed_args["m"];
	di = parsed_args["di"];
	a = parsed_args["a"];
	b = parsed_args["b"];
	reg = parsed_args["reg"];
	runcount = parsed_args["runcount"];

	batch_output = Dict()
	batch_output["params"] = Dict(parsed_args)
	

	# # one time for generating parametrized polynomials
	# @var α₁ α₂ α₃ α₄ α₅ α₆ α₇ α₈ α₉ α₁₀ β₁ β₂ β₃ β₄ β₅ β₆ β₇ β₈ β₉ β₁₀
	# X = [α₁ α₂ α₃ α₄ α₅; α₆ α₇ α₈ α₉ α₁₀]
	# Y = [β₁ β₂ β₃ β₄ β₅; β₆ β₇ β₈ β₉ β₁₀]

	# Example 2 of paper
	# X = [7 -8 3 -5 10; -7 10 6 -2 6]
	# Y = [9 9 -8 1 10; 10 3 -8 9 10]

	# run level constants
	Unif = Uniform(a, b)	# used for constructing the Tikhonov matrices
	X = randn(dx, m)		# each column is an data point
	Y = randn(dy, m)		# each column is a target point


	@info "starting process..."

	# run level metadata
	@info "run level constants: " parsed_args a b X Y


	# generate weight matrices
	println("\ngenerating Wᵢ matrices...")
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di)
	@info "W_list: " W_list


	try
		runs = []
		for run = 1:runcount
			println("\nStarting run #: ", run)
			@info "Starting run #: " run


			# Generate Tikhonov regularization matrices
			if reg == "y"
				println("\ngenerating Λᵢ matrices...")
				Λ_list = utils.generate_Tikhonov_matrices(Unif, W_list)
				@info "Λ_list: " Λ_list
			else
				println("\nsetting Λᵢ matrices to 0...")
				Λ_list = utils.generate_zero_matrices(W_list)
				@info "Λ_list: " Λ_list
			end

			# Generate U matrices
			println("\ngenerating Uᵢ matrices...")
			U_list = utils.generate_U_matrices(W_list)
			@info "U_list: " U_list

			# Generate V matrices
			println("\ngenerating Vᵢ matrices...")
			V_list = utils.generate_V_matrices(W_list)
			@info "V_list: " V_list

			# Generate gradient polynomials
			println("\ngenerating gradient polynomials...")
			p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)	# TODO: kwargs
			println("\ntotal number of polynomials: ", length(p_list))
			@info "polynomials: " p_list
	
			# Generate the system of polynomials
			∇L = System(p_list)	# variables are ordered alphabetically
			n = nvariables(∇L)
			println("\ntotal number of variables: ", n)
			println("\nsolving the polynomial system...")


			# Solve the system and retrieve results
			retval = @timed solve(∇L)	# retval contains the result along with stats
			result = retval.value
			run_time = retval.time

			@info "result: " result
			@info "run time: " run_time

			if isnothing(result)
				throw("Solve returned nothing!")
			end

			cbb = utils.get_CBB(∇L)
			n_dm = convert(Int64, ceil(utils.get_N_DM(H, n)))
			n_r = utils.get_N_R(result)
			n_c = utils.get_N_C(result)

			run_output = Dict([("cbb", cbb), ("n_dm", n_dm), ("n_r", n_r), ("n_c", n_c)])
			push!(runs, run_output) # append output for each run
		end

		# write batch output to file
		batch_output["runs"] = runs

		open("./output/output.json", "w") do io
    		JSON3.write(io, batch_output)
		end

	catch(e)
		@error "Error while processing! " e
	finally
		@info "processing complete."
	end

end

response = @timed main()
@info "total elapsed time: " response.time



