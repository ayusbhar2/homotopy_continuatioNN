include("utils.jl")
import .Utils as utils

using ArgParse
using Dates
using Distributions
using HomotopyContinuation
using JSON
using Logging
using OrderedCollections
using Random


CONFIG_FILE = "config.json"
LOG_FILE = "./logs/log.txt"
OUTPUT_FILE = "./output/output.csv"
Random.seed!(1234)


io = open(LOG_FILE, "w+")
simple_logger = ConsoleLogger(io, show_limited=false)
global_logger(simple_logger)


sample_results = OrderedDict(
	"n" => -1,
	"CBB" => -1,
	"BKK" => -1,
	"N_C" => -1,
	"N_DM" => -1,
	"N_R" => -1
	)

params = ("reg", "di", "H", "m", "dx", "dy", "a", "b")


# prepare header for output file
header = "No.,"  # first column heading
for p in params
	global header = header * p * ","
end
for k in keys(sample_results)
	 global header = header * k * ","
end
header = chop(header) * "\n"


# write header to output
f = open(OUTPUT_FILE, "a")
write(f, header)
close(f)


function main()

	parsed_args = JSON.parsefile(CONFIG_FILE)
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


	# # one time for generating parametrized polynomials
	# @var α₁ α₂ α₃ α₄ α₅ α₆ α₇ α₈ α₉ α₁₀ β₁ β₂ β₃ β₄ β₅ β₆ β₇ β₈ β₉ β₁₀
	# X = [α₁ α₂ α₃ α₄ α₅; α₆ α₇ α₈ α₉ α₁₀]
	# Y = [β₁ β₂ β₃ β₄ β₅; β₆ β₇ β₈ β₉ β₁₀]

	# Example 2 of paper
	# X = [7 -8 3 -5 10; -7 10 6 -2 6]
	# Y = [9 9 -8 1 10; 10 3 -8 9 10]

	# batch level constants
	Unif = Uniform(a, b)	# used for constructing the Tikhonov matrices
	X = randn(dx, m)		# each column is an data point
	Y = randn(dy, m)		# each column is a target point


	@info "starting process..."
	@info "batch level constants: " parsed_args a b X Y

	println("\ngenerating the start system...")


	println("\ngenerating Wᵢ matrices...")
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di)
	@info "W_list: " W_list

	println("\ngenerating Uᵢ matrices...")
	U_list = utils.generate_U_matrices(W_list)
	@info "U_list: " U_list

	println("\ngenerating Vᵢ matrices...")
	V_list = utils.generate_V_matrices(W_list)
	@info "V_list: " V_list


	if reg == "y"								# TODO: make this a top level check
		try

			println("\ndefining parameters Λᵢ...")
			Λ_list = utils.generate_parameter_matrices(W_list)
			@info "Λ_list: " Λ_list


			println("\ngenerating gradient polynomials...")
			p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)	# TODO: kwargs
			println("\ntotal number of polynomials: ", length(p_list))
			@info "polynomials: " p_list


			println("\ngenerating the parametrized system...")
			parameters=collect(Iterators.flatten(Λ_list))
			∇L = System(p_list; parameters=parameters)	# variables are ordered alphabetically
			n = nvariables(∇L)
			println("\ntotal number of variables: ", n)


			println("\ngenerating initial parameter values...")
			Λ⁰_list = utils.generate_Tikhonov_matrices(Unif, W_list)
			@info "Λ⁰_list: " Λ⁰_list


			println("\nsolving the initial system (polyhedral)...")
			λ_start = collect(Iterators.flatten(Λ⁰_list))
			retval = @timed solve(∇L; 
								  target_parameters=λ_start,
								  threading=true
					)	# retval contains the result along with stats
			result = retval.value
			if isnothing(result)
				throw("Solve returned nothing!")
			end
			run_time = retval.time
			@info "result: " result
			@info "run time: " run_time


			println("\ncollecting sample results...")
			sample_results["n"] = n
			sample_results["CBB"] = utils.get_CBB(∇L)
			sample_results["N_C"] = utils.get_N_C(result)
			sample_results["N_DM"] = convert(Int64, ceil(utils.get_N_DM(H, n)))
			sample_results["N_R"] = utils.get_N_R(result)
			@info "sample results: " sample_results


			println("\nwriting sample results to file...")
			row = string(run) * "," #  run number
			for p in params
				row = row * string(parsed_args[p]) * ","
			end
			for (k, v) in sample_results		# key order is fixed
				 row = row * string(v) * ","
			end
			row = chop(row) * "\n"

			f = open(OUTPUT_FILE, "a")
			write(f, row)
			close(f)


			if runcount > 1
				for run = 2:runcount

					println("generating target parameter values...")
					Λ¹_list = generate_Tikhonov_matrices(U, W_list)
					@info "Λ¹_list: " Λ¹_list

					println("solving target system with parameter homotopy...")
					λ_target = collect(Iterators.flatten(Λ¹_list))
					retval = @timed solve(∇L, solutions(result);
										  start_parameters=λ_start,
										  target_parameters=λ_target,
										  threading=true
							)	# retval contains the result along with stats
					result = retval.value
					run_time = retval.time

					if isnothing(result)
						throw("Solve returned nothing!")
					end

					@info "result: " result
					@info "run time: " run_time
					
				end
			end

		catch(e)
			@error "Error while processing! " e
		finally
			close(f)
			@info "processing complete."
		end

	end

end

response = @timed main()
@info "total elapsed time: " response.time



