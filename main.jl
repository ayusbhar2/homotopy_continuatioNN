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
# Random.seed!(1234)


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


	Dist = Uniform(a, b)	# used for constructing the Tikhonov matrices
	X = randn(dx, m)		# each column is an data point
	Y = randn(dy, m)		# each column is a target point


	@info "starting process..."
	@info "batch level constants: " parsed_args a b X Y




## ~ pre-processing ~ ##
	
	println("\ngenerating Wᵢ matrices...")
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di)
	@info "W_list: " W_list

	println("\ngenerating Uᵢ matrices...")
	U_list = utils.generate_U_matrices(W_list)
	@info "U_list: " U_list

	println("\ngenerating Vᵢ matrices...")
	V_list = utils.generate_V_matrices(W_list)
	@info "V_list: " V_list

	println("\ndefining parameters Λᵢ...")
	Λ_list = utils.generate_parameter_matrices(W_list)
	@info "Λ_list: " Λ_list

	println("\ngenerating gradient polynomials...")
	p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)	# TODO: kwargs
	println("\ntotal number of polynomials: ", length(p_list))
	@info "polynomials: " p_list

	println("\ngenerating the parametrized system...")
	parameters=collect(Iterators.flatten(Λ_list))
	∇L = System(p_list; parameters=parameters)	# variables are ordered lexicographically
	n = nvariables(∇L)
	println("\ntotal number of variables: ", n)
	println("\ntotal number of parameters: ", length(parameters))

	println("\nassigning initial parameter values...")
	Λ⁰_list = utils.generate_Tikhonov_matrices(Dist, W_list)
	# Λ⁰_list = utils.generate_complex_Tikhonov_matrices(W_list)

	@info "Λ⁰_list: " Λ⁰_list
	λ_start = collect(Iterators.flatten(Λ⁰_list))
	@info "start parameters: " λ_start




## ~ Solving initial system ~ ##

	run = 1
	println("\nwriting sample results to file...")
	@info "run # " run

	println("\nsolving the initial system (polyhedral)...")
	retval = @timed solve(∇L; 
						  target_parameters=λ_start,
						  threading=true
			)
	result0 = retval.value
	solve_time = retval.time

	@info "result: " result0
	@info "solutions: " solutions(result0)
	# @info "solve_time: " solve_time

	println("\ncollecting sample results...")
	sample_results["n"] = n
	sample_results["CBB"] = utils.get_CBB(∇L)
	sample_results["N_C"] = utils.get_N_C(result0)
	sample_results["N_DM"] = convert(Int64, ceil(utils.get_N_DM(H, n)))
	sample_results["N_R"] = utils.get_N_R(result0)
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




## ~ Parameter homotopy for subsequent systems ~ ##

	try
		if runcount > 1
			λ_target_list = []
			for i = 2:runcount
				Λ¹_list = utils.generate_Tikhonov_matrices(Dist, W_list)
				# Λ¹_list = utils.generate_complex_Tikhonov_matrices(W_list)
				λ_target = collect(Iterators.flatten(Λ¹_list))
				push!(λ_target_list, λ_target)
			end

			@info "starting parameter homotopy..."
			@info "target parameter list: " λ_target_list
			retval = @timed solve(∇L, solutions(result0);
								  start_parameters=λ_start,
								  target_parameters=λ_target_list,
								  threading=true)
			result_list = retval.value
			solve_time = retval.time
			# @info "solve_time: " solve_time


			for result in result_list
				run += 1
				@info "run # " run
				@info "result: " result[1]
				@info "solutions: " solutions(result[1])

				println("\ncollecting sample results...")
				sample_results["N_C"] = utils.get_N_C(result[1])
				sample_results["N_DM"] = convert(Int64, ceil(utils.get_N_DM(H, n)))
				sample_results["N_R"] = utils.get_N_R(result[1])
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

				write(f, row)
			end

		end
	catch(e)
		println(e)
		@error e
	finally
		close(f)
	end

end

response = @timed main()
@info "total elapsed time: " response.time



