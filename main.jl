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


function main()

	@info "START..."


	## ~ setup ~ ##

	batch_args = JSON.parsefile(CONFIG_FILE)
	s = sort(collect(batch_args); by=x->x[1])
	parsed_args = OrderedDict(s)

	for (k, v) in parsed_args
		println("$k => $v")
	end
	@info "parsed_args: " parsed_args

	# prepare header for output file
	header = "No.,"  # first column heading
	for p in keys(parsed_args)
		header = header * p * ","
	end
	for k in keys(sample_results)
		 header = header * k * ","
	end
	header = chop(header) * "\n"

	# write header to output
	f = open(OUTPUT_FILE, "a")
	write(f, header)
	close(f)



	## ~ pre-processing ~ ##

	H = parsed_args["H"]
	di = parsed_args["di"]
	dx = parsed_args["dx"]
	dy = parsed_args["dy"]
	m = parsed_args["m"]
	
	println("\ngenerating Wᵢ matrices...")
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di)
	@info "W_list: " W_list

	println("\ngenerating Uᵢ matrices...")
	U_list = utils.generate_U_matrices(W_list)
	@info "U_list: " U_list

	println("\ngenerating Vᵢ matrices...")
	V_list = utils.generate_V_matrices(W_list)
	@info "V_list: " V_list

	# parameters = []

	# if reg_status
	# 	if reg_param
	# 		println("\ngenerating parameterized Λᵢ matrices...")
	# 		Λ_list = utils.generate_parameterized_Tikhonov_matrices(W_list)
	# 		@info "Λ_list: " Λ_list

	# 		push!(parameters,collect(Iterators.flatten(Λ_list)))

	# 	elseif reg_type == "real"
	# 		println("\ngenerating real Λᵢ matrices...")
	# 		Λ_list = utils.generate_real_Tikhonov_matrices(Λ_dist, W_list)
	# 		@info "Λ_list: " Λ_list
	# 	else
	# 		println("\ngenerating complex Λᵢ matrices...")
	# 		Λ_list = utils.generate_complex_Tikhonov_matrices(W_list)
	# 		@info "Λ_list: " Λ_list
	# 	end
	# end

# 	if X_param
# 		println("\ngenerating parameterized X matrix...")
# 		X = utils.generate_parameter_matrix(dx, m, "x")
# 		@info "X: " X

# 		push!(parameters,collect(Iterators.flatten(X)))
# 	else
# 		println("\ngenerating real X matrix...")
# 		X = randn(dx, m)		# each column is an data point
# 		@info "X: " X
# 	end

# 	if Y_param
# 		println("\ngenerating parameterized Y matrix...")
# 		Y = utils.generate_parameter_matrix(dx, m, "y")
# 		@info "Y: " Y

# 		push!(parameters,collect(Iterators.flatten(Y)))
# 	else
# 		println("\ngenerating real Y matrix...")
# 		Y = randn(dy, m)		# each column is an data point
# 		@info "Y: " Y
# 	end

# 	println("\ngenerating gradient equations...")
# 	p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)	# TODO: kwargs
# 	@info "polynomials: " p_list

# 	println("\ndefiing the parametrized system...")
# 	parameters = collect(Iterators.flatten(parameters))
# 	∇L = System(p_list; parameters=parameters)	# variables are ordered lexicographically
# 	n = nvariables(∇L)

# 	println("\ntotal number of polynomials: ", length(p_list))
# 	println("\ntotal number of variables: ", n)
# 	println("\ntotal number of parameters: ", length(parameters))



## ~ Solving initial system ~ ##

	# run = 1
	# println("\nrun # ", run)
	# @info "run # " run

	# if reg_param
	# 	println("\nassigning initial parameter values to Λᵢ...")
	# 	Λ⁰_list = utils.generate_Tikhonov_matrices(Dist, W_list)
	# 	# Λ⁰_list = utils.generate_complex_Tikhonov_matrices(W_list)

# 	@info "Λ⁰_list: " Λ⁰_list
# 	λ_start = collect(Iterators.flatten(Λ⁰_list))
# 	@info "start parameters: " λ_start

# 	println("\nsolving the initial system (polyhedral)...")
# 	retval = @timed solve(∇L; target_parameters=λ_start, threading=true)

# 	result0 = retval.value
# 	solve_time = retval.time

# 	@info "result: " result0
# 	@info "solutions: " solutions(result0)
# 	# @info "solve_time: " solve_time

# 	println("\ncollecting sample results...")
# 	global sample_results = utils.collect_results(sample_results, parsed_args,
# 												  ∇L, result0)
# 	@info "sample results: " sample_results

# 	println("\nwriting sample results to file...")
# 	row = string(run) * "," #  run number
# 	for p in params
# 		row = row * string(parsed_args[p]) * ","
# 	end
# 	for (k, v) in sample_results		# key order is fixed
# 		 row = row * string(v) * ","
# 	end
# 	row = chop(row) * "\n"

# 	f = open(OUTPUT_FILE, "a")
# 	write(f, row)




# ## ~ Parameter homotopy for subsequent systems ~ ##

# 	try
# 		if runcount > 1
# 			λ_target_list = []
# 			for i = 2:runcount
# 				Λ¹_list = utils.generate_Tikhonov_matrices(Dist, W_list)
# 				# Λ¹_list = utils.generate_complex_Tikhonov_matrices(W_list)
# 				λ_target = collect(Iterators.flatten(Λ¹_list))
# 				push!(λ_target_list, λ_target)
# 			end

# 			@info "starting parameter homotopy..."
# 			@info "target parameter list: " λ_target_list
# 			retval = @timed solve(∇L, solutions(result0);
# 								  start_parameters=λ_start,
# 								  target_parameters=λ_target_list,
# 								  threading=true)
# 			result_list = retval.value
# 			solve_time = retval.time
# 			# @info "solve_time: " solve_time


# 			for result in result_list
# 				run += 1
# 				println("\nrun # ", run)
# 				@info "run # " run
# 				@info "result: " result[1]
# 				@info "solutions: " solutions(result[1])

# 				println("\ncollecting sample results...")
# 				global sample_results = utils.collect_results(
# 					sample_results, parsed_args, ∇L, result[1])
# 				@info "sample results: " sample_results

# 				println("\nwriting sample results to file...")
# 				row = string(run) * "," #  run number
# 				for p in params
# 					row = row * string(parsed_args[p]) * ","
# 				end
# 				for (k, v) in sample_results		# key order is fixed
# 					 row = row * string(v) * ","
# 				end
# 				row = chop(row) * "\n"
# 				write(f, row)
# 			end
# 		end
# 	catch(e)
# 		println(e)
# 		@error e
# 	finally
# 		close(f)
# 	end

end

response = @timed main()
@info "total elapsed time: " response.time



