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



	## ~ SETUP ~ ##

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



	## ~ PRE-PROCESSING ~ ##

	H = parsed_args["H"]
	di = parsed_args["di"]
	dx = parsed_args["dx"]
	dy = parsed_args["dy"]
	m = parsed_args["m"]

	regularize = parsed_args["reg"]
	reg_parameterized = parsed_args["reg_parameterized"]

	x_parameterized = parsed_args["x_parameterized"]
	x_dist_params = parsed_args["x_dist_params"]

	y_parameterized = parsed_args["y_parameterized"]
	y_dist_params = parsed_args["y_dist_params"]

	start_params_complex = parsed_args["start_params_complex"]

	a = parsed_args["reg_dist_params"][1]
	b = parsed_args["reg_dist_params"][2]

	μx = x_dist_params[1]
	σx = x_dist_params[2]
	Nx = Normal(μx, σx)

	μy = x_dist_params[1]
	σy = x_dist_params[2]
	Ny = Normal(μy, σy)

	
	println("\ngenerating Wᵢ matrices...")
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di)
	@info "W_list: " W_list

	println("\ngenerating Uᵢ matrices...")
	U_list = utils.generate_U_matrices(W_list)
	@info "U_list: " U_list

	println("\ngenerating Vᵢ matrices...")
	V_list = utils.generate_V_matrices(W_list)
	@info "V_list: " V_list

	parameters = []
	Λ_list = []
	X = Matrix{}
	Y = Matrix{}
	reg_param_count = 0

	if regularize
		if reg_parameterized
			println("\ngenerating parameterized Λᵢ matrices...")
			Λ_list = utils.generate_parameterized_Tikhonov_matrices(W_list)
			push!(parameters, collect(Iterators.flatten(Λ_list)))
		else
			println("\ngenerating real Λᵢ matrices...")
			Λ_list = utils.generate_real_Tikhonov_matrices(a, b, W_list)
		end
	end
	@info "Λ_list: " Λ_list

	if x_parameterized
		println("\ngenerating parameterized X matrix...")
		X = utils.generate_parameter_matrix(dx, m, "x")
		push!(parameters, collect(Iterators.flatten(X)))
	else
		println("\ngenerating real X matrix...")
		X = rand(Nx, (dx, m))
	end
	@info "X: " X

	if y_parameterized
		println("\ngenerating parameterized Y matrix...")
		Y = utils.generate_parameter_matrix(dx, m, "y")
		push!(parameters, collect(Iterators.flatten(Y)))
	else
		println("\ngenerating real Y matrix...")
		Y = rand(Ny, (dy, m))
	end
	@info "Y: " Y

	println("\ngenerating the polynomial system...")
	p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)
	@info "polynomials: " p_list

	parameters = collect(Iterators.flatten(parameters))
	∇L = System(p_list; parameters=parameters)	# variables are ordered lexicographically

	println("\ntotal number of polynomials: ", length(p_list))
	println("\ntotal number of variables: ", nvariables(∇L))
	println("\ntotal number of parameters: ", length(parameters))



	## ~ STAGE 1 ~ ##

	println("\nSTAGE # 1 ...")
	println("\nParameter Homotopy: assigning start values...")
	@info "Parameter Homotopy: assigning start values..."

	start_values = []

	if reg_parameterized
		if start_params_complex
			Λ₀_list = utils.generate_complex_Tikhonov_matrices(Λ_list)
		else
			Λ₀_list = utils.generate_real_Tikhonov_matrices(a, b, Λ_list)
		end
		push!(start_values, collect(Iterators.flatten(Λ₀_list)))
		# @info "Λ₀_list: " Λ₀_list
	end

	if x_parameterized
		if start_params_complex
			X₀ = randn(ComplexF64, size(X))
		else
			X₀ = rand(Nx, size(X))
		end
		push!(start_values, collect(Iterators.flatten(X₀)))
		# @info "X₀: " X₀
	end

	if y_parameterized
		if start_params_complex
			Y₀ = randn(ComplexF64, size(Y))
		else
			Y₀ = rand(Ny, size(Y))
		end
		push!(start_values, collect(Iterators.flatten(Y₀)))
		# @info "Y₀: " Y₀
	end
	start_values = collect(Iterators.flatten(start_values))

	@info "parameters(∇L) " parameters(∇L)
	@info "start_values " start_values


	run = 1
	@info "run # " run

	println("\nParameter Homotopy: solving the initial system (polyhedral)...")
	retval = @timed solve(∇L; target_parameters=start_values, threading=true)

	result0 = retval.value
	solve_time = retval.time

	@info "solve_time: " solve_time
	@info "result: " result0
	@info "solutions: " solutions(result0)

	println("\ncollecting sample results...")
	global sample_results = utils.collect_results(sample_results, parsed_args,
												  ∇L, result0)
	@info "sample results: " sample_results

	println("\nwriting sample results to file...")
	row = string(run) * "," #  run number
	for p in keys(parsed_args)
		row = row * replace(string(parsed_args[p]), "," => "") * ","
	end
	for (k, v) in sample_results		# key order is fixed
		 row = row * string(v) * ","
	end
	row = chop(row) * "\n"

	f = open(OUTPUT_FILE, "a")
	write(f, row)



	# ## ~ STAGE 2 ~ ##

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



