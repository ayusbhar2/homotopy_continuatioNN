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


function generate_row(col1, type, parsed_args, sample_results)
	row = string(col1) * ","

	for (k, v) in parsed_args
		if type == "row"
			row = row * replace(string(v), "," => "") * ","
		elseif type == "header"
			row = row * string(k) * ","
		end
	end
	for (k, v) in sample_results
		if type == "row"
			row = row * string(v) * ","
		elseif type == "header"
			row = row * string(k) * ","
		end
	end
	row = chop(row) * "\n"
	return row
end



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

	# write output file header
	header = generate_row("No.", "header", parsed_args, sample_results)
	f = open(OUTPUT_FILE, "a")
	write(f, header)
	close(f)



	## ~ PRE-PROCESSING ~ ##

	runcount = parsed_args["runcount"]

	H = parsed_args["H"]
	di = parsed_args["di"]
	dx = parsed_args["dx"]
	dy = parsed_args["dy"]
	m = parsed_args["m"]
	first_layer_conv = parsed_args["first_layer_conv"]
	stride = parsed_args["stride"]
	width = parsed_args["width"]

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
	W_list = utils.generate_weight_matrices(H, dx, dy, m, di;
		first_layer_conv=first_layer_conv, stride=stride, width=width)
	@info "W_list: " W_list

	println("\ngenerating Uᵢ matrices...")
	U_list = utils.generate_U_matrices(W_list)
	@info "U_list: " U_list

	println("\ngenerating Vᵢ matrices...")
	V_list = utils.generate_V_matrices(W_list)
	@info "V_list: " V_list



	## ~ PARAMETERS ~ ##

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
	else
		Λ_list = utils.generate_zero_matrices(W_list)
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
		Y = utils.generate_parameter_matrix(dy, m, "y")
		push!(parameters, collect(Iterators.flatten(Y)))
	else
		println("\ngenerating real Y matrix...")
		Y = rand(Ny, (dy, m))
	end
	@info "Y: " Y

	if length(parameters) == 0
		@error "No parameters specified for Parameter Homotopy!"
		error("No parameters specified for Parameter Homotopy!")
	else
		parameters = collect(Iterators.flatten(parameters))
	end



	## ~ SYSTEM ~ ##

	println("\ngenerating the polynomial system...")

	if first_layer_conv
		p_list = utils.generate_gradient_polynomials_with_convolution(W_list, Λ_list, X, Y)
	else
		p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)
	end
	@info "polynomials: " p_list

	∇L = System(p_list; parameters=parameters)	# variables are ordered lexicographically

	println("\ntotal number of polynomials: ", length(p_list))
	println("\ntotal number of variables: ", nvariables(∇L))
	println("\ntotal number of parameters: ", length(parameters))



	## ~ STAGE 1 ~ ##

	println("\nSTAGE # 1 ...")

	run = 1
	@info "run # " run

	println("\nParameter Homotopy: assigning start values...")
	@info "Parameter Homotopy: assigning start values..."

	start_params = utils.generate_param_values(a, b, Nx, Ny, regularize,
		reg_parameterized, x_parameterized, y_parameterized,
		Λ_list, X, Y; complex=start_params_complex) #  start params should be complex

	@info "parameters(∇L) " parameters(∇L)
	@info "start_params " start_params

	println("\nParameter Homotopy: solving the initial system (polyhedral)...")
	retval = @timed solve(∇L; target_parameters=start_params, threading=true)

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



	## ~ STAGE 2 ~ ##


	try
		if runcount > 1
			println("\nSTAGE # 2...")
			for run = 2:runcount
		
					@info "run # " run

					println("\nParameter Homotopy: generating target params...")
					@info "Parameter Homotopy: generating target params..."

					target_params = utils.generate_param_values(a, b, Nx, Ny, regularize,
						reg_parameterized, x_parameterized, y_parameterized,
						Λ_list, X, Y; complex=false) # subsequent params should be real
					@info "target parameter list: " target_params

					retval = @timed solve(∇L, solutions(result0);
										  start_parameters=start_params,
										  target_parameters=target_params,
										  threading=true)
					result = retval.value
					solve_time = retval.time

					@info "solve_time: " solve_time
					@info "result: " result
					@info "solutions: " solutions(result)

					println("\ncollecting sample results...")
					global sample_results = utils.collect_results(
						sample_results, parsed_args, ∇L, result)
					@info "sample results: " sample_results

					println("\nwriting sample results to file...")
					row = generate_row(string(run), "row", parsed_args, sample_results)
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



