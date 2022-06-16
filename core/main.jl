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


io = open(LOG_FILE, "w+")
simple_logger = ConsoleLogger(io, Logging.Debug; show_limited=false)
global_logger(simple_logger)


sample_results = OrderedDict(
	"variables"=>"",
	"n" => -1,
	"CBB" => -1,
	"BKK" => -1,
	"N_C" => -1,
	"N_DM" => -1,
	"N_R" => -1,
	"Real_sols" => "",
	"All_sols" => "",
	"L_values" => "",
	"Idx_vals" => ""
	)


function generate_row(col1, type, parsed_args, sample_results)
	delim = "&"
	row = string(col1) * delim

	for (k, v) in parsed_args
		if type == "row"
			row = row * string(v) * delim
		elseif type == "header"
			row = row * string(k) * delim
		end
	end
	for (k, v) in sample_results
		if type == "row"
			row = row * string(v) * delim
		elseif type == "header"
			row = row * string(k) * delim
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

	fix_seed = parsed_args["fix_seed"]
	if fix_seed
		Random.seed!(1234)
	end
	runcount = parsed_args["runcount"]

	H = parsed_args["H"]
	di = parsed_args["di"]
	dx = parsed_args["dx"]
	dy = parsed_args["dy"]
	m = parsed_args["m"]
	convolution = parsed_args["convolution"]
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
		convolution=convolution, stride=stride, width=width)
	@debug "W_list: " W_list



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
	@debug "Λ_list: " Λ_list

	if x_parameterized
		println("\ngenerating parameterized X matrix...")
		X = utils.generate_parameter_matrix(dx, m, "x")
		push!(parameters, collect(Iterators.flatten(X)))
	else
		println("\ngenerating real X matrix...")
		X = rand(Nx, (dx, m))
	end
	@debug "X: " X

	if y_parameterized
		println("\ngenerating parameterized Y matrix...")
		Y = utils.generate_parameter_matrix(dy, m, "y")
		push!(parameters, collect(Iterators.flatten(Y)))
	else
		println("\ngenerating real Y matrix...")
		Y = rand(Ny, (dy, m))
	end
	@debug "Y: " Y

	if length(parameters) == 0
		@error "No parameters specified for Parameter Homotopy!"
		error("No parameters specified for Parameter Homotopy!")
	else
		parameters = collect(Iterators.flatten(parameters))
	end




	## ~ SYSTEM ~ ##

	
	println("\ngenerating the loss function...")
	L = utils.generate_loss_func(W_list, Λ_list, X, Y)
	variables = utils.extract_and_sort_variables(W_list)
	grad_L = differentiate(L, variables)
	∇²L = differentiate(grad_L, variables)

	println("\ngenerating the polynomial system...")
	if convolution
		p_list = grad_L
	else
		println("\ngenerating Uᵢ matrices...")
		U_list = utils.generate_U_matrices(W_list)
		@debug "U_list: " U_list

		println("\ngenerating Vᵢ matrices...")
		V_list = utils.generate_V_matrices(W_list)
		@debug "V_list: " V_list

		p_list = utils.generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)
	end
	@debug "polynomials: " p_list

	∇L = System(p_list; parameters=parameters, variables=variables)	# variables are sorted lexicographically

	println("\ntotal number of polynomials: ", length(p_list))
	println("\ntotal number of variables: ", length(variables))
	println("\ntotal number of parameters: ", length(parameters))

	bkk = paths_to_track(∇L)
	sample_results["BKK"] = bkk




	## ~ STAGE 1 ~ ##

	println("\nSTAGE # 1 ...")

	run = 0
	@info "run # " run

	params0 = utils.generate_param_values(a, b, Nx, Ny, regularize,
		reg_parameterized, x_parameterized, y_parameterized,
		Λ_list, X, Y; complex=start_params_complex) #  start params should be complex

	@info "system parameters: " parameters(∇L)
	@info "system parameter_values: " params0

	println("\nParameter Homotopy: solving the initial system (polyhedral)...")
	retval = @timed solve(∇L; target_parameters=params0, threading=true)

	result0 = retval.value
	solve_time0 = retval.time
	solutions0 = solutions(result0)

	@debug "solve_time: " solve_time0
	@debug "result: " result0
	@info "system variables: " variables
	@info "system solutions: " solutions0




	## ~ STAGE 2 ~ ##


	try
		println("\nSTAGE # 2...")

		f = open(OUTPUT_FILE, "a")
		for run = 1:runcount
			@info "run # " run

			params1 = utils.generate_param_values(a, b, Nx, Ny, regularize,
				reg_parameterized, x_parameterized, y_parameterized,
				Λ_list, X, Y; complex=false) # subsequent params should be real
			@info "system parameter_values: " params1

			println("\nParameter Homotopy: solving target system...")
			retval = @timed solve(∇L, solutions0; start_parameters=params0,
				target_parameters=params1, threading=true)
			result1 = retval.value
			solve_time1 = retval.time
			solutions1 = solutions(result1)

			@debug "solve_time: " solve_time1
			@debug "result: " result1
			@info "solutions: " solutions1

			println("\ncollecting sample results...")
			global sample_results = utils.collect_results(L, ∇L, ∇²L, result1, params1,
				parsed_args, sample_results)
			@debug "sample results: " sample_results

			println("\nwriting sample results to file...")
			row = generate_row(string(run), "row", parsed_args, sample_results)
			write(f, row)
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



