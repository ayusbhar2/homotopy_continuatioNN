module Utils

using Distributions
using HomotopyContinuation
using LinearAlgebra: I, eigvals, issymmetric
using OrderedCollections


function real_and_nonpositive(e)
	r = real(e)
	if r == e
		if r <= 0
			retval = true
		else
			retval = false
		end
	else
		retval = false
	end
	return retval
end


function to_number_exp(p)
	return to_number(expand(p))
end


function eval_poly(f, args)
  f_args = evaluate(f, args)
  if f_args isa Expression
     return to_number(expand(f_args))
  elseif (f_args isa Vector{Expression} || f_args isa Matrix{Expression})
  	 return map(to_number_exp, f_args)  # apply to_number_exp() element-wise
  else
     return f_args
  end
end


function make_variable(exp)
	return Variable(exp)
end


function extract_and_sort_variables(W_list)
	vars = unique(collect(Iterators.flatten(W_list)))
	vars = deleteat!(vars, vars .== 0)
	vars = map(make_variable, vars)
	vars = sort(vars)
	# println(typeof(vars))
	return vars
end


function _reshape_rowmajor(v, m, n)
	M = Matrix{Variable}(undef, m, n)
	i = 0
	while i < m
		row = v[i*n+1:(i+1)*n]
		M[i+1, :] = row
		i += 1
	end
	return M
end


function _varstring_to_matrix(s, m, n)
	t = eval(Meta.parse(s))
	v = collect(t)
	return _reshape_rowmajor(v, m, n)

end


function get_CBB(F::System)
	prod(degrees(F))
end


function get_BKK(F::System)

end


function get_N_R(R::Result)
	length(solutions(R; only_real=true))
end


function get_N_C(R::Result)
	length(solutions(R; only_real=false))
end


function get_N_DM(H, n)
	sqrt(2) * (2*H+1)^((n+1)/2)
end


function get_norm_squared(v)
	if length(size(v)) > 1  # matrix
		sum = 0
		for i=1:size(v)[1]
			for j = 1:size(v)[2]
				sum += v[i,j]^2
			end
		end
	else  					# vector
		sum = 0
		for i = 1:length(v)
			sum += (v[i])^2
		end
	end
	return sum
end


function generate_parameter_matrix(m, n, name)
	s = string("@var ", name, "[1:", m, ",1:", n, "]")
	t = eval(Meta.parse(s))
	# println(t[1])
	return t[1]
end


function generate_zero_matrices(W_list)
	Λ_list = Any[]
	for i =1:length(W_list)
		Λᵢ = zeros(size(W_list[i]))
		# println("Λ", i, " :", size(Λᵢ))
		push!(Λ_list, Λᵢ)
	end
	return Λ_list
end


function generate_real_Tikhonov_matrices(a, b, W_list)
	Λ_list = Any[]
	U = Uniform(a, b)
	for i =1:length(W_list)
		Λᵢ = rand(U, size(W_list[i]))
		# println("Λ", i, " :", size(Λᵢ))
		push!(Λ_list, Λᵢ)
	end
	return Λ_list
end


function generate_complex_Tikhonov_matrices(W_list)
	Λ_list = Any[]
	for i =1:length(W_list)
		Λᵢ = rand(ComplexF64, size(W_list[i]))
		# println("Λ", i, " :", size(Λᵢ))
		push!(Λ_list, Λᵢ)
	end
	return Λ_list
end


function generate_parameterized_Tikhonov_matrices(W_list)

	Λ_list = Any[]
	for i =1:length(W_list)
		m = size(W_list[i])[1]; n = size(W_list[i])[2]
		Λᵢ = generate_parameter_matrix(m, n, "λ$i")
		# println("Λ", i, " :", size(Λᵢ))
		push!(Λ_list, Λᵢ)
	end
	return Λ_list
end


function generate_U_matrices(W_list)
	# NOTE: does not transpose the U matrices
	U_list = Any[]
	len = length(W_list)

	for i = 1:len
		if i == len
			U = I
			# println("U", i, ": I")
		else
			U = reduce(*, reverse(W_list[i+1:len]))	# W_list contains matrices in reverse order
			# println("U", i, " :", size(U))
		end
		push!(U_list, U)
	end

	return U_list
end


function generate_V_matrices(W_list)
	V_list = Any[]
	len = length(W_list)

	for i = 1:len
		if i == 1
			V = I
			# println("V", i, ": I")
		else
			V = reduce(*, reverse(W_list[1:i-1]))	# W_list contains matrices in reverse order
			# println("V", i, " :", size(V))
		end
		push!(V_list, V)
	end

	return V_list
end


function generate_conv_layer(di, dx; stride=1, width=1)
	if (di - 1) * stride + width > dx
		error("Mismatch! Combination of di, dx, stride and width leads to an invalide convolution.")
	else
		W = []
		s = "@var "
		for i = 1:width
			s = string(s, "t", i, " ")
		end
		t = eval(Meta.parse(s))		# declare variables
		c = collect(t)

		kernel = join(c, " ") * " "
		exp = "["
		for j = 0:(di - 1)
			l = repeat("0 ", j*stride)
			r = repeat("0 ", dx - (j*stride + width))
			exp = exp * string(l, kernel, r, ";")
		end
		exp = chop(exp) * "]"
		W = eval(Meta.parse(exp))

	end
	# println("convolution layer Matrix: ", W)
	return W
end


""" Generate a list of weight matrices (one per layer) given the architectural
parameters of the neural network.

Input:

H (Int)	: Number of hidden layers in the network
dx (Int): Length of the input vector
dy (Int): Length of the output vector
m (Int)	: Number of examples
di (Int): (Fixed)	 number of neurons in all hidden layers

Output:

W_list: (Vector{Matrix{Variable}}): List of weight matrices for the network
"""
function generate_weight_matrices(H, dx, dy, m, di; 
	convolution=false, stride=1, width=1)

	W_list = Vector{Matrix}(undef, H+1)
	for i = 1:(H+1)
		# println("## i = ", i)
		s = "@var "
		if i == 1
			if convolution
				Wᵢ = generate_conv_layer(di, dx; stride=stride, width=width)
			else
				# define di * dx new variables and fill them into Wᵢ
				for j = 1:di
					for k = 1:dx
						s = string(s,"w",i,j,k, " ")
					end
				end
				Wᵢ = _varstring_to_matrix(s, di, dx)
				# println("W", i, " :\n", Wᵢ)
			end
		elseif i == H+1
			# define dy * di new variables and fill them into Wᵢ
			for j = 1:dy
				for k = 1:di
					s = string(s,"w",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, dy, di)
			# println("W", i, " :\n", Wᵢ)
		else
			# define di * di new variables and fill them into Wᵢ
			for j = 1:di
				for k = 1:di
					s = string(s,"w",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, di, di)
			# println("W", i, " :\n", Wᵢ)
		end
		W_list[i] = Wᵢ
		# println(string("W", i," :", size(Wᵢ)))
	end

	return W_list
end


function generate_loss_func(W_list, Λ_list, X, Y)
	W = reduce(*, reverse(W_list))
	sum1 = 0
	for i = 1:size(X)[2]
		x = X[:, i]
		y = Y[:, i]
		v = W * x - y
		sum1 = sum1 + get_norm_squared(v)
	end

	sum2 = 0
	for i = 1:length(W_list)
		M = W_list[i] .* Λ_list[i]
		sum2 += get_norm_squared(M)
	end

	sum = 0.5*(sum1 + sum2)
	# println("L: ", sum)

	return sum

end



# TODO: kwargs
function generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)
	p_list = Any[]
	W = reduce(*, reverse(W_list))	# weight matrices are multiplied in reverse order
	for i = 1:length(W_list)
		∂L_Wᵢ = transpose(U_list[i]) * (W * X * transpose(X) - Y * transpose(X)) * transpose(V_list[i]) + Λ_list[i] .* W_list[i]
		# println("∂L_Wᵢ :", ∂L_Wᵢ)

		v = vec(transpose(∂L_Wᵢ))	# Julia arrays are column major
		for p in v
			# println("p: ", p)
			push!(p_list, p)
		end
	end

	return p_list
end


function generate_param_values(a, b, Nx, Ny, regularize, reg_parameterized, x_parameterized,
							   y_parameterized,
							   Λ_list, X, Y; complex=false)
	param_values = []

	if regularize && reg_parameterized
		if complex
			Λ₀_list = generate_complex_Tikhonov_matrices(Λ_list)
		else
			Λ₀_list = generate_real_Tikhonov_matrices(a, b, Λ_list)
		end
		push!(param_values, collect(Iterators.flatten(Λ₀_list)))
	end

	if x_parameterized
		if complex
			X₀ = randn(ComplexF64, size(X))
		else
			X₀ = rand(Nx, size(X))
		end
		push!(param_values, collect(Iterators.flatten(X₀)))
	end

	if y_parameterized
		if complex
			Y₀ = randn(ComplexF64, size(Y))
		else
			Y₀ = rand(Ny, size(Y))
		end
		push!(param_values, collect(Iterators.flatten(Y₀)))
	end

	if length(param_values) > 0
		param_values = collect(Iterators.flatten(param_values))
	end

	return param_values
end


function collect_results(L, F::System, ∇²L, R::Result, param_values, parsed_args, sample_results)

	n = nvariables(F)
	H = parsed_args["H"]
	names = cat(variables(F), parameters(F); dims=1)

	sample_results["variables"] = variables(F)
	sample_results["n"] = n
	sample_results["CBB"] = get_CBB(F)
	sample_results["N_C"] = get_N_C(R)
	sample_results["N_DM"] = convert(Int64, ceil(get_N_DM(H, n)))
	sample_results["N_R"] = get_N_R(R)

	loss_values = []
	idx_values = []
	r_sols = []
	all_solutions = solutions(R)
	real_solutions = solutions(R; only_real=true)

	for real_sol in real_solutions
		r_sol = map(real, real_sol)	# discard the imaginary part
		push!(r_sols, r_sol)

		values = cat(r_sol, param_values; dims=1)
		loss = eval_poly(L, names => values)
		push!(loss_values, loss)

		hess = eval_poly(∇²L, names => values)
		evals = eigvals(hess)
		idx = sum(real_and_nonpositive.(evals))
		push!(idx_values, idx)
	end

	sample_results["All_sols"] = all_solutions
	sample_results["Real_sols"] = r_sols
	sample_results["L_values"] = loss_values
	sample_results["Idx_vals"] = idx_values


	return sample_results

end



end # Utils