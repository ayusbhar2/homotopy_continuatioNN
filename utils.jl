module Utils

using Distributions
using HomotopyContinuation
using LinearAlgebra: I
using OrderedCollections



function _varstring_to_matrix(s, m, n)
	t = eval(Meta.parse(s))
	v = collect(t)
	return reshape(v, (m, n))
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


function generate_parameter_matrix(m, n, name)
	# Generates a parameter matrix with dimensions same as W 
	s = string("@var ", name, "[1:", m, ",1:", n, "]")
	t = eval(Meta.parse(s))
	# println(t[1])
	return t[1]
end


function generate_parameterized_Tikhonov_matrices(W_list)

	Λ_list = Any[]
	for i =1:length(W_list)
		m = size(W_list[i])[1]; n = size(W_list[i])[2]
		Λᵢ = generate_parameter_matrix(m, n, "p$i")
		push!(Λ_list, Λᵢ)
	end
	return Λ_list
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


function generate_U_matrices(W_list)
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


# TODO: kwargs
function generate_gradient_polynomials(W_list, U_list, V_list, Λ_list, X, Y)
	p_list = Any[]
	W = reduce(*, reverse(W_list))	# weight matrices are multiplied in reverse order
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

	return p_list
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
function generate_weight_matrices(H, dx, dy, m, di)

	W_list = Vector{Matrix}(undef, H+1)
	for i = 1:(H+1)
		s = "@var "
		if i == 1
			# define di * dx new variables and fill them into Wᵢ
			for j = 1:di
				for k = 1:dx
					s = string(s,"w",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, di, dx)
		elseif i == H+1
			# define dy * di new variables and fill them into Wᵢ
			for j = 1:dy
				for k = 1:di
					s = string(s,"w",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, dy, di)
		else
			# define di * di new variables and fill them into Wᵢ
			for j = 1:di
				for k = 1:di
					s = string(s,"w",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, di, di)
		end
		W_list[i] = Wᵢ
		# println(string("W", i," :", size(Wᵢ)))
	end

	return W_list
end

function collect_results(sample_results, parsed_args, F::System, R::Result)

	n = nvariables(F)
	H = parsed_args["H"]

	sample_results["n"] = n
	sample_results["CBB"] = get_CBB(F)
	sample_results["N_C"] = get_N_C(R)
	sample_results["N_DM"] = convert(Int64, ceil(get_N_DM(H, n)))
	sample_results["N_R"] = get_N_R(R)

	return sample_results

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
		# @info "Λ₀_list: " Λ₀_list
	end

	if x_parameterized
		if complex
			X₀ = randn(ComplexF64, size(X))
		else
			X₀ = rand(Nx, size(X))
		end
		push!(param_values, collect(Iterators.flatten(X₀)))
		# @info "X₀: " X₀
	end

	if y_parameterized
		if complex
			Y₀ = randn(ComplexF64, size(Y))
		else
			Y₀ = rand(Ny, size(Y))
		end
		push!(param_values, collect(Iterators.flatten(Y₀)))
		# @info "Y₀: " Y₀
	end

	if length(param_values) > 0
		param_values = collect(Iterators.flatten(param_values))
	end

	return param_values
end


end # Utils