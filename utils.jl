module Utils

using Distributions
using HomotopyContinuation
using LinearAlgebra: I


# function _parse_vars(s)


function _varstring_to_matrix(s, m, n)
	t = eval(Meta.parse(s))
	v = collect(t)
	return reshape(v, (m, n))
end


function get_CBB(F::System)
	prod(degrees(F.expressions))
end


# function get_BKK(F::System)
# 	#####
# end


function get_N_R(R::Result)
	length(solutions(R; only_real=true))
end


function get_N_C(R::Result)
	length(solutions(R)) - get_N_R(R)
end


function get_N_DM(H, n)
	sqrt(2) * (H+1)^((n+1)/2)
end


function generate_Tikhonov_matrix(D, W)
	return rand(D, size(W))
end


function generate_Tikhonov_matrices(D, W_list)
	Λ_list = Any[]
	for i =1:length(W_list)
		Λᵢ = generate_Tikhonov_matrix(D, W_list[i])
		println("Λ", i, " :", size(Λᵢ))
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
			println("U", i, ": I")
		else
			U = reduce(*, reverse(W_list[i+1:len]))	# W_list contains matrices in reverse order
			println("U", i, " :", size(U))
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
			println("V", i, ": I")
		else
			V = reduce(*, reverse(W_list[1:i-1]))	# W_list contains matrices in reverse order
			println("V", i, " :", size(V))
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
di (Int): (Fixed) number of neurons in all hidden layers

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
					s = string(s,"x",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, di, dx)
		elseif i == H+1
			# define dy * di new variables and fill them into Wᵢ
			for j = 1:dy
				for k = 1:di
					s = string(s,"x",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, dy, di)
		else
			# define di * di new variables and fill them into Wᵢ
			for j = 1:di
				for k = 1:di
					s = string(s,"x",i,j,k, " ")
				end
			end
			Wᵢ = _varstring_to_matrix(s, di, di)
		end
		W_list[i] = Wᵢ
		println(string("W", i," :", size(Wᵢ)))
	end

	return W_list
end

end # Utils