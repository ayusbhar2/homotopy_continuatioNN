module Utils

using Distributions
using HomotopyContinuation


function _str_to_matrix(s, m, n)
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

function generate_Tikhonov_matrix(D, dims)
	rand(D, dims)
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
		if i == 1
			# define di * dx new variables and fill them into Wᵢ
			s = "@var "
			for j = 1:di
				for k = 1:dx
					s = string(s,"x",i,j,k, " ")
				end
			end
			Wᵢ = _str_to_matrix(s, di, dx)
		elseif i == H+1
			# define dy * di new variables and fill them into Wᵢ
			s = "@var "
			for j = 1:dy
				for k = 1:di
					s = string(s,"x",i,j,k, " ")
				end
			end
			Wᵢ = _str_to_matrix(s, dy, di)
		else
			# define di * di new variables and fill them into Wᵢ
			s = "@var "
			for j = 1:di
				for k = 1:di
					s = string(s,"x",i,j,k, " ")
				end
			end
			Wᵢ = _str_to_matrix(s, di, di)
		end
		W_list[i] = Wᵢ
		println(string("i = ", i, " Wᵢ is ", size(Wᵢ)))
	end

	return W_list
end

end # Utils