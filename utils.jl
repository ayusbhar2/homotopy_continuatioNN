module Utils

using Distributions
using HomotopyContinuation

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

end # Utils