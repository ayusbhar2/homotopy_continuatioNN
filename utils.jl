module Utils

"""
Input: System of polynomials.
Output: CBB bound of the system.
"""
function get_CBB(F::System)
	prod(degrees(F.expressions))
end


function get_BKK(F::System)
	#####
end

function get_N_R(R::Result)
	length(solutions(R; only_real=true))
end

function get_N_C(R::Result)
	length(solutions(R)) - get_N_R(R)
end

function get_N_DM(H, n)
	sqrt(2) * (H+1)^((n+1)/2)
end


# function generate_weight_matrices(H, dx, dy)
# 	# NOTE:  All hidden layers have 2 neurons
# 	# i.e. dᵢ = 2 for all i.
# 	for i 
		
# 	end

# end

end # Utils