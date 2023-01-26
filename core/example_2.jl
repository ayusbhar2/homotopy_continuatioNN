# Example 2
using HomotopyContinuation

tol = 1.0e-10

function is_zero(a)
	if abs(real(a) - 0) < tol && abs(imag(a) - 0) < tol
		return true
	else
		return false
	end
end

c₁ = 1; c₂ = 5; c₃ = 3; c₄ = 7;

@var α₁ β₁ α₂ β₂

f_1 = 5*α₁*β₁^2 + 5*α₁*β₂^2 + 11*α₂*β₁^2 + 11*α₂*β₂^2 - 7*β₁ - 10*β₂ + 4*α₁ + c₁
f_2 = 11*α₁*β₁^2 + 11*α₁*β₂^2 + 25*α₂*β₁^2 + 25*α₂*β₂^2 - 15*β₁ -22*β₂ - 3*α₂ + c₂
f_3 = 5*α₁^2*β₁ + 22*α₁*α₂*β₁ + 25*α₂^2*β₁ - 7*α₁ - 15*α₂ - 2*β₁ + c₃
f_4 = 5*α₁^2*β₂ + 22*α₁*α₂*β₂ + 25*α₂^2*β₂ - 10*α₁ - 22*α₂ + 5*β₂ + c₄

F = System([f_1, f_2, f_3, f_4])

result = solve(F)
sols = solutions(result; only_real=false)
N_C = length(sols)
N_R = length(solutions(result; only_real=true))

N_sub = 0
for x in sols
	zero_count = 0
	for xᵢ in x
		# println("xᵢ: ", xᵢ)
		if is_zero(xᵢ)
			# println("x lies on a coordinate subspace: ", x)
			zero_count += 1
		end
	end
	if zero_count > 0
		global N_sub += 1
	end
	# println("\n")
end

println("# of paths tracked: ", length(path_results(result)))
println("\n N_C: ", N_C)
println("\n N_C_star :", N_C - N_sub)
println("\n N_R: ", N_R)