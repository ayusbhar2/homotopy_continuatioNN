using HomotopyContinuation
# import Utils

Random.seed!(1234)

# Data
X = [[7 -8 3 -5 10];	# each column is a single example
	 [-7 10 6 -2 6]]

Z = [[9 9 -8 1 10];		# each column is a single target
	 [10 3 -8 9 10]]


# variables
@var α1, α2, γ1, γ2


# weight matrices
W1 = [α1 α2]		# first layer with one neuron
W2 = [γ1; γ2]		# output layer with two neurons

W = W2*W1


# Tikhonov regularization constants
Λ₁ = rand(Float64, size(W1))	# entries picked unif from [0, 1).
Λ₂ = rand(Float64, size(W2))	# entries picked unif from [0, 1).


# compute regularized gratiend polynomials
∂_L_W1 = transpose(W2) * (W * X * transpose(X) - Z * transpose(X)) + Λ₁ .* W1	# gradient matrix w.r.t. layer 1
∂_L_W2 = (W * X * transpose(X) - Z * transpose(X)) * transpose(W1)	+ Λ₂ .* W2		# gradient matrix w.r.t. layer 2

f_α1 = ∂_L_W1[1,1]
f_α2 = ∂_L_W1[1,2]
f_γ1 = ∂_L_W2[1,1]
f_γ2 = ∂_L_W2[2,1]

∇L = System([f_α1, f_α2, f_γ1, f_γ2], [α1, α2, γ1, γ2])

result = solve(∇L)

println("N_R: ", get_N_R(result))
println("N_C: ", get_N_C(result))
println("N_DM: ", get_N_DM(H=1, n=nvariables(∇L)))



