include("utils.jl")
import .Utils as utils

using Distributions
using HomotopyContinuation
using Random

Random.seed!(1234)

# Constants
X = [[7 -8 3 -5 10];	# each column is a single example
	 [-7 10 6 -2 6]]

Z = [[9 9 -8 1 10];		# each column is a single target
	 [10 3 -8 9 10]]

U = Uniform(0,1)	# used for constructing the Tikhonov matrices


# variables
@var α1, α2, γ1, γ2


# weight matrices
W1 = [α1 α2]		# first layer with one neuron
W2 = [γ1; γ2]		# output layer with two neurons

W = W2*W1


# Tikhonov regularization constants
Λ₁ = utils.generate_Tikhonov_matrix(U, size(W1))
Λ₂ = utils.generate_Tikhonov_matrix(U, size(W2))


# Compute regularized gratiend polynomials
∂_L_W1 = transpose(W2) * (W * X * transpose(X) - Z * transpose(X)) + Λ₁ .* W1	# gradient matrix w.r.t. layer 1
∂_L_W2 = (W * X * transpose(X) - Z * transpose(X)) * transpose(W1)	+ Λ₂ .* W2		# gradient matrix w.r.t. layer 2

f_α1 = ∂_L_W1[1,1]
f_α2 = ∂_L_W1[1,2]
f_γ1 = ∂_L_W2[1,1]
f_γ2 = ∂_L_W2[2,1]

∇L = System([f_α1, f_α2, f_γ1, f_γ2], [α1, α2, γ1, γ2])

result = solve(∇L)

println("CBB: ", utils.get_CBB(∇L))
println("N_DM: ", convert(Int64, trunc(utils.get_N_DM(1, nvariables(∇L)))))
println("N_R: ", utils.get_N_R(result))
println("N_C: ", utils.get_N_C(result))

