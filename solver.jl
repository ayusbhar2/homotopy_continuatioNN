using HomotopyContinuation

# Data
X = [[7 -8 3 -5 10];	# each column is a single example
	 [-7 10 6 -2 6]]

Z = [[9 9 -8 1 10];		# each column is a single target
	 [10 3 -8 9 10]]

# variables
@var α1, α2, γ1, γ2

# regularization constant
λ = 0

# weight matrices
W1 = [α1 α2]		# first layer with one neuron
W2 = [γ1; γ2]		# output layer with two neurons

W = W2*W1

try

	sum_x = Matrix{Float64}(undef, 2, 5)
	sum_z = Matrix{Float64}(undef, 2, 5)

	for i = 1:5
		# println(X[:, i])
		sum_x = (X[:, i] * transpose(X[:, i]))
		sum_z = Z[:, i] * transpose(X[:, i])
	end

	# compute ∂L(W)/Wi

	∂_L_W1 = transpose(W2) * (W * sum_x - sum_z)	# gradient matrix w.r.t. layer 1
	∂_L_W2 = (W * sum_x - sum_z) * transpose(W1)	# gradient matrix w.r.t. layer 2

	f_α1 = ∂_L_W1[1,1]
	f_α2 = ∂_L_W1[1,2]
	f_γ1 = ∂_L_W2[1,1]
	f_γ2 = ∂_L_W2[2,1]

	∇L = System([f_α1, f_α2, f_γ1, f_γ2])
	result = solve(∇L)
	println(result)

catch e
	println("Inside catch. Something is not right: ", e)
end

