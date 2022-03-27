include("utils.jl")
import .Utils as utils

using ArgParse
using Distributions
using HomotopyContinuation
using LinearAlgebra: I
using Random

# configs
a = 0; b= 1;

# module level constants
Unif = Uniform(a, b)	# used for constructing the Tikhonov matrices
X = [[7 -8 3 -5 10];	# each column is a single example
	 [-7 10 6 -2 6]]

Y = [[9 9 -8 1 10];		# each column is a single target
	 [10 3 -8 9 10]]

# expected output:
#	No. H 	dx 	dy 	m 	a 	b 	CBB  N_DM  N_R  N_C
#	1	1	2	2	5	0	1	81	 8	   5	12