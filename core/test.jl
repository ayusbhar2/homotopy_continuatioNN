include("utils.jl")
import .Utils as utils

using HomotopyContinuation


function assert(condition; msg="")
	if !condition
		error("Failed: $msg")
	end
end

# test
println("> utils.generate_conv_layer")
@var t1, t2
W_expected = Matrix([t1 t2 0 0; 0 0 t1 t2])
W_actual = utils.generate_conv_layer(2, 4; stride=2, width=2)

if !(W_expected == W_actual)
	error("""Failed:
		Expecting: $W_expected
		REceived: $W_actual""")
end


# test
println("> utils.generate_weight_matrices_no_conv")
W_list_actual = utils.generate_weight_matrices(1, 2, 2, 1, 1)
@var w111 w121 w211 w212

assert(W_list_actual[1][1,1]==w111)
assert(W_list_actual[1][1,2]==w121)
assert(W_list_actual[2][1,1]==w211)
assert(W_list_actual[2][2,1]==w212)



# test
println("> utils.generate_weight_matrices_yes_conv")
W_list_actual = utils.generate_weight_matrices(1, 2, 2, 1, 2; 
	convolution=true, stride=1, width=1)
@var t1

assert(W_list_actual[1][1,1]==t1)
assert(W_list_actual[1][1,2]==0)
assert(W_list_actual[1][2,1]==0)
assert(W_list_actual[1][2,2]==t1)

@var w211 w212 w221 w222
assert(W_list_actual[2][1,1]==w211)
assert(W_list_actual[2][1,2]==w212)
assert(W_list_actual[2][2,1]==w221)
assert(W_list_actual[2][2,2]==w222)



# test
println("> utils.generate_loss_func_with_variables")

@var w111 w112 w211 w221 l111 l112 l211 l221 x11 x12 x21 x22 y11 y12 y21 y22

W1 = [w111 w112]
W2 = [w211; w221]
W_list = [W1, W2]

Λ1 = [l111 l112]
Λ2 = [l211; l221]
Λ_list = [Λ1, Λ2]

X = [x11 x12; x21 x22]
Y = [y11 y12; y21 y22]

s = eval(Meta.parse("""
	0.5*(l111^2*w111^2 + l112^2*w112^2 + l211^2*w211^2 + l221^2*w221^2 + 
		(-y11 + x11*w111*w211 + x21*w112*w211)^2 + 
		(-y12 + x12*w111*w211 + x22*w112*w211)^2 + 
		(-y21 + x11*w111*w221 + x21*w112*w221)^2 + 
		(-y22 + x12*w111*w221 + x22*w112*w221)^2)
	"""))

assert(utils.generate_loss_func(W_list, Λ_list, X, Y) == s)



# test
println("> utils.generate_loss_func_with_constants")

W1 = [1 2] 
W2 = [3; 4]
W_list = [W1, W2]

Λ1 = [5 6]
Λ2 = [7; 8]
Λ_list = [Λ1, Λ2]

X = [-1 2; 3 4]
Y = [5 6; 7 8]
assert(utils.generate_loss_func(W_list, Λ_list, X, Y)==1751.5)



# test
println("> utils.generate_gradient_polynomials_with_convolution")
@var w111 w112 w211 w221

vars = [w111 w112 w211 w221]
L = w111^2 + w112^2 + w211^2 + w221^2

p_list = utils.generate_gradient_polynomials_with_convolution(L, vars)
assert(p_list==[2*w111, 2*w112, 2*w211, 2*w221])



# test
println("> utils.extract_and_sort_variables")
@var t1 w1
vars = [t1, 0, 0, t1]
assert(utils.extract_and_sort_variables(vars)==[t1])

vars = [[w1 0; 0 w1], [t1 0; 0 t1]]
assert(utils.extract_and_sort_variables(vars)==[t1, w1])



#test
println("> utils.eval_poly_scalar")
@var x y
L = 0.5*(x + y)
result = utils.eval_poly(L, [x, y] => [1, 1])
assert(result==1.0)
assert(typeof(result)==Float64)


# test
println("> utils.eval_poly_matrix")
@var x1 x2 x3 x4
names = [x1, x2, x3, x4]
values = [-1.0, -0.1, 0.2, -0.7]
J = [1.0*x1^2*x2^2 + 1.0*x3^2, 1.0*x1^2*x4*x2]
result = utils.eval_poly(J, names => values)
# println("result: ", result)
assert(typeof(result[1])==Float64)
assert(typeof(result[2])==Float64)






