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
	first_layer_conv=true, stride=1, width=1)
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



# # test
# println("> utils.generate_weight_matrices_no_conv_di_2")
# W_list_actual = utils.generate_weight_matrices(1, 2, 2, 1, 2)
# println(W_list_actual)


