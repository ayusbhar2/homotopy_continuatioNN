include("utils.jl")
import .Utils as utils

using HomotopyContinuation


println("> utils.generate_conv_layer")
@var t1, t2
W_expected = Matrix([t1 t2 0 0; 0 0 t1 t2])
W_actual = utils.generate_conv_layer(2, 4; stride=2, width=2)

if !(W_expected == W_actual)
	error("failed")
end