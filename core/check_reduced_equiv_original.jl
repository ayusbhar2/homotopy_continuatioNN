using HomotopyContinuation

x1 = 1; y1 = 2; y2 = 3; λ11 = 4; λ21 = 5; μ11 = 6; μ12 = 7; μ21 = 8; μ22 = 9;

# original system

@var a11 a21 b11 b12 b21 b22

P = x1^2 * a11 * b11 + x1^2 * a21 * b12 - x1 * y1
Q = x1^2 * a11 * b21 + x1^2 * a21 * b22 - x1 * y2

g1 = b11 * P + b21 * Q + λ11 * a11
g2 = b12 * P + b22 * Q + λ21 * a21
g3 = a11 * P + μ11 * b11
g4 = a21 * P + μ12 * b12
g5 = a11 * Q + μ21 * b21
g6 = a21 * Q + μ22 * b22

G = System([g1, g2, g3, g4, g5, g6])

result1 = solve(G)



# reduced system:

A = a11 * x1
B = a21 * x1

E = λ11
F = (μ21 * μ22^2 * x1^2 * y2^2)
G = (μ11 * μ12^2 * x1^2 * y1^2)
D1 = (A^2 * μ12 + B^2 * μ11 + μ11 * μ12)
D2 = (A^2 * μ22 + B^2 * μ21 + μ21 * μ22)

f1 = E * D1^2 * D2^2 - F * D1^2 - G * D2^2

Ê = λ21
F̂ = (μ22 * μ21^2 * x1^2 * y2^2)
Ĝ = (μ12 * μ11^2 * x1^2 * y1^2)
# D3 = (A^2 * μ12 + B^2 * μ11 + μ11 * μ12)
# D4 = (A^2 * μ22 + B^2 * μ21 + μ21 * μ22)

f2 = Ê * D1^2 * D2^2 - F̂ * D1^2 - Ĝ * D2^2

F = System([f1, f2])

result2 = solve(F)


## Eliminate extraneous solutions from reduced system

# @var t u

# f3 = D1 * t - 1
# f4 = D2 * u - 1

# F = System([f1, f2, f3, f4])

# result2 = solve(F)


