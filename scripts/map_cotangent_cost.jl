# The first call of `map_cotangent(storage_gradient, ps, Δ)`, the reverse pass of every gradient,
# on a flat set of 369 leaves of two numbers, with a cotangent that touches 100 of them. The first
# call runs inside `Base.invokelatest`, so its compile is inside the timer. Run it once per process:
#
#     julia --startup-file=no --project=. scripts/map_cotangent_cost.jl
using NeuralNetworkParameters, LinearAlgebra
using NeuralNetworkParameters: map_cotangent
BLAS.set_num_threads(1)
const K = 369
ps = NetworkParameters(NamedTuple{ntuple(i -> Symbol(:p, i), K)}(ntuple(
    i -> [Float64(i), 1.0], K)))
Δ = (params = NamedTuple{ntuple(i -> Symbol(:p, i), 100)}(ntuple(i -> [1.0, 1.0], 100)),)
t = @elapsed Base.invokelatest(map_cotangent, storage_gradient, ps, Δ)
g = map_cotangent(storage_gradient, ps, Δ)
a = @allocated map_cotangent(storage_gradient, ps, Δ)
println("first call ", round(t; digits = 2), " s; second call ", a, " bytes; sum ",
    sum(first(flatten(g))), "; Julia ", VERSION)
