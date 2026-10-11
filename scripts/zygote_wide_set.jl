# `Zygote.gradient` of a loss over a flat set of `n` leaves of 8 × 8, on one path per process: `nnp`
# (a `NetworkParameters`), `nt` (its bare `NamedTuple`) or `flat` (the flat vector through
# `unflatten`). It prints whether the gradient was taken, or the start of the error, and the time of
# the first call. `KNOWN_ISSUES.md`, K2.
#
#     julia --project=. -e 'using TestEnv; TestEnv.activate(); include("scripts/zygote_wide_set.jl")' nnp 369
using NeuralNetworkParameters, Zygote

const PATH, N = ARGS[1], parse(Int, ARGS[2])
ks = ntuple(i -> Symbol("W", i), N)
ps = NetworkParameters(NamedTuple{ks}(ntuple(i -> Float32.(reshape(1:64, 8, 8)) ./ i, N)))
loss(p) = sum(W -> sum(abs2, W), values(p))
v, layout = flatten(ps)

function first_gradient()
    PATH == "nnp" && return Zygote.gradient(loss, ps)
    PATH == "nt" && return Zygote.gradient(loss, NeuralNetworkParameters.params(ps))
    PATH == "flat" && return Zygote.gradient(w -> loss(unflatten(layout, w)), v)
    error("the path is nnp, nt or flat; got ", PATH)
end

t = @elapsed r = try
    first_gradient()
    "gradient taken"
catch e
    sprint(showerror, e)[1:min(end, 160)]
end
println(
    PATH, " ", N, ": ", r, ", first call ", round(t; digits = 1), " s; Julia ", VERSION,
    ", Zygote ", pkgversion(Zygote))
