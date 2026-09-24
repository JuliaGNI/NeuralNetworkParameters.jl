module ZygoteRulesExt

using NeuralNetworkParameters
using NeuralNetworkParameters: NetworkParameters, params, storage_gradient, map_cotangent
import ZygoteRules

# Differentiating a function of a `NetworkParameters` calls it with the `NetworkParameters` itself and
# gives a gradient of the same shape.
#
# Zygote's generic method differentiates straight through the wrapper — `getproperty`, `values` and
# `getindex` all reduce to `getfield(ps, :params)` — and hands back the structural tangent of the
# struct, a `NamedTuple` whose one field is the tangent of the wrapped `NamedTuple`. This method
# rewraps that tangent, so the gradient of a parameter set is a parameter set, and converts the
# cotangent of each structured leaf to the gradient with respect to its storage.
#
# `invoke` names Zygote's generic `pullback(f, args...)`, which this package does not own;
# `test/derivative_tests.jl` pins its signature. `f::Function` and not `f`: with an untyped `f`, the
# call `pullback(cx::Context, f)` would match this method as well as Zygote's.
function ZygoteRules.pullback(f::Function, ps::NetworkParameters)
    y, pb = invoke(ZygoteRules.pullback, Tuple{Any, Vararg{Any}}, f, ps)
    network_parameters_pullback(Δ) = (_rewrap(ps, pb(Δ)[1]),)
    y, network_parameters_pullback
end

# The structural tangent: the natural cotangent of each leaf, converted to its storage gradient once,
# after Zygote has added the contributions of every use of the leaf. A layer named `params` is no
# ambiguity here, because the wrapper's field is always the outer one.
function _rewrap(ps::NetworkParameters, p̄::NamedTuple{(:params,)})
    NetworkParameters(map_cotangent(storage_gradient, params(ps), p̄.params))
end

# A `NetworkParameters` comes from a rule that writes the gradient itself, `flatten`'s, whose leaves
# hold the gradient with respect to the storage already.
_rewrap(::NetworkParameters, p̄::NetworkParameters) = p̄

# A reverse pass that touched none of the parameters.
_rewrap(::NetworkParameters, ::Nothing) = nothing

# `p.L1` on a `NetworkParameters`. Zygote reads a field of a struct with `literal_getfield` only for a
# type that keeps Base's `getproperty`; for this one it differentiates the overload itself, with the
# name as a runtime `Symbol`, and the forward pass does not infer. The forward pass of a loss that
# reads its layers with `p.L1` is then slower and allocates more than with this adjoint, which makes it
# as fast as on the bare `NamedTuple`. The tangent is the structural one, keyed by every layer, with
# `nothing` for the layers not read.
ZygoteRules.@adjoint function ZygoteRules.literal_getproperty(
        p::NetworkParameters{T, Keys}, ::Val{s}) where {T, Keys, s}
    getproperty(p, s), Δ -> ((params = _one_hot(NamedTuple{Keys}, Val(s), Δ),), nothing)
end

@generated function _one_hot(::Type{NamedTuple{Keys}}, ::Val{s}, Δ) where {Keys, s}
    :(NamedTuple{Keys}(($((k === s ? :Δ : :nothing for k in Keys)...),)))
end

end
