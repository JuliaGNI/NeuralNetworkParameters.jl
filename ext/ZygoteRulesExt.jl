module ZygoteRulesExt

using NeuralNetworkParameters
using NeuralNetworkParameters: NetworkParameters, storage_gradient, map_cotangent
import ZygoteRules

# Differentiating a function of a `NetworkParameters` calls it with the `NetworkParameters` itself and
# gives a gradient of the same shape.
#
# Zygote's generic method differentiates straight through the wrapper — `getproperty`, `values` and
# `getindex` all reduce to `getfield(ps, :params)` — and hands back the structural tangent of the
# struct, a `NamedTuple` whose one field is the tangent of the wrapped `NamedTuple`. This method
# rewraps that tangent with `map_cotangent`, so the gradient of a parameter set is a parameter set, and
# converts the cotangent of each structured leaf to the gradient with respect to its storage once,
# after Zygote has added the contributions of every use of the leaf. The walk reads the other shapes
# a set's cotangent comes in by the same rule: a `NetworkParameters`, which the `flatten` rule writes
# with the storage gradient at every leaf already, comes back as it is, and a reverse pass that
# touched none of the parameters, `nothing` or a structural tangent holding a zero, gives `nothing`.
# A layer named `params` is no ambiguity, because the wrapper's field is always the outer one.
#
# `invoke` names Zygote's generic `pullback(f, args...)`, which this package does not own;
# `test/derivative_tests.jl` pins its signature. `f::Function` and not `f`: with an untyped `f`, the
# call `pullback(cx::Context, f)` would match this method as well as Zygote's.
function ZygoteRules.pullback(f::Function, ps::NetworkParameters)
    y, pb = invoke(ZygoteRules.pullback, Tuple{Any, Vararg{Any}}, f, ps)
    network_parameters_pullback(Δ) = (map_cotangent(storage_gradient, ps, _set_cotangent(pb(Δ))),)
    y, network_parameters_pullback
end

# Zygote's pullback gives `nothing` for the whole tuple of arguments when a rule gave the set a zero.
_set_cotangent(::Nothing) = nothing
_set_cotangent(Δs::Tuple) = first(Δs)

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
