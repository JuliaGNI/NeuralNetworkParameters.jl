module ZygoteRulesExt

using NeuralNetworkParameters
using NeuralNetworkParameters: NetworkParameters, params
import ZygoteRules

# Differentiating a function of a `NetworkParameters` gives a gradient of the same shape.
#
# Without this, the reverse pass sees an ordinary struct and hands back a tangent for it — a
# `NamedTuple` whose one field is the wrapped `NamedTuple` — so the gradient of a parameter set is not
# a parameter set and cannot be fed back to anything expecting one. Differentiating with respect to
# the `NamedTuple` and rewrapping the result afterwards keeps the two shapes in step.
#
# This method belongs here rather than in `AbstractNeuralNetworks`: with the parameter type defined
# in this package, `ZygoteRules.pullback` is the only foreign name in the signature, and a method
# needs to own just one of them.
function ZygoteRules.pullback(f::Function, ps::NetworkParameters)
    y, pb = ZygoteRules.pullback(f, NamedTuple{keys(ps)}(values(ps)))

    function network_parameters_pullback(output)
        (_rewrap(ps, pb(output)[1]),)
    end

    y, network_parameters_pullback
end

# The reverse pass ran against `NamedTuple{keys(ps)}`, so what it hands back is keyed by the set's own
# layers already and goes into the wrapper as it stands.
#
# The set's keys are asked about *first*, and that ordering is the point rather than an accident. A set
# whose one layer is called `params` is an ordinary set — `getproperty` reaches into the wrapped
# `NamedTuple`, so the field itself is read with `params(ps)` and the name is free for a layer — and
# its tangent is indistinguishable from the structural one below. Told apart by shape alone it was
# unwrapped a level too far, and the gradient came back with that layer's own shape lost, silently and
# without an error.
#
# Both questions are answered at compile time: the keys of a `NamedTuple` and of a `NetworkParameters`
# are type parameters, so neither the comparison nor the `isa` survives into the reverse pass.
function _rewrap(ps::NetworkParameters, p̄::NamedTuple)
    keys(p̄) === keys(ps) && return NetworkParameters(p̄)
    # A tangent built for the *struct* instead, whose one field is the wrapped `NamedTuple`: unwrap it
    # and ask the same question of what is inside.
    p̄ isa NamedTuple{(:params,)} && return _rewrap(ps, p̄.params)
    _rewrap_error(ps, p̄)
end

# A reverse pass that touched none of the parameters hands back a structural zero for the whole set,
# and there is nothing to rewrap: `keys(ps)` has as many entries as the set has layers and one
# `nothing` cannot stand for each of them. So the hole is passed straight out, which is what Zygote
# gives for the bare `NamedTuple` — a loss reading only the layout, or a frozen sub-network, would
# otherwise raise where the unwrapped parameters return `nothing`.
_rewrap(::NetworkParameters, ::Nothing) = nothing

_rewrap(ps::NetworkParameters, p̄) = _rewrap_error(ps, p̄)

# Naming both shapes, where spreading the tangent over `keys(ps)` would raise about a length — or,
# for a set of one layer, not raise at all.
@noinline function _rewrap_error(ps, p̄)
    throw(ArgumentError(string("the reverse pass returned a `", typeof(p̄),
        "` for parameters keyed ", keys(ps),
        "; expected a `NamedTuple` over those keys, or `nothing`")))
end

end
