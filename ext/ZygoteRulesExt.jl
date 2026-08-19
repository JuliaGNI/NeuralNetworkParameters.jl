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
# This method belongs here rather than in `AbstractNeuralNetworks`, where it used to live: with the
# parameter type defined in this package, `ZygoteRules.pullback` is the only foreign name in the
# signature, and a method needs to own just one of them.
function ZygoteRules.pullback(f::Function, ps::NetworkParameters)
    y, pb = ZygoteRules.pullback(f, NamedTuple{keys(ps)}(values(ps)))

    function network_parameters_pullback(output)
        p̄ = pb(output)[1]
        (NetworkParameters{keys(ps)}(_values(p̄)),)
    end

    y, network_parameters_pullback
end

_values(nt::NamedTuple) = values(nt)
_values(nt::NamedTuple{(:params,), Tuple{AT}}) where {AT <: NamedTuple} = _values(nt.params)

end
