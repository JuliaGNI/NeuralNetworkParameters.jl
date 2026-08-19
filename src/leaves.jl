@doc raw"""
    freeparameters(x)

The differentiable storage of a leaf `x` — the numbers that a flat parameter vector should contain,
and the coordinates an optimizer should work in.

For an ordinary array this is the array itself. For a structured type it is whatever the type keeps
its degrees of freedom in: an ``n \times n`` symmetric matrix stores ``n(n+1)/2`` numbers, and it is
those that belong in the flat vector — not the ``n^2`` entries of the dense interface, which do not
even have the right length.

The return value may be

- the leaf itself, which marks it as *terminal* — the recursion stops and the numbers are copied
  straight out of it;
- another array or number;
- a `Tuple` or `NamedTuple` of either, for a type whose freedom lives in several blocks.

Together with [`rebuild`](@ref) this is the whole extension protocol: define the two for a type and
it can be flattened, walked and written to HDF5, with nothing in this package knowing about it.

# Extending

```julia
NeuralNetworkParameters.freeparameters(A::SymmetricMatrix) = A.S
NeuralNetworkParameters.rebuild(A::SymmetricMatrix, data)  = SymmetricMatrix(data, A.n)
```

`GeometricOptimizers` already exposes exactly this relation as `Base.parent` for its manifolds,
`VectorStorageMatrix`es and horizontal lifts, so one delegating method covers all of them:

```julia
NeuralNetworkParameters.freeparameters(x::Union{Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix}) =
    parent(x)
```

`Base.parent` is deliberately *not* used as the protocol here: `parent(::SubArray)` is the whole
underlying buffer rather than the view's own entries, so Base's relation and this one are not the
same.

# Examples

```jldoctest
using NeuralNetworkParameters: freeparameters

A = [1.0 2.0; 3.0 4.0]
freeparameters(A) === A

# output

true
```
"""
freeparameters(x::AbstractArray) = x
freeparameters(x::Number) = x

function freeparameters(x)
    throw(ArgumentError(_no_protocol_message(x, :freeparameters)))
end

@doc raw"""
    rebuild(prototype, data)

Rebuild a leaf shaped like `prototype` from the storage `data`. The inverse of
[`freeparameters`](@ref), and for an ordinary array simply `data`.

`prototype` is the leaf the storage came from, which is what carries the *non*-differentiable
information along: the `n` of a `SymmetricMatrix`, the `N` and `n` of a horizontal lift. Taking them
from a prototype rather than from the type is what lets `data` have a different element type from
`prototype` — `ForwardDiff.Dual`s, when a flattened parameter set is differentiated through — and it
is also why the concrete type comes back unchanged, where reconstructing from a type name can quietly
return a sibling type instead.

See [`freeparameters`](@ref) for how to extend this.
"""
rebuild(::AbstractArray, data) = data
rebuild(::Number, data) = data

function rebuild(prototype, data)
    throw(ArgumentError(_no_protocol_message(prototype, :rebuild)))
end

"""
    parameter_metadata(x)

The non-differentiable fields of a leaf, as a `NamedTuple`, for the benefit of storage formats that
have no prototype to rebuild against.

Empty by default. [`rebuild`](@ref) recovers this information from its prototype, which is enough for
flattening — but a parameter set read back from a file has no prototype, so a type whose storage does
not determine it (the `n` of a `SymmetricMatrix` does follow from `length(S)`; the `N` and `n` of a
`StiefelLieAlgHorMatrix` do not) has to write it out. See [`register_parameter_type!`](@ref).
"""
parameter_metadata(x) = NamedTuple()

function _no_protocol_message(x, f::Symbol)
    string("no method of `", f, "` for `", typeof(x), "`.\n",
        "A leaf of a parameter set has to say where its differentiable storage lives. Define\n",
        "    NeuralNetworkParameters.freeparameters(::", nameof(typeof(x)), ") = ...\n",
        "    NeuralNetworkParameters.rebuild(::", nameof(typeof(x)), ", data) = ...\n",
        "or, if the type already exposes its storage as `Base.parent`, delegate to it.")
end

"""
    isparametertree(x)

Whether `x` is a *branch* of a parameter set — a [`NetworkParameters`](@ref), `NamedTuple` or
`Tuple` that the walks recurse into — as opposed to a leaf.
"""
isparametertree(::NetworkParameters) = true
isparametertree(::NamedTuple) = true
isparametertree(::Tuple) = true
isparametertree(::Any) = false

"""
    isterminal(x)

Whether the numbers of the leaf `x` can be copied straight out of it, i.e. whether
[`freeparameters`](@ref) returns `x` itself.
"""
isterminal(x) = freeparameters(x) === x

"""
    parameter_eltype(ps)

The element type to flatten `ps` into: `promote_type` over the element types of its leaves.

`Float32` parameters therefore flatten to a `Vector{Float32}`. Defaulting to `Float64` instead — as
`ParameterHandling.flatten` does — silently doubles the width of every single-precision network that
passes through, and every quantity computed from the flat vector downstream with it.

# Examples

```jldoctest
using NeuralNetworkParameters: parameter_eltype

parameter_eltype((a = Float32[1, 2], b = Float32[3;;]))

# output

Float32
```

A mixed parameter set promotes:

```jldoctest
using NeuralNetworkParameters: parameter_eltype

parameter_eltype((a = Float32[1, 2], b = [3.0]))

# output

Float64
```
"""
parameter_eltype(ps::NetworkParameters) = parameter_eltype(params(ps))
parameter_eltype(ps::Union{NamedTuple, Tuple}) = _promote_eltypes(values(ps))
parameter_eltype(x::Number) = typeof(x)

function parameter_eltype(x)
    s = freeparameters(x)
    s === x ? eltype(x) : parameter_eltype(s)
end

_promote_eltypes(::Tuple{}) = Union{}
function _promote_eltypes(xs::Tuple)
    promote_type(parameter_eltype(first(xs)), _promote_eltypes(Base.tail(xs)))
end
