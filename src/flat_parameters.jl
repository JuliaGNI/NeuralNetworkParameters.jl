"""
    FlatParameters(data, layout)
    FlatParameters(ps)

A parameter set in its flat form: an `AbstractVector` of every number in the set, carrying the
[`ParameterLayout`](@ref) that puts it back together.

This is the representation to differentiate with respect to. It behaves as an ordinary vector, so
`ForwardDiff`, a linear solver or an optimizer can work on it directly, while still knowing how to
return an answer in the shape of the network. `parent` hands out the bare `Vector` for anything that
would rather not see a wrapper.

`similar` *keeps the layout*, so scratch space derived from a flat parameter set — a gradient, a
momentum buffer — stays self-describing.

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
fp = FlatParameters(ps)
(length(fp), fp[2])

# output

(3, 2.0)
```

Entries can be read back by layer, and the whole set converted:

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
fp = FlatParameters(ps)
(fp.L1.b, NetworkParameters(fp) == ps)

# output

([3.0], true)
```
"""
struct FlatParameters{T, DT <: AbstractVector{T}, LT <: ParameterLayout} <:
       AbstractVector{T}
    data::DT
    layout::LT

    function FlatParameters(data::DT, layout::LT) where {
            T, DT <: AbstractVector{T}, LT <: ParameterLayout}
        length(data) == length(layout) ||
            throw(DimensionMismatch(string("flat vector has length ", length(data),
                ", layout needs ", length(layout))))
        new{T, DT, LT}(data, layout)
    end
end

FlatParameters(ps) = FlatParameters(flatten(ps)...)
FlatParameters(::Type{T}, ps) where {T} = FlatParameters(flatten(T, ps)...)

"""
    flatlayout(fp)

The [`ParameterLayout`](@ref) carried by `fp`.
"""
flatlayout(fp::FlatParameters) = getfield(fp, :layout)

Base.parent(fp::FlatParameters) = getfield(fp, :data)

Base.size(fp::FlatParameters) = size(parent(fp))
Base.IndexStyle(::Type{<:FlatParameters}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(fp::FlatParameters, i::Int) = parent(fp)[i]
Base.@propagate_inbounds Base.setindex!(fp::FlatParameters, x, i::Int) = (parent(fp)[i] = x)

Base.similar(fp::FlatParameters) = FlatParameters(similar(parent(fp)), flatlayout(fp))
function Base.similar(fp::FlatParameters, ::Type{S}) where {S}
    FlatParameters(similar(parent(fp), S), flatlayout(fp))
end

# Broadcasting and every other generic path reaches `similar(x, T, dims)`. The layout only survives a
# same-length result; anything else is an ordinary array, since a layout that does not match its data
# is worse than none.
function Base.similar(fp::FlatParameters, ::Type{S}, dims::Dims) where {S}
    dims == size(parent(fp)) ? FlatParameters(similar(parent(fp), S, dims), flatlayout(fp)) :
    similar(parent(fp), S, dims)
end

Base.copy(fp::FlatParameters) = FlatParameters(copy(parent(fp)), flatlayout(fp))

# Reading a layer off the flat form. `getfield` wins, so `fp.data` and `fp.layout` keep working and
# nothing in Base is surprised by the overload.
@inline function Base.getproperty(fp::FlatParameters, s::Symbol)
    s === :data && return getfield(fp, :data)
    s === :layout && return getfield(fp, :layout)
    unflatten(_child_layout(flatlayout(fp), s), parent(fp))
end

Base.propertynames(fp::FlatParameters) = (:data, :layout, _child_keys(flatlayout(fp))...)

# `@inline` for the reason `getproperty` above is: the two spellings read a layer off the same way and
# have to infer the same way. Without it the literal `:L1` of `fp[:L1]` is not propagated into
# `_child_layout`, the child layout comes back as the union of every layer's, and `unflatten` is
# dispatched dynamically — where `fp.L1` infers the one concrete layer.
@inline function Base.getindex(fp::FlatParameters, s::Symbol)
    unflatten(_child_layout(flatlayout(fp), s), parent(fp))
end

_child_layout(l::ParametersLayout, s::Symbol) = _child_layout(l.inner, s)

function _child_layout(l::NestedLayout, s::Symbol)
    haskey(l.children, s) ||
        throw(ArgumentError(string(
            "flat parameters have no entry `", s, "`; expected one of ",
            keys(l.children))))
    l.children[s]
end

function _child_layout(l::ParameterLayout, s::Symbol)
    throw(ArgumentError(string("flat parameters over a ", nameof(typeof(l)),
        " have no entry `", s, "`")))
end

_child_keys(l::ParametersLayout) = _child_keys(l.inner)
_child_keys(l::NestedLayout) = keys(l.children)
_child_keys(::ParameterLayout) = ()

"""
    NetworkParameters(fp::FlatParameters)

Unflatten `fp` back into its structured form. Errors when `fp`'s layout is over a bare `NamedTuple`
rather than a [`NetworkParameters`](@ref) — use [`unflatten`](@ref) directly for that.
"""
function NetworkParameters(fp::FlatParameters)
    _as_network_parameters(unflatten(flatlayout(fp), parent(fp)))
end

_as_network_parameters(ps::NetworkParameters) = ps
function _as_network_parameters(ps)
    throw(ArgumentError(string("these flat parameters unflatten to a ", typeof(ps),
        ", not a NetworkParameters; call `unflatten(flatlayout(fp), parent(fp))` instead")))
end

"""
    unflatten(fp::FlatParameters)

The structured form of `fp`, using the layout it carries.
"""
unflatten(fp::FlatParameters) = unflatten(flatlayout(fp), parent(fp))

# A leaf read *out of* a flat parameter set takes its numbers from the vector `fp` wraps. Left to the
# generic path it would take `fp[l.range]`, and at full length — a set of one leaf — `similar` hands
# that back as a `FlatParameters` still carrying the layout, so the leaf would come back as a reshaped
# parameter set rather than the ordinary array `unflatten` promises.
#
# On the leaf and not on `ParameterLayout`: the branch cases only pass `data` down, so the leaf is
# both where the read happens and the one place a method is not ambiguous with them — `ParameterLayout`
# against `FlatParameters` is more specific in its second argument and less in its first.
@inline unflatten(l::LeafLayout, fp::FlatParameters) = _reshape_leaf(parent(fp)[l.range], l.size)

flatten!(fp::FlatParameters, ps) = (flatten!(parent(fp), ps, flatlayout(fp)); fp)
unflatten!(ps, fp::FlatParameters) = unflatten!(ps, flatlayout(fp), parent(fp))

function Base.show(io::IO, ::MIME"text/plain", fp::FlatParameters)
    print(io, length(fp), "-element FlatParameters{", eltype(fp), "} over ",
        join(_child_keys(flatlayout(fp)), ", "))
end

# Broadcasting keeps the layout when the result still has the same length, and drops to an ordinary
# array when it does not — a layout that no longer matches its data would be worse than none.
Base.BroadcastStyle(::Type{<:FlatParameters}) = Broadcast.ArrayStyle{FlatParameters}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{FlatParameters}},
        ::Type{S}) where {S}
    fp = _find_flat(bc)
    length(bc) == length(fp) ? FlatParameters(similar(parent(fp), S), flatlayout(fp)) :
    similar(parent(fp), S, length(bc))
end

_find_flat(bc::Broadcast.Broadcasted) = _find_flat(bc.args)
_find_flat(args::Tuple) = _find_flat(_find_flat(first(args)), Base.tail(args))
_find_flat(x) = x
_find_flat(::Tuple{}) = nothing
_find_flat(fp::FlatParameters, ::Any) = fp
_find_flat(::Any, rest) = _find_flat(rest)
