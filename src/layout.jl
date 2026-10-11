"""
    ParameterLayout

Where each leaf of a parameter set lands in its flat vector, and what has to be done to put it back.

A layout is an ordinary value, built once by [`parameterlayout`](@ref) and then reusable: it can be
stored in an optimizer cache, handed to a solver, or compared with `==`. The alternative — returning a
closure that undoes the flattening, as `ParameterHandling.flatten` does — is none of those things, and
a chain of closures is not type stable either.

The five concrete layouts mirror the five things a parameter set is made of: [`ParametersLayout`](@ref)
for a [`NetworkParameters`](@ref), [`NestedLayout`](@ref) for a `NamedTuple`, [`TupleLayout`](@ref) for
a `Tuple`, [`WrappedLayout`](@ref) for a leaf with structured storage, and [`LeafLayout`](@ref) for one
whose numbers are copied directly.
"""
abstract type ParameterLayout end

"""
    LeafLayout(range, size)

A leaf whose numbers are copied straight into the flat vector: it occupies `range`, and comes back
`reshape`d to `size` (`()` for a scalar).

This is the terminal case — [`freeparameters`](@ref) hands back something of the leaf's own type —
so there is nothing to rebuild and the layout is the shape alone. Two leaves of the same shape
therefore share one layout type whatever they hold, and a stored layout keeps no reference to the
parameters it was built from. [`WrappedLayout`](@ref) is the other case, and it does keep a
prototype.
"""
struct LeafLayout{N} <: ParameterLayout
    range::UnitRange{Int}
    size::NTuple{N, Int}
end

"""
    WrappedLayout(prototype, inner)

A leaf whose [`freeparameters`](@ref) are themselves structured — a `SymmetricMatrix` storing a
vector, or a horizontal lift storing two blocks. `inner` is the layout of that storage; unflattening
runs it and then calls [`rebuild`](@ref) on `prototype`.
"""
struct WrappedLayout{P, LT <: ParameterLayout} <: ParameterLayout
    prototype::P
    inner::LT
end

"""
    NestedLayout(children, range)

The layout of a `NamedTuple`: one child layout per key, together spanning `range`.
"""
struct NestedLayout{Keys, VT <: Tuple} <: ParameterLayout
    children::NamedTuple{Keys, VT}
    range::UnitRange{Int}
end

"""
    TupleLayout(children, range)

The layout of a `Tuple`, positional counterpart of [`NestedLayout`](@ref).
"""
struct TupleLayout{VT <: Tuple} <: ParameterLayout
    children::VT
    range::UnitRange{Int}
end

"""
    ParametersLayout(inner)

The layout of a [`NetworkParameters`](@ref): the layout of the `NamedTuple` it wraps, tagged so that
unflattening returns a `NetworkParameters` again rather than a bare `NamedTuple`.
"""
struct ParametersLayout{LT <: NestedLayout} <: ParameterLayout
    inner::LT
end

"""
    parameterrange(layout)

The stretch of the flat vector that `layout` occupies.
"""
@inline parameterrange(l::LeafLayout) = l.range
@inline parameterrange(l::NestedLayout) = l.range
@inline parameterrange(l::TupleLayout) = l.range
@inline parameterrange(l::WrappedLayout) = parameterrange(l.inner)
@inline parameterrange(l::ParametersLayout) = parameterrange(l.inner)

@inline Base.length(l::ParameterLayout) = length(parameterrange(l))

Base.:(==)(a::LeafLayout, b::LeafLayout) = a.range == b.range && a.size == b.size
Base.:(==)(a::WrappedLayout, b::WrappedLayout) = a.inner == b.inner
function Base.:(==)(a::NestedLayout, b::NestedLayout)
    a.range == b.range && a.children == b.children
end
Base.:(==)(a::TupleLayout, b::TupleLayout) = a.range == b.range && a.children == b.children
Base.:(==)(a::ParametersLayout, b::ParametersLayout) = a.inner == b.inner

function Base.show(io::IO, l::ParameterLayout)
    print(io, nameof(typeof(l)), "(", first(parameterrange(l)),
        ":", last(parameterrange(l)), ")")
end

@doc raw"""
    parameterlayout(ps)

Build the [`ParameterLayout`](@ref) of `ps` in one walk.

The leaves are laid out in the order they are encountered, depth first, so the flat vector reads like
the parameter set does.

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
layout = parameterlayout(ps)
(length(layout), parameterrange(layout.inner.children.L1.children.b))

# output

(4, 3:3)
```
"""
parameterlayout(ps) = first(_layout(ps, 0))

# `_layout(x, offset)` returns the layout of `x` placed after `offset` entries, and the offset past
# its last entry.
function _layout(ps::NetworkParameters, offset::Int)
    inner, off = _layout(params(ps), offset)
    ParametersLayout(inner), off
end

# A branch: a map over the children that carries the offset, so the ranges are in the order
# `flatten` writes. An empty branch has the empty range `(offset + 1):offset`.
function _layout(ps::NamedTuple{Keys}, offset::Int) where {Keys}
    children, off = _layout_children(ps, offset)
    NestedLayout(NamedTuple{Keys}(children), (offset + 1):off), off
end

function _layout(ps::Tuple, offset::Int)
    children, off = _layout_children(ps, offset)
    TupleLayout(children, (offset + 1):off), off
end

# Each child laid out after the one before it, and the offset past the last of them.
_layout_children(ps, offset::Int) = _unrolled_mapaccum(_layout_after, offset, ps)

@inline _layout_after(offset::Int, x) = _layout(x, offset)

# A leaf is terminal or wrapped by dispatch on the type of its storage: a storage of the leaf's own
# type is the leaf, so its numbers are copied straight out of it. An `===` test in one method body
# would give each leaf layout a union type, which inference cannot fold.
@inline _layout(x, offset::Int) = _layout_storage(x, freeparameters(x), offset)

# same type: terminal, the numbers are copied straight out of `x`
@inline _layout_storage(x::T, ::T, offset::Int) where {T} = _leaf_layout(x, offset)

# different type: the storage is structured, so lay it out and rebuild around it
@inline _layout_storage(x, s, offset::Int) = _wrapped_layout(x, s, offset)

@inline function _leaf_layout(x, offset::Int)
    n = length(x)
    LeafLayout((offset + 1):(offset + n), _leafsize(x)), offset + n
end

@inline function _wrapped_layout(x, s, offset::Int)
    inner, off = _layout(s, offset)
    WrappedLayout(x, inner), off
end

_leafsize(x::AbstractArray) = size(x)
_leafsize(::Number) = ()

"""
    flatlength(ps)

The number of entries `ps` flattens to, without allocating the flat vector.

Deliberately not called `parameterlength`: `AbstractNeuralNetworks` has a function of that name for
the parameter count of a *model*, and the two would collide on `using`.

This returns an `Int`, so nothing downstream of the call depends on the layout's type. Anything that
wants only the size should call it; anything that has to unflatten later needs the layout and cannot.
"""
flatlength(ps) = length(parameterlayout(ps))
flatlength(l::ParameterLayout) = length(l)
