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
    LeafLayout(range, size, prototype)

A leaf whose numbers are copied straight into the flat vector: it occupies `range`, and comes back
`reshape`d to `size` (`()` for a scalar).

`prototype` is the leaf it was built from, kept so that [`rebuild`](@ref) has something to rebuild
against.
"""
struct LeafLayout{N, P} <: ParameterLayout
    range::UnitRange{Int}
    size::NTuple{N, Int}
    prototype::P
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
parameterrange(l::LeafLayout) = l.range
parameterrange(l::NestedLayout) = l.range
parameterrange(l::TupleLayout) = l.range
parameterrange(l::WrappedLayout) = parameterrange(l.inner)
parameterrange(l::ParametersLayout) = parameterrange(l.inner)

Base.length(l::ParameterLayout) = length(parameterrange(l))

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

function _layout(ps::NetworkParameters, offset::Int)
    inner, off = _layout(params(ps), offset)
    ParametersLayout(inner), off
end

function _layout(ps::NamedTuple, offset::Int)
    children, off = _layout_children(values(ps), offset)
    NestedLayout(NamedTuple{keys(ps)}(children), (offset + 1):off), off
end

function _layout(ps::Tuple, offset::Int)
    children, off = _layout_children(ps, offset)
    TupleLayout(children, (offset + 1):off), off
end

function _layout(x, offset::Int)
    s = freeparameters(x)
    if s === x
        n = length(x)
        LeafLayout((offset + 1):(offset + n), _leafsize(x), x), offset + n
    else
        inner, off = _layout(s, offset)
        WrappedLayout(x, inner), off
    end
end

_layout_children(::Tuple{}, offset::Int) = ((), offset)

function _layout_children(xs::Tuple, offset::Int)
    child, off = _layout(first(xs), offset)
    rest, final = _layout_children(Base.tail(xs), off)
    (child, rest...), final
end

_leafsize(x::AbstractArray) = size(x)
_leafsize(::Number) = ()

"""
    flatlength(ps)

The number of entries `ps` flattens to, without allocating the flat vector.

Deliberately not called `parameterlength`: `AbstractNeuralNetworks` has a function of that name for
the parameter count of a *model*, and the two would collide on `using`.
"""
flatlength(ps) = length(parameterlayout(ps))
flatlength(l::ParameterLayout) = length(l)
