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

function _layout(ps::NetworkParameters, offset::Int)
    inner, off = _layout(params(ps), offset)
    ParametersLayout(inner), off
end

# The two branch cases, each **one** `@generated` body that lays the children out *and* builds the
# layout around them.
#
# Splitting those two steps is what this used to do, and it is the expensive way round. A generated
# body that returns a `k`-element tuple is cheap on its own — 1.27 s at 369 children — and so is
# wrapping a tuple you already hold in a `NamedTuple` and a struct — 0.13 s. Composing them across a
# function call is neither: inference has to carry the whole 369-element tuple type from the callee
# into the caller's own inference, and `parameterlayout` on that set cost **11.7 s**, of which
# the child walk alone was 0.49 s of it and the wrapping was the rest. Fused into one body, and with
# the leaf union below removed, it is **1.36 s**.
#
# Measured every way round before settling on this. A `@noinline` barrier on the child walk does not
# help (11.78 s). Neither does supplying `NestedLayout`'s type parameters instead of letting them be
# solved. And computing the child lengths first so that the children could be laid out *independently*
# — the obvious way to break the serial offset dependency — is **worse**, 15.4 s, because it adds two
# more `k`-long bodies to pay for; the serial dependency was never the cost, as an "every child at the
# same offset" control shows at 1.17 s against the real walk's 1.27 s.
#
# The offset threads left to right, as the recursion this replaces did: the ranges a layout hands out
# are the order `flatten` writes in.
@generated function _layout(ps::NamedTuple{Keys}, offset::Int) where {Keys}
    n = length(Keys)
    n == 0 && return :((NestedLayout(NamedTuple{$Keys}(()), (offset + 1):offset), offset))
    body = [:((child_1, off_1) = _layout(getfield(ps, 1), offset))]
    for i in 2:n
        push!(body, :(($(Symbol(:child_, i)), $(Symbol(:off_, i))) =
            _layout(getfield(ps, $i), $(Symbol(:off_, i - 1)))))
    end
    children = Expr(:tuple, (Symbol(:child_, i) for i in 1:n)...)
    final = Symbol(:off_, n)
    quote
        $(body...)
        NestedLayout(NamedTuple{$Keys}($children), (offset + 1):$final), $final
    end
end

@generated function _layout(ps::Tuple, offset::Int)
    n = fieldcount(ps)
    n == 0 && return :((TupleLayout((), (offset + 1):offset), offset))
    body = [:((child_1, off_1) = _layout(getfield(ps, 1), offset))]
    for i in 2:n
        push!(body, :(($(Symbol(:child_, i)), $(Symbol(:off_, i))) =
            _layout(getfield(ps, $i), $(Symbol(:off_, i - 1)))))
    end
    children = Expr(:tuple, (Symbol(:child_, i) for i in 1:n)...)
    final = Symbol(:off_, n)
    quote
        $(body...)
        TupleLayout($children, (offset + 1):$final), $final
    end
end

# A leaf, terminal or wrapped, decided by **dispatch on the storage's type** rather than by an `if`
# on `freeparameters(x) === x`.
#
# The identity test is the right *question* — it is what `isterminal` asks — and asking it inside one
# method body was measurably the wrong way to ask it. Inference could not fold the branch, so
# `_layout` on a plain `Matrix{Float32}` came back as a **three-way union**: a `LeafLayout`, and two
# shapes of `WrappedLayout`. One union per child is survivable; a branch of 369 children each of a
# three-way union is not, and it is what made `parameterlayout` cost 11.7 s on the flat parameter set
# of GMLDatasets' MNIST transformer.
#
# Splitting on `::T`/`::T` puts the question to the method table, where it is answered once per leaf
# type at compile time. Keeping the `===` *inside* the same-type method does not work and was tried:
# inference cannot prove that two arguments are the same object, so the union came straight back. The
# identity has to be decided by dispatch or not at all.
#
# What that gives up is a leaf whose `freeparameters` returns a *distinct object of its own type*,
# which this now treats as terminal where the `===` would have wrapped it. Nothing is lost, because
# such a type never worked: `_layout` would descend into the storage, ask it for *its*
# `freeparameters`, and recurse without end unless it eventually changed type. A leaf either hands
# back itself or hands back something else.
#
# [`isterminal`](@ref) still asks the identity question, and is still right to — it is a predicate a
# caller evaluates on a value it holds, not a dispatch this has to infer through.
@inline _layout(x, offset::Int) = _layout_storage(x, freeparameters(x), offset)

# same type: terminal, the numbers are copied straight out of `x`
@inline _layout_storage(x::T, ::T, offset::Int) where {T} = _leaf_layout(x, offset)

# different type: the storage is structured, so lay it out and rebuild around it
@inline _layout_storage(x, s, offset::Int) = _wrapped_layout(x, s, offset)

@inline function _leaf_layout(x, offset::Int)
    n = length(x)
    LeafLayout((offset + 1):(offset + n), _leafsize(x), x), offset + n
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
"""
flatlength(ps) = length(parameterlayout(ps))
flatlength(l::ParameterLayout) = length(l)
