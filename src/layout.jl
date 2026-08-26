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

# The container case: the child walk, and then a step that wraps what it returns.
#
# Until 0.2.3 this was the expensive shape on Julia 1.11, and the cost was read as a cost of the
# composition itself — a `@generated` body yielding a large type, composed across a function call with
# a step that wraps it. It was not. `parameterlayout` on a flat 369-leaf set cost 1.35 s as a bare
# `NamedTuple` and **13.40 s** inside a `NetworkParameters`; at 768 leaves, 2.77 s against **87.77 s**.
# The cause was [`LeafLayout`](@ref)'s `prototype` type parameter, which put each leaf's concrete array
# type into the layout type of every branch above it — 1849 nodes in the type tree of a 369-leaf
# wrapped layout, against 742 without it. This method is where a caller paid for that, because it
# infers through the child walk's whole return type to reach `ParametersLayout(inner)`, and inference
# on such a type grows faster than the type does: 2.5 times the type was 13 times the time.
#
# With the parameter gone the two shapes cost the same — 1.05 s wrapped against 1.04 s bare at 369,
# 3.01 s against 3.07 s at 768, 0.84 s against 0.85 s on a 16 × 24 set of the same 384 leaves — and so
# do the three supported Julia versions: 1.05 s on 1.11, 1.13 s on 1.12, 1.03 s on 1.13, at 369 wrapped,
# where the compat floor used to be the one that behaved differently. Nesting was never the driver and
# is not one now, at 384 leaves in 16 branches against 369 in one.
#
# So there is nothing left here to move, and fusing the two steps into one `@generated` body — which
# the note this replaces measured, and declined — answers a question that no longer arises.
#
# What survives is where to look rather than what to change. **The total is what a consumer compiles,
# and it does not depend on which method holds it.** On 0.2.2 the bare and wrapped paths split their
# cost completely differently and still summed to nearly the same figure: 1.35 + 14.38 + 4.23 s bare
# against 13.50 + 4.14 + 4.26 s wrapped, over `parameterlayout`, `flatten` and `unflatten`. Moving a
# second out of one entry point puts it into another; taking the type parameter out took the whole
# 369-wrapped path from 21.90 s to **8.64 s**.
#
# The six shape-against-shape figures are `scripts/leaf_layout_cost.jl` and the three-column splits are
# `scripts/wide_branch_cost.jl`, one process per row in both. The two agree on the 369-wrapped row they
# share to within a tenth of a second, 13.40 s against 13.50 s, which is the spread to read the rest at.
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
# the child walk alone was 0.49 s of it and the wrapping was the rest. Fused into one body, with the
# leaf union below removed and (since 0.2.3) the leaf's array type out of its layout type, it is
# **1.04 s**.
#
# The container case above is that same composition, and it is left standing. The note there says why:
# what made it expensive was not the composition.
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

That used to be a large difference in compile time as well — on Julia 1.11, 1.26 s here against 13.20 s
through [`parameterlayout`](@ref) on a 369-leaf `NetworkParameters`. It is not one any more: 0.2.3 took
that set to 1.05 s through `parameterlayout` against 1.03 s here. The reason to prefer this is the type
it hands back, not the clock.
"""
flatlength(ps) = length(parameterlayout(ps))
flatlength(l::ParameterLayout) = length(l)
