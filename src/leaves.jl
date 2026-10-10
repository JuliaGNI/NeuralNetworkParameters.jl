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

Those two are the whole protocol for a leaf that is an `AbstractArray` subtype, which every structured
parameter type in the ecosystem is. A leaf that keeps its numbers behind some *other* interface needs
one method more, because [`parameter_eltype`](@ref) must not raise and so cannot ask an arbitrary type
where its storage is:

```julia
NeuralNetworkParameters.parameter_eltype(x::MyLeaf) = parameter_eltype(freeparameters(x))
```

Without it the leaf contributes no element type, and [`flatten`](@ref) says so
rather than guessing one.

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
@inline freeparameters(x::AbstractArray) = x
@inline freeparameters(x::Number) = x

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

@doc raw"""
    storage_gradient(leaf, Δ)

The gradient with respect to the storage of `leaf` — its [`freeparameters`](@ref) — from the cotangent
`Δ` that reverse-mode differentiation gives for it.

A structured leaf presents one interface and stores another, and AD differentiates the interface. For
a symmetric matrix with storage ``S``, the cotangent is the *natural* one, a matrix ``G`` paired with
the dense interface; the parameter gradient is ``\partial L/\partial S``, which counts each stored
off-diagonal entry twice: ``G_{ij} + G_{ji}``. The two differ by a factor of two off the diagonal, so
a type whose storage is not its interface defines a method that converts one to the other.

This package calls it exactly once per top-level leaf of a set, on the accumulated cotangent, through
[`map_cotangent`](@ref), at the two places where an AD cotangent becomes a parameter gradient: the
gradient that the `ZygoteRules` extension returns for a [`NetworkParameters`](@ref), and the flat
gradient that the reverse rule of [`unflatten`](@ref) returns. The flat gradient is the structured one,
flattened. It dispatches on the primal `leaf`, because the type of `Δ` depends on how the loss used the
leaf: a leaf used twice gets a dense `Matrix`, whatever its own type is. It is never called with a
structural zero (`nothing` or a `ChainRulesCore.AbstractZero`): an untouched leaf gets a zero leaf of
its type instead. It is not called for a leaf whose cotangent is part of a `NetworkParameters`, which is
what the reverse rule of [`flatten`](@ref) returns: its leaves hold the storage gradient already.

The extension converts the gradient of `Zygote.pullback(f, ps)` and `Zygote.gradient(f, ps)` for a
`Function` `f` and a set `ps` that is the only argument, which is how a loss is differentiated with
respect to a set. Zygote's own methods answer every other call: a further argument, a callable
struct for `f`, or a set held inside another argument. Those return the structural tangent
`(params = …,)` with the natural cotangent at each leaf, and do not call this function.

A method converts an *array* cotangent only, and returns a leaf of the same type, whose storage the
flat path reads with [`freeparameters`](@ref). Its element type is the leaf's,
[`parameter_eltype`](@ref)`(leaf)`, whatever the precision of the cotangent: a loss that multiplies a
`Float32` leaf by a `Float64` constant gives that leaf a `Float64` cotangent. The default method
converts an array or a number cotangent of an `AbstractFloat` or `Integer` element type to that
element type, with no copy where the types match, as `ChainRulesCore.ProjectTo` does. A cotangent of
another number type, such as a `ForwardDiff.Dual` from forward-over-reverse differentiation, passes
through unchanged. A gradient whose leaves end up with two element types raises the `ArgumentError`
of [`NetworkParameters`](@ref) for two element types.

A structural tangent over the leaf's fields, a `NamedTuple` or a `ChainRulesCore.Tangent`, comes from
a loss that read the storage field directly and holds ``\partial L/\partial S`` already. The default
method for it rebuilds the leaf with [`rebuild`](@ref) around the components of the tangent that
belong to its storage — matched by key for a `NamedTuple` of blocks, by field name for a `Tuple` of
blocks, whose i-th block must be the leaf's i-th field, and for one block by the one field it is —
each converted by this function in turn, with a zero block where the tangent has none. A storage that
meets none of these raises an `ArgumentError`.

# Extending

```julia
function NeuralNetworkParameters.storage_gradient(A::SymmetricMatrix, G::AbstractMatrix)
    # a `SymmetricMatrix` whose storage is the lower triangle of `G + transpose(G)`, with the
    # diagonal counted once
end
```
"""
storage_gradient(leaf, Δ) = _in_eltype(parameter_eltype(leaf), Δ)

# Only a precision is converted, as `ChainRulesCore.ProjectTo` does: a cotangent of another number type,
# a `ForwardDiff.Dual` among them, passes through. A leaf with no numbers has no element type to
# convert to.
const _Precision = Union{AbstractFloat, Integer}
function _in_eltype(::Type{T}, Δ::AbstractArray{<:_Precision}) where {T}
    T === Union{} ? Δ : convert(AbstractArray{T}, Δ)
end
_in_eltype(::Type{T}, Δ::_Precision) where {T} = T === Union{} ? Δ : convert(T, Δ)
_in_eltype(_, Δ) = Δ

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

The element type to flatten `ps` into: the element type of its leaves, which is one type. Leaves of
two numeric element types raise an `ArgumentError` that names both; nothing is promoted.

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

# Implementation

This function raises only for two element types, where [`freeparameters`](@ref) raises for a leaf
without the protocol: every [`NetworkParameters`](@ref) runs its constructor through here, including
sets that hold no numbers at all — a tree with `nothing` in place of a leaf, or
`SymbolicNeuralNetworks` wrapping generated functions in one. A leaf this package cannot read numbers
out of contributes no element type, exactly as an empty set does, and both report `Union{}`. A set
that cannot be flattened is still a set with an element type; the protocol error comes from
[`parameterlayout`](@ref), which decides something with it.

The recursion follows [`freeparameters`](@ref) for an `AbstractArray` leaf — every structured
parameter type in the ecosystem — and a leaf that keeps its numbers behind another interface opts in
with one method more, described under [`freeparameters`](@ref). Asking an arbitrary type where its
storage is would mean raising, which this function must not do.

It follows it at the level of *values*, `freeparameters(x) === x`, and that is a guarantee rather than
an implementation detail. A structured leaf reports the element type of its **storage**, which need not
be the `eltype` of the interface it presents: a leaf that is an `AbstractMatrix{Float64}` over a
`Vector{Float32}` flattens into a `Vector{Float32}`, because that is what its numbers are. Deciding
the element type from the leaf's *type* instead would read the interface and be wrong about such a leaf,
which is why it is not decided there — and there is no cost to buy with it.

For a [`NetworkParameters`](@ref) the answer is already on the type, put there by its constructor, so
nothing is recomputed and the call folds to a constant. A bare `NamedTuple` or `Tuple` is read in
place at literal indices, one `@generated` body per branch shape, for the reason the head of
`walk.jl` gives — so it too costs nothing at any width, depth or shape, and `flatten(ps)` costs
exactly what `flatten(T, ps)` costs.
"""
@inline parameter_eltype(::NetworkParameters{T}) where {T} = T
@inline parameter_eltype(ps::Union{NamedTuple, Tuple}) = _promote_eltypes(ps)
@inline parameter_eltype(x::Number) = typeof(x)

@inline function parameter_eltype(x::AbstractArray)
    s = freeparameters(x)
    s === x ? eltype(x) : parameter_eltype(s)
end

# A leaf this package cannot read numbers out of contributes no element type, exactly as an empty set
# does: `nothing` or a `ChainRules` structural zero in place of a leaf, or a
# generated function that `SymbolicNeuralNetworks` keeps in a parameter set. `freeparameters` raises
# for all of these and this function must not, since every `NetworkParameters` runs its constructor
# through it. A leaf that *does* keep numbers behind a non-array interface opts in with
# `NeuralNetworkParameters.parameter_eltype(x::MyLeaf) = parameter_eltype(freeparameters(x))`;
# without it `flatten` raises rather than guessing an element type for numbers it can see.
@inline parameter_eltype(x) = Union{}

# Written out rather than recursed for the reason the head of `walk.jl` gives: a `Base.tail` chain
# costs one specialisation per child, and this one is on `flatten`'s path.
#
# It takes the *branch* and not its `values`, which is the other half of that treatment. `values` of a
# branch materialises a temporary tuple, which is free while the branch stays in registers and is not
# beyond that: `Base` unrolls a tuple up to 32 fields and drops to its `Any32` fallback past it.
# `getfield` at a literal index reads a `NamedTuple` and a `Tuple` alike, as `_flatten_children!` does
# for the same reason.
@generated function _promote_eltypes(xs::Union{NamedTuple, Tuple})
    fieldcount(xs) == 0 && return :(Union{})
    expr = :(parameter_eltype(getfield(xs, 1)))
    for i in 2:fieldcount(xs)
        expr = :(_join_eltype($expr, parameter_eltype(getfield(xs, $i))))
    end
    expr
end

# A set has one element type. `Union{}`, from a leaf with no numbers, joins any type; two numeric
# types that differ are an error, and nothing is promoted.
@inline function _join_eltype(S::Type, T::Type)
    S === Union{} && return T
    (T === Union{} || T === S) && return S
    _mixed_eltype_error(S, T)
end

@noinline function _mixed_eltype_error(S, T)
    throw(ArgumentError(string(
        "the leaves of a parameter set have one element type, and these have two: ", S, " and ",
        T, ". Convert the leaves to one of them before building the set.")))
end
