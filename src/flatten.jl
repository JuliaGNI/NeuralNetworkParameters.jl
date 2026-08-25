@doc raw"""
    flatten(ps)
    flatten(T, ps)

Copy every number of `ps` into one flat `Vector`, and return it together with the
[`ParameterLayout`](@ref) needed to put it back.

Without `T` the element type is [`parameter_eltype`](@ref)`(ps)`, i.e. the parameters' own — a
`Float32` network flattens to a `Vector{Float32}`.

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
v, layout = flatten(ps)
v

# output

3-element Vector{Float64}:
 1.0
 2.0
 3.0
```

[`unflatten`](@ref) is its inverse:

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
v, layout = flatten(ps)
unflatten(layout, v) == ps

# output

true
```

# Implementation

The copy is a `copyto!` per leaf over a known range, so it runs at memory bandwidth and works
unchanged for GPU arrays — no element is ever indexed individually.

Copying rather than viewing is deliberate. A flat vector of views could not carry a different element
type from the parameters, and that is exactly what differentiating through the flat form needs: the
`unflatten` on the forward pass of `ForwardDiff` has to produce `Dual`-valued parameters over a
`Dual`-valued vector. See [`flatten!`](@ref) for the allocation-free form used in inner loops.
"""
flatten(ps) = flatten(parameter_eltype(ps), ps)

function flatten(::Type{T}, ps) where {T}
    layout = parameterlayout(ps)
    T === Union{} && length(layout) > 0 && _no_element_type_error()
    v = Vector{T}(undef, length(layout))
    flatten!(v, ps, layout)
    v, layout
end

# `parameter_eltype` reports `Union{}` for a set with nothing to promote, and an empty set flattens to
# a `Vector{Union{}}` quite legitimately. If the layout did find numbers, though, the set holds a leaf
# whose storage the promotion never reached — one that keeps it behind a non-array interface and has
# no `parameter_eltype` of its own. A `Vector{Union{}}` would fail on the first `copyto!` instead of
# naming the missing method.
@noinline function _no_element_type_error()
    throw(ArgumentError(string(
        "the leaves of these parameters promote to `Union{}`, so there is no element type to ",
        "flatten into — yet the layout found numbers to copy.\n",
        "A leaf that keeps its numbers behind an interface that is not an array's has to say what ",
        "they are:\n",
        "    NeuralNetworkParameters.parameter_eltype(x::MyLeaf) = parameter_eltype(freeparameters(x))\n",
        "Alternatively, name the element type at the call: `flatten(T, ps)`.")))
end

"""
    flatten!(v, ps, [layout])

Write the numbers of `ps` into the existing vector `v`, and return `v`.

Allocation-free when the `layout` is supplied and the call is made from compiled code, which is the
point: an optimizer that flattens its parameters once per step, or twice per inner product, should not
allocate a fresh vector each time. (Called from an uninferred context — the top level of a script, say —
the recursion is not specialised and costs a few tens of bytes per leaf on Julia 1.10.)

```julia
v, layout = flatten(ps)          # once
flatten!(v, ps, layout)          # per iteration, zero allocations
```
"""
flatten!(v::AbstractVector, ps) = flatten!(v, ps, parameterlayout(ps))

function flatten!(v::AbstractVector, ps, layout::ParameterLayout)
    length(v) == length(layout) ||
        throw(DimensionMismatch(string(
            "flat vector has length ", length(v), ", layout needs ",
            length(layout))))
    _flatten!(v, ps, layout)
    v
end

@inline _flatten!(v, ps::NetworkParameters, l::ParametersLayout) = _flatten!(v, params(ps), l.inner)
@inline _flatten!(v, ps::NamedTuple, l::NestedLayout) = _flatten_children!(v, ps, l.children)
@inline _flatten!(v, ps::Tuple, l::TupleLayout) = _flatten_children!(v, ps, l.children)
@inline _flatten!(v, x, l::WrappedLayout) = _flatten!(v, freeparameters(x), l.inner)

@inline function _flatten!(v, x, l::LeafLayout)
    _copy_out!(v, first(l.range), x, length(l.range))
    nothing
end

# `getfield` and not `values(·)[i]`: taking `values` of a branch first materialises a temporary tuple
# per branch, which at 64 children is where the last of the allocations were. `getfield(·, i)` reads
# the child in place and serves a `NamedTuple` and a `Tuple` alike.
@generated function _flatten_children!(v, xs, ls)
    n = _children_arity(xs, ls)
    calls = [:(_flatten!(v, getfield(xs, $i), getfield(ls, $i))) for i in 1:n]
    quote
        $(calls...)
        nothing
    end
end

@inline _copy_out!(v, doffs::Int, x::AbstractArray, n::Int) = (
    copyto!(v, doffs, x, firstindex(x), n); nothing)
@inline _copy_out!(v, doffs::Int, x::Number, ::Int) = (v[doffs] = x; nothing)

@doc raw"""
    unflatten(layout, v)

Rebuild the parameter set that `layout` describes from the flat vector `v`.

`v` may have a different element type from the parameters the layout was built from — that is what
makes the flat form usable for derivatives:

```julia
g = ForwardDiff.gradient(w -> loss(unflatten(layout, w)), v)
unflatten(layout, g)     # the gradient, in the shape of the parameters
```

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
v, layout = flatten(ps)
unflatten(layout, [10.0, 20.0, 30.0]).L1.W

# output

1×2 Matrix{Float64}:
 10.0  20.0
```

# Implementation

Each leaf is *copied* out of `v` rather than viewed into it, so the leaves are ordinary arrays and
cannot alias each other or the flat vector.
"""
@inline unflatten(l::ParametersLayout, v::AbstractVector) = NetworkParameters(unflatten(l.inner, v))
@inline unflatten(l::NestedLayout, v::AbstractVector) =
    NamedTuple{keys(l.children)}(_unflatten_children(l.children, v))
@inline unflatten(l::TupleLayout, v::AbstractVector) = _unflatten_children(l.children, v)
@inline unflatten(l::WrappedLayout, v::AbstractVector) = rebuild(l.prototype, unflatten(l.inner, v))
@inline unflatten(l::LeafLayout, v::AbstractVector) = _reshape_leaf(v[l.range], l.size)

@inline _reshape_leaf(data::AbstractVector, ::Tuple{}) = data[begin]
@inline _reshape_leaf(data::AbstractVector, size::Tuple) = reshape(data, size...)

@doc raw"""
    unflatten(layout, J::AbstractMatrix)

Split the rows of `J` into the shape of the parameter set — for a Jacobian taken with respect to the
flat vector, so that the block belonging to each parameter can be read off.

Each leaf becomes the ``n_\mathrm{leaf} \times \mathrm{size}(J, 2)`` row block it occupies. The leaves
are *not* rebuilt: a block of a Jacobian is not a parameter, so there is nothing to rebuild it into.
"""
@inline unflatten(l::ParametersLayout, J::AbstractMatrix) = NetworkParameters(unflatten(l.inner, J))
@inline unflatten(l::NestedLayout, J::AbstractMatrix) =
    NamedTuple{keys(l.children)}(_unflatten_children(l.children, J))
@inline unflatten(l::TupleLayout, J::AbstractMatrix) = _unflatten_children(l.children, J)
@inline unflatten(l::WrappedLayout, J::AbstractMatrix) = unflatten(l.inner, J)
@inline unflatten(l::LeafLayout, J::AbstractMatrix) = J[l.range, :]

# `unflatten` over the children of a branch layout, in the shape the rest of this package walks a
# layout tree: `Base.tail` recursion, inlined, rather than `map` over a closure.
#
# The two are equally inferable, but `map` leaves the closure over `data` to be elided and not every
# version elides it. On Julia 1.11 the recursion halves what an `unflatten` call costs, from two heap
# allocations per leaf to one; on 1.10 it is worth as much again on some shapes and nothing on others.
# It also keeps the walk off `Base`'s `Any32` fallback, which `map` drops to past 32 children and
# which returns a tuple with no concrete type.
#
# `data` is the flat vector or the Jacobian; which of the two decides the `unflatten` method, so one
# helper serves both.
@generated function _unflatten_children(layouts, data)
    calls = [:(unflatten(getfield(layouts, $i), data)) for i in 1:fieldcount(layouts)]
    :(($(calls...),))
end

"""
    unflatten!(ps, layout, v)

Write the numbers of `v` into the leaves of the existing `ps`, and return `ps`.

Allocation-free on the same terms as [`flatten!`](@ref), whose counterpart it is. The write goes through
[`freeparameters`](@ref), so only the storage of a structured leaf is touched — a `SymmetricMatrix`
has its `n(n+1)/2` numbers replaced and stays symmetric.

Requires mutable leaves; a parameter set with a scalar leaf has to use the out-of-place
[`unflatten`](@ref).
"""
function unflatten!(ps, layout::ParameterLayout, v::AbstractVector)
    length(v) == length(layout) ||
        throw(DimensionMismatch(string(
            "flat vector has length ", length(v), ", layout needs ",
            length(layout))))
    _unflatten!(ps, layout, v)
    ps
end

@inline _unflatten!(ps::NetworkParameters, l::ParametersLayout, v) = _unflatten!(params(ps), l.inner, v)
@inline _unflatten!(ps::NamedTuple, l::NestedLayout, v) = _unflatten_children!(
    ps, l.children, v)
@inline _unflatten!(ps::Tuple, l::TupleLayout, v) = _unflatten_children!(ps, l.children, v)
@inline _unflatten!(x, l::WrappedLayout, v) = _unflatten!(freeparameters(x), l.inner, v)

@inline function _unflatten!(x::AbstractArray, l::LeafLayout, v)
    copyto!(x, firstindex(x), v, first(l.range), length(l.range))
    nothing
end

function _unflatten!(x::Number, ::LeafLayout, _)
    throw(ArgumentError(string("cannot write into the immutable leaf `", x,
        "`; use the out-of-place `unflatten` for a parameter set with scalar leaves")))
end

@generated function _unflatten_children!(xs, ls, v)
    n = _children_arity(xs, ls)
    calls = [:(_unflatten!(getfield(xs, $i), getfield(ls, $i), v)) for i in 1:n]
    quote
        $(calls...)
        nothing
    end
end
