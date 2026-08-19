# Recursive walks over a parameter set.
#
# Every operation this package and its consumers perform on a parameter set is one of these: recurse
# into the `NamedTuple`s, do something at each leaf, put the result back in the same shape. Writing
# the recursion once — over the `freeparameters`/`rebuild` protocol, so it never needs to know which
# structured types exist — is what keeps `changebackend`, `map_to_cpu`, `_statify`, the elementwise
# optimizer primitives and the HDF5 traversal from each being their own copy of it.
#
# The recursions are written with `Base.tail` rather than with `map` over `keys` so that they stay
# type stable and allocation free, which `flatten!`/`unflatten!` rely on.

"""
    mapparameters(f, ps)
    mapparameters(f, ps, rest...)

Apply `f` to every leaf of `ps`, returning a parameter set of the same shape.

With further arguments the trees are walked in lockstep and `f` receives one leaf from each, which is
how two parameter sets are combined entrywise. Their keys have to agree.

`f` sees *whole* leaves — a `SymmetricMatrix` arrives as a `SymmetricMatrix`. Use [`mapstorage`](@ref)
to see only the differentiable storage instead.

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
mapparameters(x -> 2x, ps).L1.W

# output

1×2 Matrix{Float64}:
 2.0  4.0
```

Combining two sets:

```jldoctest
using NeuralNetworkParameters

a = NetworkParameters((L1 = (b = [1.0, 2.0],),))
c = NetworkParameters((L1 = (b = [10.0, 20.0],),))
mapparameters(+, a, c).L1.b

# output

2-element Vector{Float64}:
 11.0
 22.0
```
"""
@inline mapparameters(f, ps::NetworkParameters,
    rest::Vararg{Any, N}) where {N} = NetworkParameters(mapparameters(
    f, params(ps), map(_as_namedtuple, rest)...))

@inline function mapparameters(f, ps::NamedTuple, rest::Vararg{Any, N}) where {N}
    nts = map(_as_namedtuple, rest)
    _check_keys(keys(ps), nts)
    NamedTuple{keys(ps)}(_map_zip(mapparameters, f, values(ps), map(values, nts)...))
end

@inline mapparameters(f, ps::Tuple, rest::Vararg{Any, N}) where {N} = _map_zip(
    mapparameters, f, ps, map(_as_tuple, rest)...)

@inline mapparameters(f, x, rest::Vararg{Any, N}) where {N} = f(x, rest...)

"""
    mapstorage(f, ps, rest...)

Like [`mapparameters`](@ref), but `f` is applied to the [`freeparameters`](@ref) of each leaf and the
leaf is [`rebuild`](@ref)ed around the result.

This is the level at which entrywise arithmetic on a structured parameter is meaningful. Halving a
`SymmetricMatrix` means halving the ``n(n+1)/2`` numbers it stores; broadcasting over its dense
``n \\times n`` interface would do twice the work, and for a skew-symmetric or triangular matrix there
is no `setindex!` to broadcast through at all.

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0; 3.0 4.0],),))
mapstorage(x -> x ./ 2, ps).L1.W

# output

2×2 Matrix{Float64}:
 0.5  1.0
 1.5  2.0
```
"""
@inline mapstorage(f, ps::NetworkParameters, rest::Vararg{
    Any, N}) where {N} = NetworkParameters(mapstorage(f, params(ps), map(_as_namedtuple, rest)...))

@inline function mapstorage(f, ps::NamedTuple, rest::Vararg{Any, N}) where {N}
    nts = map(_as_namedtuple, rest)
    _check_keys(keys(ps), nts)
    NamedTuple{keys(ps)}(_map_zip(mapstorage, f, values(ps), map(values, nts)...))
end

@inline mapstorage(f, ps::Tuple, rest::Vararg{Any, N}) where {N} = _map_zip(
    mapstorage, f, ps, map(_as_tuple, rest)...)

@inline function mapstorage(f, x, rest::Vararg{Any, N}) where {N}
    s = freeparameters(x)
    s === x && return f(x, rest...)
    rebuild(x, mapstorage(f, s, map(freeparameters, rest)...))
end

"""
    foreachparameters(f, ps, rest...)

Walk the leaves of `ps` for the side effect of `f`, in lockstep with `rest`. Returns `nothing`.

A branch or leaf of `rest` that is `nothing` **skips** that position entirely — `f` is not called
there. This is what lets a gradient tree that is missing the entries of a frozen or non-trainable
layer be walked against the parameters it belongs to, without having to fill the holes in first.

See [`mapparameters!`](@ref) for the in-place variant that returns its destination.
"""
@inline function foreachparameters(f, ps::Union{NetworkParameters, NamedTuple},
        rest::Vararg{Any, N}) where {N}
    nt = _as_namedtuple(ps)
    _foreach_zip(f, freeparameters, values(nt), map(r -> _values_for(r, keys(nt)), rest)...)
    nothing
end

@inline function foreachparameters(f, ps::Tuple, rest::Vararg{Any, N}) where {N}
    _foreach_zip(f, freeparameters, ps, map(r -> _tuple_for(r, length(ps)), rest)...)
    nothing
end

@inline function foreachparameters(f, x, rest::Vararg{Any, N}) where {N}
    _anynothing(rest) && return nothing
    f(x, rest...)
    nothing
end

"""
    mapparameters!(f, dest, srcs...)

Walk `dest` and `srcs` in lockstep, calling `f(dest_leaf, src_leaves...)` for its effect on
`dest_leaf`, and return `dest`.

As with [`foreachparameters`](@ref), a `nothing` in `srcs` skips that position.
"""
@inline function mapparameters!(f, dest, srcs::Vararg{Any, N}) where {N}
    foreachparameters(f, dest, srcs...)
    dest
end

"""
    mapstorage!(f, dest, srcs...)

Like [`mapparameters!`](@ref), but `f` is handed the [`freeparameters`](@ref) of each leaf.

No `rebuild` is needed: the storage of a leaf is the leaf's own memory, so writing into it is writing
into the leaf.
"""
@inline function mapstorage!(f, dest, srcs::Vararg{Any, N}) where {N}
    _foreach_storage(f, dest, srcs...)
    dest
end

@inline function _foreach_storage(f, ps::Union{NetworkParameters, NamedTuple},
        rest::Vararg{Any, N}) where {N}
    nt = _as_namedtuple(ps)
    _foreach_zip(f, _storage_recurse, values(nt), map(r -> _values_for(r, keys(nt)), rest)...)
    nothing
end

@inline function _foreach_storage(f, ps::Tuple, rest::Vararg{Any, N}) where {N}
    _foreach_zip(f, _storage_recurse, ps, map(r -> _tuple_for(r, length(ps)), rest)...)
    nothing
end

@inline function _foreach_storage(f, x, rest::Vararg{Any, N}) where {N}
    _anynothing(rest) && return nothing
    s = freeparameters(x)
    if s === x
        f(x, rest...)
    else
        _foreach_storage(f, s, map(freeparameters, rest)...)
    end
    nothing
end

# `foreachparameters` recurses through `freeparameters` only at the leaves it is given, whereas
# `mapstorage!` has to keep descending; the marker selects which of the two a `_foreach_zip` step
# continues with.
_storage_recurse(args...) = nothing

"""
    foldparameters(op, init, ps)

Left-fold `op` over the leaves of `ps`, starting from `init`.

The leaves are visited in the order `flatten` writes them, so a fold and a flattening agree.

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
foldparameters((n, x) -> n + length(x), 0, ps)

# output

4
```
"""
@inline foldparameters(op, init, ps::NetworkParameters) = foldparameters(op, init, params(ps))

@inline foldparameters(op, init, ps::Union{NamedTuple, Tuple}) = _fold_children(op, init, values(ps))

@inline foldparameters(op, init, x) = op(init, x)

@inline _fold_children(op, acc, ::Tuple{}) = acc
@inline _fold_children(op, acc, xs::Tuple) = _fold_children(
    op, foldparameters(op, acc, first(xs)), Base.tail(xs))

# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------

@inline _as_namedtuple(x::NetworkParameters) = params(x)
@inline _as_namedtuple(x::NamedTuple) = x
@inline _as_namedtuple(x::Nothing) = nothing

@inline _as_tuple(x::Tuple) = x
@inline _as_tuple(x::Nothing) = nothing

@inline _values_for(::Nothing, ks::Tuple) = map(_ -> nothing, ks)
@inline _values_for(x, ::Tuple) = values(_as_namedtuple(x))

@inline _tuple_for(::Nothing, n::Int) = ntuple(_ -> nothing, n)
@inline _tuple_for(x, ::Int) = x

@inline _anynothing(::Tuple{}) = false
@inline _anynothing(t::Tuple) = first(t) === nothing || _anynothing(Base.tail(t))

@inline _map_zip(recurse, f, ::Tuple{}) = ()
@inline _map_zip(recurse, f, xs::Tuple) = (
    recurse(f, first(xs)), _map_zip(recurse, f, Base.tail(xs))...)
@inline _map_zip(recurse, f, ::Tuple{}, rest::Tuple{}...) = ()
@inline _map_zip(recurse, f, xs::Tuple, rest::Tuple...) = (
    recurse(f, first(xs), map(first, rest)...),
    _map_zip(recurse, f, Base.tail(xs), map(Base.tail, rest)...)...)

@inline _foreach_zip(f, kind, ::Tuple{}) = nothing
@inline function _foreach_zip(f, kind, xs::Tuple)
    _foreach_step(f, kind, first(xs))
    _foreach_zip(f, kind, Base.tail(xs))
end
@inline _foreach_zip(f, kind, ::Tuple{}, rest::Tuple{}...) = nothing
@inline function _foreach_zip(f, kind, xs::Tuple, rest::Tuple...)
    _foreach_step(f, kind, first(xs), map(first, rest)...)
    _foreach_zip(f, kind, Base.tail(xs), map(Base.tail, rest)...)
end

@inline _foreach_step(f, ::typeof(freeparameters), args::Vararg{
    Any, N}) where {N} = foreachparameters(f, args...)
@inline _foreach_step(f, ::typeof(_storage_recurse), args::Vararg{
    Any, N}) where {N} = _foreach_storage(f, args...)

@inline _check_keys(::Tuple, ::Tuple{}) = nothing
@inline function _check_keys(ks::Tuple, nts::Tuple)
    other = first(nts)
    if other !== nothing && keys(other) != ks
        throw(ArgumentError(string("parameter sets have different keys: ", ks, " and ", keys(other))))
    end
    _check_keys(ks, Base.tail(nts))
end
