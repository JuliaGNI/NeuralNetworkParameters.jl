# Recursive walks over a parameter set.
#
# Every operation this package and its consumers perform on a parameter set is one of these: recurse
# into the `NamedTuple`s, do something at each leaf, put the result back in the same shape. Writing
# the recursion once — over the `freeparameters`/`rebuild` protocol, so it never needs to know which
# structured types exist — is what keeps `changebackend`, `map_to_cpu`, `_statify`, the elementwise
# optimizer primitives and the HDF5 traversal from each being their own copy of it.
#
# The walk *down* the tree — `mapparameters` and friends dispatching on `NetworkParameters`,
# `NamedTuple`, `Tuple` or a leaf — is ordinary multiple dispatch. The walk *across* the children of
# one branch is not, and it is where the care goes. Neither `map` over `keys` nor a `for` loop will do:
# `map` over a closure cost the out-of-place `unflatten` an allocation per leaf on Julia 1.11, and a
# loop over `keys` indexes a heterogeneous tuple at a runtime `i`, which is type-unstable and costs a
# dynamic dispatch — and a boxed allocation — per child. `flatten!`/`unflatten!` are allocation-free
# only because neither happens.
#
# These across-children walks used to be `@inline`d `Base.tail` chains, which met that bar and
# introduced a different problem: `Base.tail` produces a *new tuple type at every level*, so a branch
# with `k` children costs `k` specialisations whose argument types are each `O(k)` long, and inference
# on that grows as `k³`. Measured, `flatten` on a flat set: 0.17 s at 32 children, 2.2 s at 64, 17.6 s
# at 128, and past ten minutes at the 369 of the MNIST transformer in GMLDatasets.jl — which is a real
# parameter set, not a synthetic worst case. Dropping the `@inline` does not help; it makes the same
# shape slower still (3.2 s at 64) *and* costs 187 808 bytes, because the chain is what kept the
# per-leaf `copyto!` statically dispatched.
#
# So `_children_across` writes the flat body out instead: one `@generated` expansion per branch shape,
# `k` statements indexing `xs[1] … xs[k]` at *literal* indices. One specialisation instead of `k`, no
# new tuple types at all, every index inferred at a constant, and the same straight-line code the
# chain used to inline down to. See open issue D9 in the CHANGELOG for the numbers before and after.

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
    NamedTuple{keys(ps)}(_map_zip(mapparameters, f, ps, nts...))
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
    NamedTuple{keys(ps)}(_map_zip(mapstorage, f, ps, nts...))
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
    _foreach_zip(f, freeparameters, nt, map(r -> _values_for(r, keys(nt)), rest)...)
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
    _foreach_zip(f, _storage_recurse, nt, map(r -> _values_for(r, keys(nt)), rest)...)
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

@inline foldparameters(op, init, ps::Union{NamedTuple, Tuple}) = _fold_children(op, init, ps)

@inline foldparameters(op, init, x) = op(init, x)

# A *left* fold, and the expansion keeps it one: `op` is the caller's and need not be associative,
# so the nesting below is built inside out rather than halved.
@generated function _fold_children(op, acc, xs)
    expr = :acc
    for i in 1:fieldcount(xs)
        expr = :(foldparameters(op, $expr, getfield(xs, $i)))
    end
    expr
end

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

# `||` and not `any`, so it still short-circuits: the whole point is not to look at the rest once a
# `nothing` has been found.
@generated function _anynothing(t::Tuple)
    fieldcount(t) == 0 && return :(false)
    tests = [:(getfield(t, $i) === nothing) for i in 1:fieldcount(t)]
    foldr((a, b) -> :($a || $b), tests)
end

# One method for both arities: with no `rest` the inner splat is empty and this is the plain map.
@generated function _map_zip(recurse, f, xs, rest...)
    n = _children_arity(xs, rest...)
    calls = [Expr(:call, :recurse, :f, :(getfield(xs, $i)),
                  (:(getfield(rest[$j], $i)) for j in 1:length(rest))...) for i in 1:n]
    :(($(calls...),))
end

@generated function _foreach_zip(f, kind, xs, rest...)
    n = _children_arity(xs, rest...)
    calls = [Expr(:call, :_foreach_step, :f, :kind, :(getfield(xs, $i)),
                  (:(getfield(rest[$j], $i)) for j in 1:length(rest))...) for i in 1:n]
    quote
        $(calls...)
        nothing
    end
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

# The arity check the `Base.tail` chains used to get for free, by running out of `Tuple{}` methods.
# Raised from the generated bodies, so it is a clear error at specialisation time rather than a
# `MethodError` naming a tuple type nobody wrote.
function _children_arity(xs, rest...)
    n = fieldcount(xs)
    for (j, r) in enumerate(rest)
        fieldcount(r) == n || throw(ArgumentError(string(
            "parameter sets walked together must have the same number of children at every level; ",
            "got ", n, " and, in argument ", j + 1, ", ", fieldcount(r), ".")))
    end
    n
end
