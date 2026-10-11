# Recursive walks over a parameter set.
#
# Each walk recurses into the `NamedTuple`s and `Tuple`s, does something at each leaf and, for a map,
# puts the result back in the same shape. The recursion is written once, over the
# `freeparameters`/`rebuild` protocol, so it never needs to know which structured types exist. The
# walk down the tree is dispatch on `NetworkParameters`, `NamedTuple`, `Tuple` or a leaf; the walk
# across the children of one branch is one call of a primitive of `unrolled.jl`.

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
    _rewrap_children(ps, _map_zip(mapparameters, f, ps, nts...))
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

A `nothing` in place of a leaf reaches `f` as far as there is one leaf to pair it with: the storage of
a `SymmetricMatrix` or a manifold element is a single array, so `f` is handed that array and the
`nothing`. A leaf whose storage is *several* blocks has nothing to pair one `nothing` with, and raises
as a `nothing` branch does — an out-of-place walk has nothing to put in the hole either way.

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
    _rewrap_children(ps, _map_zip(mapstorage, f, ps, nts...))
end

@inline mapstorage(f, ps::Tuple, rest::Vararg{Any, N}) where {N} = _map_zip(
    mapstorage, f, ps, map(_as_tuple, rest)...)

@inline function mapstorage(f, x, rest::Vararg{Any, N}) where {N}
    s = freeparameters(x)
    s === x && return f(x, rest...)
    rebuild(x, mapstorage(f, s, map(_leaf_storage, rest)...))
end

"""
    foreachparameters(f, ps, rest...)

Walk the leaves of `ps` for the side effect of `f`, in lockstep with `rest`. Returns `nothing`.

The children of a keyed branch are paired **by key**, and the keys have to agree — a `rest` whose keys
are the same set in a different order is an `ArgumentError` and not a silent crossing-over. A `Tuple`
branch is paired positionally, since the blocks of a multi-block leaf have no keys to agree on.

A branch or leaf of `rest` that is `nothing` **skips** that position entirely — `f` is not called
there. This is what lets a tree that is missing the entries of a frozen or non-trainable layer be
walked against the parameters it belongs to, without having to fill the holes in first.

Allocation-free, at any width of branch, any depth of nesting and any number of `rest`: the branches are
indexed in place rather than taken apart, so nothing is materialised on the way in.

See [`mapparameters!`](@ref) for the in-place variant that returns its destination.
"""
@inline function foreachparameters(f, ps::Union{NetworkParameters, NamedTuple, Tuple},
        rest::Vararg{Any, N}) where {N}
    b = _branch(ps)
    _unrolled_foreach(_FixFirst(foreachparameters, f), b, map(Base.Fix1(_like, b), rest)...)
end

@inline function foreachparameters(f, x, rest::Vararg{Any, N}) where {N}
    any(isnothing, rest) && return nothing
    f(x, rest...)
    nothing
end

"""
    mapparameters!(f, dest, srcs...)

Walk `dest` and `srcs` in lockstep, calling `f(dest_leaf, src_leaves...)` for its effect on
`dest_leaf`, and return `dest`.

This is the walk an optimizer runs every iteration, and it allocates nothing at any width or depth. As
with [`foreachparameters`](@ref), a `nothing` in `srcs` skips that position, and the keys of a keyed
branch have to agree.
"""
@inline function mapparameters!(f, dest, srcs::Vararg{Any, N}) where {N}
    foreachparameters(f, dest, srcs...)
    dest
end

"""
    mapstorage!(f, dest, srcs...)

Like [`mapparameters!`](@ref), but `f` is handed the [`freeparameters`](@ref) of each leaf.

No `rebuild` is needed: the storage of a leaf is the leaf's own memory, so writing into it is writing
into the leaf. Allocation-free on the same terms.
"""
@inline function mapstorage!(f, dest, srcs::Vararg{Any, N}) where {N}
    _foreach_storage(f, dest, srcs...)
    dest
end

@inline function _foreach_storage(f, ps::Union{NetworkParameters, NamedTuple, Tuple},
        rest::Vararg{Any, N}) where {N}
    b = _branch(ps)
    _unrolled_foreach(_FixFirst(_foreach_storage, f), b, map(Base.Fix1(_like, b), rest)...)
end

@inline function _foreach_storage(f, x, rest::Vararg{Any, N}) where {N}
    any(isnothing, rest) && return nothing
    s = freeparameters(x)
    if s === x
        f(x, rest...)
    else
        _foreach_storage(f, s, map(freeparameters, rest)...)
    end
    nothing
end

"""
    foldparameters(op, init, ps)
    foldparameters(op, init, ps, rest...)

Left-fold `op` over the leaves of `ps`, starting from `init`.

With further arguments the trees are walked in lockstep and `op` receives one leaf from each —
`op(acc, a_leaf, b_leaf, …)` — which is how an inner product or a quadrature norm over a parameter
set is taken without flattening it first. The children of a keyed branch are paired **by key** and the
keys have to agree, exactly as for [`mapparameters`](@ref); a `Tuple` branch is paired positionally,
since the blocks of a multi-block leaf have no keys to agree on.

The leaves are visited in the order [`flatten`](@ref) writes them, so a fold and a flattening agree.

`op` sees *whole* leaves — a `SymmetricMatrix` arrives as a `SymmetricMatrix`. Use [`foldstorage`](@ref)
to fold over the differentiable storage instead.

Where [`foreachparameters`](@ref) skips a set that is `nothing`, a fold raises: it reduces every leaf
it is given, so a set left out would make the result a partial sum without saying so. A `nothing` in
place of a single *leaf* still reaches `op`, exactly as it reaches `f` in [`mapparameters`](@ref) —
what a missing leaf contributes to the sum is the caller's to decide, and only the caller's.

Allocation-free at any width, depth and arity — **provided a caller that hands `op` on through a
function of its own annotates it `::F where {F}` there.** Julia does not specialise on a function
argument it never sees called, so without that annotation `op` arrives boxed and every leaf costs a
dynamic dispatch. A closure that *captures* a function needs nothing, a closure being its own type,
which is why `(acc, x) -> acc + abs2(f(x))` is the way to fold a function of each leaf and no second
function argument is needed here.

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
foldparameters((n, x) -> n + length(x), 0, ps)

# output

4
```

Two sets in lockstep, which is ``\\sum_i a_ib_i`` without a flat vector of either:

```jldoctest
using NeuralNetworkParameters

a = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
b = NetworkParameters((L1 = (W = [4.0 5.0], b = [6.0]),))
foldparameters((acc, x, y) -> acc + sum(x .* y), 0.0, a, b)

# output

32.0
```
"""
@inline function foldparameters(op, init, ps::Union{NetworkParameters, NamedTuple, Tuple},
        rest::Vararg{Any, N}) where {N}
    b = _branch(ps)
    rs = map(Base.Fix1(_like, b), rest)
    any(isnothing, rs) && _no_folded_skip_error()
    _unrolled_foldl(_FixFirst(foldparameters, op), init, b, rs...)
end

@inline foldparameters(op, init, x, rest::Vararg{Any, N}) where {N} = op(init, x, rest...)

"""
    foldstorage(op, init, ps, rest...)

Like [`foldparameters`](@ref), but `op` is handed the [`freeparameters`](@ref) of each leaf.

This is the level at which a reduction over a structured parameter is meaningful, and it is the level
[`flatten`](@ref) writes. The pairing of a `SymmetricMatrix` is over the ``n(n+1)/2`` numbers it
stores; reading its dense ``n \\times n`` interface instead would count every off-diagonal entry
twice, and for a skew-symmetric or triangular matrix there is no dense reading to be had at all.

A `nothing` in place of a leaf reaches `op` as far as there is one leaf to pair it with, on the terms
[`mapstorage`](@ref) states: the storage of a `SymmetricMatrix` is a single array, and a leaf whose
storage is several blocks raises instead, since one `nothing` cannot stand for each of them.

# Examples

```jldoctest
using NeuralNetworkParameters

a = NetworkParameters((L1 = (b = [1.0, 2.0],),))
b = NetworkParameters((L1 = (b = [3.0, 4.0],),))
foldstorage((acc, x, y) -> acc + sum(x .* y), 0.0, a, b)

# output

11.0
```
"""
@inline function foldstorage(op, init, ps::Union{NetworkParameters, NamedTuple, Tuple},
        rest::Vararg{Any, N}) where {N}
    b = _branch(ps)
    rs = map(Base.Fix1(_like, b), rest)
    any(isnothing, rs) && _no_folded_skip_error()
    _unrolled_foldl(_FixFirst(foldstorage, op), init, b, rs...)
end

@inline function foldstorage(op, init, x, rest::Vararg{Any, N}) where {N}
    s = freeparameters(x)
    s === x && return op(init, x, rest...)
    foldstorage(op, init, s, map(_leaf_storage, rest)...)
end

@noinline _no_folded_skip_error() = throw(ArgumentError(string(
    "`foldparameters` and `foldstorage` reduce every leaf they are given, so a set that is `nothing` ",
    "would make the result a partial sum without saying so; fill the gaps in, or walk it with ",
    "`foreachparameters` and accumulate into a `Ref`.")))

# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------

@inline _as_namedtuple(x::NetworkParameters) = params(x)
@inline _as_namedtuple(x::NamedTuple) = x
@inline _as_namedtuple(x::Nothing) = nothing

@inline _as_tuple(x::Tuple) = x
@inline _as_tuple(x::Nothing) = nothing

# The branch of `ps`, and a further set `x` as a branch of the same kind: keyed beside a keyed
# branch, positional beside a positional one.
@inline _branch(ps::NetworkParameters) = params(ps)
@inline _branch(ps) = ps
@inline _like(::NamedTuple, x) = _as_namedtuple(x)
@inline _like(::Tuple, x) = _as_tuple(x)

# The storage of a further set's leaf, where a `nothing` stays a `nothing`. A leaf whose storage is
# one array then hands `f` or `op` that array and the `nothing`; a leaf whose storage is several blocks
# has nothing to pair one `nothing` with, and the branch walk below it raises.
@inline _leaf_storage(::Nothing) = nothing
@inline _leaf_storage(x) = freeparameters(x)

# The out-of-place walk across the children of a branch. A branch of at most
# `_WRITE_OUT_MAX_CHILDREN` children takes the unrolled map; a wider one takes `Base.map`, which
# compiles faster and returns a container whose type is not inferred. `mapparameters` and
# `mapstorage` run once per cache or per device move, not per iteration, so for them the compile time
# matters more. 32 is the width to which `Base` unrolls a tuple itself.
const _WRITE_OUT_MAX_CHILDREN = 32

@inline function _map_zip(recurse, f, xs, rest::Vararg{Any, N}) where {N}
    # An out-of-place walk has nothing to put where a set is `nothing`.
    any(isnothing, rest) && _no_skipped_set_error()
    _children_arity(typeof(xs), map(typeof, rest))
    g = _FixFirst(recurse, f)
    fieldcount(typeof(xs)) > _WRITE_OUT_MAX_CHILDREN ? map(g, xs, rest...) :
    _unrolled_map(g, xs, rest...)
end

@noinline _no_skipped_set_error() = throw(ArgumentError(string(
    "`mapparameters` and `mapstorage` build a new set and so have nothing to put where a `nothing` ",
    "stands; use `mapparameters!`, `mapstorage!` or `foreachparameters`, which skip that position.")))

# The children of a branch in the branch's own shape. `Base.map` over a `NamedTuple` returns one keyed
# already.
@inline _rewrap_children(::NamedTuple{Keys}, children::NamedTuple{Keys}) where {Keys} = children
@inline _rewrap_children(::NamedTuple{Keys}, children) where {Keys} = NamedTuple{Keys}(children)
@inline _rewrap_children(::Tuple, children) = children

@inline _check_keys(::Tuple, ::Tuple{}) = nothing
@inline function _check_keys(ks::Tuple, nts::Tuple)
    other = first(nts)
    if other !== nothing && keys(other) != ks
        throw(ArgumentError(string("parameter sets have different keys: ", ks, " and ", keys(other))))
    end
    _check_keys(ks, Base.tail(nts))
end
