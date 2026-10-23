# Walks over a parameter set.
#
# Each walk goes into the `NamedTuple`s and `Tuple`s, does something at each leaf and, for a map,
# puts the result back in the same shape. The walk is written once, over the
# `freeparameters`/`rebuild` protocol, so it never needs to know which structured types exist.
#
# `mapparameters`, the in-place walks and the folds read the whole tree off the types of their
# arguments and are one generated body each, with one call per leaf and no recursion at run time. A
# walk often runs inside the function of another walk, such as an optimizer step or the building of a
# cache per leaf, and inference widens the types of a method that it finds twice on its stack with
# more deeply nested arguments; a walk that recurses through its own methods there dispatches
# dynamically, allocates at every leaf, and a map returns a type that is not concrete. `mapstorage`
# builds a new tree as well: its walk down the tree is dispatch on `NetworkParameters`, `NamedTuple`,
# `Tuple` or a leaf, and its walk across the children of one branch is one call of a primitive of
# `unrolled.jl`.

const _Branch = Union{NetworkParameters, NamedTuple, Tuple}

# The type of the further set of type `R` beside a branch of type `B`, as `_like` makes it: `Nothing`
# for a set that is `nothing`, `missing` where `R` is not concrete and only the run time knows, and
# `nothing` where `_like` has no method.
function _like_type(@nospecialize(B::Type), @nospecialize(R::Type))
    R === Nothing && return Nothing
    isconcretetype(R) || return missing
    B <: NamedTuple && R <: NetworkParameters && return fieldtype(R, 1)
    B <: NamedTuple && R <: NamedTuple && return R
    B <: Tuple && R <: Tuple && return R
    nothing
end

# `mapparameters` maps across a branch of more than `_WRITE_OUT_MAX_CHILDREN` children with `Base.map`,
# which compiles faster than a body written out per child and returns a container whose type is not
# inferred, and `mapstorage` does so through `_map_zip`. The out-of-place walks run once per cache or
# per device move, not per iteration, so for them the compile time matters more. 32 is the width to
# which `Base` unrolls a tuple itself.
const _WRITE_OUT_MAX_CHILDREN = 32

# The nodes of a whole-tree walk, in the order `flatten` writes the leaves, pushed to `nodes`. A node
# is `(x, rs, leaf, skip, wide)`: the expression that reads a leaf of the first set, those that read
# the node beside it in each further set, whether its type is known to be a leaf's and the type of
# each node beside it is concrete, so that none can be a `nothing` the type does not show, whether a
# further set is `nothing` there, and whether it is a branch that the walk maps across at run time. A
# node whose children only the run time knows, because a type is not concrete, is pushed as a leaf of
# a type that is not known, and the walk dispatches on it. An `Expr` pushed instead ends the walk:
# `_like` where it raises, or `stop` at a branch beside a further set of `nothing`, where `stop` is not
# `nothing`.
#
# An `out_of_place` walk checks the keys of a branch before its number of children, as
# `mapparameters` reports two keyed branches of different widths by their keys, and pushes a branch of
# more than `_WRITE_OUT_MAX_CHILDREN` children whole, after these checks. The in-place walks and the
# folds report the number of children first, as the primitives do.
#
# Returns the shape of the node, or `nothing` where the walk ended: an expression that builds the node
# again from the value of each node it holds, which is named by `_node_name` of the node's index, so
# that a map returns a tree of the shape of its first set. The types are held in `Vector{Any}`s and
# not specialised on, since this runs once per generated body.
function _tree_nodes!(nodes::Vector{Any}, @nospecialize(X::Type), Rs::Vector{Any}, x,
        rs::Vector{Any}, stop, out_of_place::Bool)
    if !(isconcretetype(X) && X <: _Branch)
        push!(nodes,
            (x, rs, isconcretetype(X) && all(isconcretetype, Rs),
                any(R -> R === Nothing, Rs), false))
        return _node_name(length(nodes))
    end
    wrapped = X <: NetworkParameters
    if wrapped
        X, x = fieldtype(X, 1), :(getfield($x, 1))
    end
    rewrap(shape) = wrapped ? :(NetworkParameters($shape)) : shape
    Ls = Any[_like_type(X, R) for R in Rs]
    if any(ismissing, Ls)
        push!(nodes, (x, rs, false, false, false))
        return rewrap(_node_name(length(nodes)))
    end
    j = findfirst(isnothing, Ls)
    if j !== nothing
        push!(nodes, :(_like($x, $(rs[j]))))
        return nothing
    end
    if stop !== nothing && any(L -> L === Nothing, Ls)
        push!(nodes, stop)
        return nothing
    end
    ks = _branch_keys(X)
    out_of_place && foreach(j -> _check_child_keys(ks, Ls[j], j), eachindex(Ls))
    n = _children_arity(X, Tuple(Ls))
    out_of_place || foreach(j -> _check_child_keys(ks, Ls[j], j), eachindex(Ls))
    ls = Any[R <: NetworkParameters ? :(getfield($r, 1)) : r for (R, r) in zip(Rs, rs)]
    if out_of_place && n > _WRITE_OUT_MAX_CHILDREN
        push!(nodes, (x, ls, false, false, true))
        return rewrap(_node_name(length(nodes)))
    end
    children = Any[]
    for i in 1:n
        Cs = Any[L === Nothing ? Nothing : fieldtype(L, i) for L in Ls]
        cs = Any[L === Nothing ? :nothing : :(getfield($l, $i)) for (L, l) in zip(Ls, ls)]
        child = _tree_nodes!(
            nodes, fieldtype(X, i), Cs, :(getfield($x, $i)), cs, stop, out_of_place)
        child === nothing && return nothing
        push!(children, child)
    end
    rewrap(ks === nothing ? :(($(children...),)) : :(NamedTuple{$ks}(($(children...),))))
end

# The name of the value of node `k` in the body of a map
_node_name(k::Int) = Symbol(:node, k)

# The nodes of a generated walk of `ps` and the further sets `rest`, read from its arguments, and the
# shape of `ps`
function _tree_nodes(@nospecialize(ps::Type), @nospecialize(rest::Tuple), stop,
        out_of_place::Bool = false)
    nodes = Any[]
    shape = _tree_nodes!(
        nodes, ps, Any[rest...], :ps, Any[:(getfield(rest, $j)) for j in eachindex(rest)],
        stop, out_of_place)
    nodes, shape
end

# The body of an in-place walk. Nothing where a further set is `nothing`; `f(x, rs...)` at a leaf of a
# known type where `direct`; and `walk(f, x, rs...)` at every other node, which at a leaf is the leaf
# method of `walk`. A leaf method between the body and `f` would be on the inference stack twice
# when the walk runs inside the function of another one, both times called from the same generated
# method, and inference would widen the second; so a walk whose leaf method is only a call of `f`
# calls `f` directly.
function _foreach_body(walk::Symbol, direct::Bool, @nospecialize(ps::Type),
        @nospecialize(rest::Tuple))
    body = Any[Expr(:meta, :inline)]
    for n in first(_tree_nodes(ps, rest, nothing))
        if n isa Expr
            push!(body, n)
        else
            x, rs, leaf, skip, _ = n
            skip ||
                push!(body, direct && leaf ? :(f($x, $(rs...))) : :($walk(f, $x, $(rs...))))
        end
    end
    Expr(:block, body..., :(return nothing))
end

# The body of a fold, from `init` and in order: `op(acc, x, rs...)` at a leaf of a known type where
# `direct`, for the reason of `_foreach_body`, and `walk(op, acc, x, rs...)` at every other node.
function _fold_body(walk::Symbol, direct::Bool, @nospecialize(ps::Type),
        @nospecialize(rest::Tuple))
    body = Any[Expr(:meta, :inline)]
    acc = :init
    for n in first(_tree_nodes(ps, rest, :(_no_folded_skip_error())))
        if n isa Expr
            push!(body, n)
        else
            x, rs, leaf, _, _ = n
            next = gensym(:acc)
            call = direct && leaf ? :(op($acc, $x, $(rs...))) :
                   :($walk(op, $acc, $x, $(rs...)))
            push!(body, :($next = $call))
            acc = next
        end
    end
    Expr(:block, body..., :(return $acc))
end

# The body of `mapparameters`: the value of each node, then the tree of these values in the shape of
# `ps`. The value is `f(x, rs...)` at a leaf of a known type, for the reason of `_foreach_body`,
# `Base.map` of the walk across a branch of more than `_WRITE_OUT_MAX_CHILDREN` children, and
# `mapparameters(f, x, rs...)` at every other node. A further set that is `nothing` beside a branch
# raises, since a new tree has nothing to put there; where the walk raises, it calls `f` on no leaf.
function _map_body(@nospecialize(ps::Type), @nospecialize(rest::Tuple))
    nodes, shape = _tree_nodes(ps, rest, :(_no_skipped_set_error()), true)
    shape === nothing && return Expr(:block, Expr(:meta, :inline), last(nodes))
    body = Any[Expr(:meta, :inline)]
    for (k, (x, rs, leaf, _, wide)) in enumerate(nodes)
        value = wide ? :(map(_FixFirst(mapparameters, f), $x, $(rs...))) :
                leaf ? :(f($x, $(rs...))) : :(mapparameters(f, $x, $(rs...)))
        push!(body, :($(_node_name(k)) = $value))
    end
    Expr(:block, body..., :(return $shape))
end

# A generator runs when a walk first meets a tree type, so its helpers are compiled into the package
# image rather than at that first call.
precompile(_foreach_body, (Symbol, Bool, Type, Tuple))
precompile(_fold_body, (Symbol, Bool, Type, Tuple))
precompile(_map_body, (Type, Tuple))

"""
    mapparameters(f, ps)
    mapparameters(f, ps, rest...)

Apply `f` to every leaf of `ps`, returning a parameter set of the same shape.

With further arguments the trees are walked in lockstep and `f` receives one leaf from each, which is
how two parameter sets are combined entrywise. Their keys have to agree.

`f` sees *whole* leaves — a `SymmetricMatrix` arrives as a `SymmetricMatrix`. Use [`mapstorage`](@ref)
to see only the differentiable storage instead.

The result type is inferred, also where `f` runs another `mapparameters`, as an optimizer does that
builds its cache per leaf: the whole tree is one generated body, with one call of `f` per leaf. A
branch of more than 32 children is the exception: it is mapped with `Base.map`, which compiles faster
and whose result type is not inferred. A key or arity mismatch raises before `f` runs on any leaf,
except below such a branch.

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
@generated function mapparameters(f, ps::_Branch, rest::Vararg{Any, N}) where {N}
    _map_body(ps, rest)
end

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

Allocation-free, at any width of branch, any depth of nesting and any number of `rest`, and inside the
function of another walk: the whole tree is one generated body, which indexes the branches in place
rather than taking them apart, so nothing is materialised on the way in.

See [`mapparameters!`](@ref) for the in-place variant that returns its destination.
"""
@generated function foreachparameters(f, ps::_Branch, rest::Vararg{Any, N}) where {N}
    _foreach_body(:foreachparameters, true, ps, rest)
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
into the leaf. Allocation-free on the same terms, except inside the function of another walk that
takes fewer sets, where this walk takes four or more: there it allocates at every leaf.
"""
@inline function mapstorage!(f, dest, srcs::Vararg{Any, N}) where {N}
    _foreach_storage(f, dest, srcs...)
    dest
end

@generated function _foreach_storage(f, ps::_Branch, rest::Vararg{Any, N}) where {N}
    _foreach_body(:_foreach_storage, false, ps, rest)
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

Allocation-free at any width, depth and arity, and inside the function of another walk —
**provided a caller that hands `op` on through a function of its own annotates it `::F where {F}`
there.** Julia does not specialise on a function
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
@generated function foldparameters(op, init, ps::_Branch, rest::Vararg{Any, N}) where {N}
    _fold_body(:foldparameters, true, ps, rest)
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

Allocation-free on the terms of `foldparameters`, except inside the function of another walk that
takes fewer sets, where this fold takes four or more: there it allocates at every leaf.

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
@generated function foldstorage(op, init, ps::_Branch, rest::Vararg{Any, N}) where {N}
    _fold_body(:foldstorage, false, ps, rest)
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

# A further set `x` as a branch of the same kind as the one beside it: keyed beside a keyed branch,
# positional beside a positional one. A generated walk calls it only where it raises.
@inline _like(::NamedTuple, x) = _as_namedtuple(x)
@inline _like(::Tuple, x) = _as_tuple(x)

# The storage of a further set's leaf, where a `nothing` stays a `nothing`. A leaf whose storage is
# one array then hands `f` or `op` that array and the `nothing`; a leaf whose storage is several blocks
# has nothing to pair one `nothing` with, and the branch walk below it raises.
@inline _leaf_storage(::Nothing) = nothing
@inline _leaf_storage(x) = freeparameters(x)

# The walk of `mapstorage` across the children of a branch: the unrolled map across a branch of at
# most `_WRITE_OUT_MAX_CHILDREN` children, and `Base.map` across a wider one.
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
