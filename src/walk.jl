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
# So the across-children walks write the flat body out instead: one `@generated` expansion per branch
# shape, `k` statements reading `getfield(xs, 1) … getfield(xs, k)` at *literal* indices. One
# specialisation instead of `k`, no new tuple types at all, every index inferred at a constant, and
# the same straight-line code the chain used to inline down to. They are `_foreach_zip`, `_fold_zip`
# and `_anynothing` here, `_flatten_children!`, `_unflatten_children` and `_unflatten_children!` in
# `flatten.jl`, the two branch cases of `_layout` in `layout.jl`, `_promote_eltypes` in `leaves.jl`,
# and `_accumulate_named!` and `_matching_named` on the reverse pass in `derivatives.jl`.
#
# A walk that takes *several* sets in lockstep indexes each of them in place too, and that is the same
# point made about the second argument. Taking `values` of a branch to zip it materialises a temporary
# tuple per branch per argument, which is free while the branch stays in registers and is not beyond
# that: `mapparameters!` on a flat 48-child set cost 800 bytes a call that way and 6 144 at 369, and a
# branch of branches three to four times that. So `_foreach_zip` reads every one of its arguments with
# `getfield` at a literal index, exactly as `_flatten_children!` reads the layout, and the arity and the
# keys are checked in the generator where they cost nothing.
#
# **Writing a body out is not free, and one walk here does not.** It costs compilation per branch
# shape, and for a walk that runs once rather than once per iteration that is the wrong trade — see
# `_map_zip`, which hands branches wider than 32 children back to `Base.map` and says why at length.
# The distinction to apply is not "wide or narrow" but "in an inner loop or not".

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
there. This is what lets a gradient tree that is missing the entries of a frozen or non-trainable
layer be walked against the parameters it belongs to, without having to fill the holes in first.

Allocation-free, at any width of branch, any depth of nesting and any number of `rest`: the branches are
indexed in place rather than taken apart, so nothing is materialised on the way in.

See [`mapparameters!`](@ref) for the in-place variant that returns its destination.
"""
@inline function foreachparameters(f, ps::Union{NetworkParameters, NamedTuple},
        rest::Vararg{Any, N}) where {N}
    _foreach_zip(f, freeparameters, _as_namedtuple(ps), map(_as_namedtuple, rest)...)
    nothing
end

@inline function foreachparameters(f, ps::Tuple, rest::Vararg{Any, N}) where {N}
    _foreach_zip(f, freeparameters, ps, map(_as_tuple, rest)...)
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

@inline function _foreach_storage(f, ps::Union{NetworkParameters, NamedTuple},
        rest::Vararg{Any, N}) where {N}
    _foreach_zip(f, _storage_recurse, _as_namedtuple(ps), map(_as_namedtuple, rest)...)
    nothing
end

@inline function _foreach_storage(f, ps::Tuple, rest::Vararg{Any, N}) where {N}
    _foreach_zip(f, _storage_recurse, ps, map(_as_tuple, rest)...)
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
dynamic dispatch: 3 088 bytes a call on a 369-leaf set at arity one and 6 160 at arity two, against
zero with it, identically on Julia 1.11.9 and 1.13.0-rc3. A closure that *captures* a function needs
nothing, a closure being its own type, which is why `(acc, x) -> acc + abs2(f(x))` is the way to fold
a function of each leaf and no second function argument is needed here.

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
@inline foldparameters(op, init, ps::Union{NetworkParameters, NamedTuple},
    rest::Vararg{Any, N}) where {N} = _fold_zip(
    op, freeparameters, init, _as_namedtuple(ps), map(_as_namedtuple, rest)...)

@inline foldparameters(op, init, ps::Tuple, rest::Vararg{Any, N}) where {N} = _fold_zip(
    op, freeparameters, init, ps, map(_as_tuple, rest)...)

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
@inline foldstorage(op, init, ps::Union{NetworkParameters, NamedTuple},
    rest::Vararg{Any, N}) where {N} = _fold_zip(
    op, _storage_recurse, init, _as_namedtuple(ps), map(_as_namedtuple, rest)...)

@inline foldstorage(op, init, ps::Tuple, rest::Vararg{Any, N}) where {N} = _fold_zip(
    op, _storage_recurse, init, ps, map(_as_tuple, rest)...)

@inline function foldstorage(op, init, x, rest::Vararg{Any, N}) where {N}
    s = freeparameters(x)
    s === x && return op(init, x, rest...)
    foldstorage(op, init, s, map(_leaf_storage, rest)...)
end

# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------

# The arity check the `Base.tail` chains used to get for free, by running out of `Tuple{}` methods.
# Raised from the generated bodies, so it is a clear error at specialisation time rather than a
# `MethodError` naming a tuple type nobody wrote.
#
# **This has to be defined before every generated body that calls it, and that is not a matter of
# taste.** A `@generated` function's generator runs in the world age of its own method definition, so
# a helper defined further down the file is invisible to it: the generator raises
# `MethodError: no method matching _children_arity(…)  The applicable method may be too new`. Loading
# a precompiled package hides this, because deserialising the cache gives every method in the module
# one world age -- so it is only seen when the sources are evaluated, which is what
# `--compiled-modules=no` does and what `test/world_age_tests.jl` pins. `src/flatten.jl`'s two callers
# are covered by `walk.jl` being included first.
#
# The same holds for everything else a generator here reaches: `_check_arity`, `_arity_error`,
# `_branch_keys`, `_child_keys_error` and `_child_expr` below, and `_no_folded_skip_error` further
# down. All of them sit above `_foreach_zip` and `_fold_zip`, which are the generated bodies that call
# them, and none may be moved past either.
#
# `rest` is the *types* of the further sets, as a tuple, and the recursion below is over that tuple —
# whose length is the arity of the walk, one or two, never the width of a branch. A `Nothing` among them
# is a set the caller is skipping and has no children to count. Written as a chain and not a loop
# because `_map_zip` calls this at run time rather than from a generator, and `enumerate` over a
# heterogeneous tuple of types costs it an allocation there.
function _children_arity(xs, rest::Tuple)
    n = fieldcount(xs)
    _check_arity(n, rest, 2)
    n
end

@inline _check_arity(::Int, ::Tuple{}, ::Int) = nothing
@inline function _check_arity(n::Int, rest::Tuple, j::Int)
    r = first(rest)
    r === Nothing || fieldcount(r) == n || _arity_error(n, j, fieldcount(r))
    _check_arity(n, Base.tail(rest), j + 1)
end

@noinline _arity_error(n::Int, j::Int, m::Int) = throw(ArgumentError(string(
    "parameter sets walked together must have the same number of children at every level; ",
    "got ", n, " and, in argument ", j, ", ", m, ".")))

# The keys of a branch type, or `nothing` for a branch that has none. A generated body uses it to pair
# the children of two named branches *by key*, and to say so when they cannot be paired.
_branch_keys(::Type{<:NamedTuple{Keys}}) where {Keys} = Keys
_branch_keys(::Type) = nothing

@noinline _child_keys_error(ks, rk, j::Int) = throw(ArgumentError(string(
    "parameter sets have different keys: ", ks, " and, in argument ", j, ", ", rk)))

# One child of one further set, as an expression: the literal `nothing` for a set being skipped, and
# `getfield` at a literal index otherwise. Named branches have their keys checked here, in the
# generator, so the pairing is by key and the check costs nothing at run time — which is the guard
# `mapparameters` pays `_check_keys` for and the in-place walks used to go without.
function _child_expr(ks, r::Type, j::Int, i::Int)
    r === Nothing && return :nothing
    rk = _branch_keys(r)
    # `j + 1` and not `j`: the message counts the parameter sets, of which `xs` is the first
    ks === nothing || rk === nothing || rk === ks || _child_keys_error(ks, rk, j + 1)
    :(getfield(rest[$j], $i))
end

@inline _as_namedtuple(x::NetworkParameters) = params(x)
@inline _as_namedtuple(x::NamedTuple) = x
@inline _as_namedtuple(x::Nothing) = nothing

@inline _as_tuple(x::Tuple) = x
@inline _as_tuple(x::Nothing) = nothing

# The storage of a further set's leaf, where a `nothing` stays a `nothing`. A hole has no storage to
# ask for, and `freeparameters` answers a hole the only way it can — with the leaf protocol's own
# error, telling the caller to define `freeparameters(::Nothing)`, which is the one thing that is not
# the answer. So the hole survives the descent and the level below decides what it means: a leaf whose
# storage is one array hands `op` or `f` that array and the `nothing`, exactly as `mapparameters` and
# `foldparameters` hand over the whole leaf and the `nothing`; a leaf whose storage is *several* blocks
# has nothing to pair one `nothing` with, and the branch walk raises its own error saying so.
@inline _leaf_storage(::Nothing) = nothing
@inline _leaf_storage(x) = freeparameters(x)

# `||` and not `any`, so it still short-circuits: the whole point is not to look at the rest once a
# `nothing` has been found.
@generated function _anynothing(t::Tuple)
    fieldcount(t) == 0 && return :(false)
    tests = [:(getfield(t, $i) === nothing) for i in 1:fieldcount(t)]
    foldr((a, b) -> :($a || $b), tests)
end

# The out-of-place walk, and **the one walk here that hands wide branches back to `Base.map`.**
#
# Every other across-children walk in this package is written out at every width, and this one is not,
# because what they need and what it needs are different things. `_flatten_children!`,
# `_unflatten_children!` and `_foreach_zip` run once per optimizer *iteration* or more, and
# `_unflatten_children` runs once per objective evaluation through the closure a `Gradient` is built
# from; for those, type stability and zero allocation are the property the package exists to provide,
# and they are worth any amount of compilation. `_map_zip` is what `mapparameters` and `mapstorage`
# are, and a consumer calls those once per cache, once per `changebackend`, once per `map_to_cpu` —
# not per iteration.
#
# The cost of writing it out is real and it is paid per (function, branch shape) pair. Measured cold on
# a flat 369-entry set, Julia 1.11.9: `map(zero, ps)` compiles in 0.01 s and a written-out
# `mapparameters(zero, ps)` in 1.51 s. `GeometricOptimizers` reaches six such primitives while building
# one `OptimizerCache`, which took that from 1.57 s to **71.16 s** on the parameter set of GMLDatasets'
# MNIST transformer — a shape a consumer really has.
#
# `Base.map` has no such cliff because past 32 fields it drops to the `Any32` loop, and that fallback
# is exactly what it costs: inference gives up on the result, and it allocates about 30 % more than the
# written-out body (19 824 bytes against 15 424 at 128 children). For a walk that runs once, that buys
# one dynamic dispatch at the call site — the *object* it returns is concretely typed either way, so
# everything downstream of it specialises normally. For a walk in an inner loop it would buy one per
# call, which is why the split is here and not everywhere.
#
# So: written out to 32 children, `map` beyond. 32 and not another number because it is the width
# `Base` itself unrolls a tuple to, so below the threshold nothing is given up and above it nothing is
# gained. A network's layer is far below it; the flat set that made D12 a defect is far above.
#
# (The closure `map` is handed here used to be the objection to it, and is no longer. `map` over a
# closure was reported not to be elided on Julia 1.10, which is why the walks in this package became
# `Base.tail` chains in the first place. On 1.11, 1.12 and 1.13 alike a closure costs nothing against a
# plain function — 6240 bytes against 6224 at 40 children, 19 824 against 19 808 at 128 — so that
# reason expired with the 1.10 floor.)
const _WRITE_OUT_MAX_CHILDREN = 32

# The branch is on `fieldcount` of a *type*, so it folds to a constant and only one arm is compiled.
# It is here rather than inside the `@generated` body below because a generated function may not emit
# a closure — "The function body AST defined by this @generated function is not pure" — and the `map`
# arm needs one.
@inline function _map_zip(recurse, f, xs, rest::Vararg{Any, N}) where {N}
    # A skipped set is the `foreach` family's affair: there is nothing for an out-of-place walk to put
    # in the hole, so the answer is an error and not a guess. `_children_arity` lets a `Nothing` past,
    # because to `_foreach_zip` it means "call nothing here".
    _anynothing(rest) && _no_skipped_set_error()
    _children_arity(typeof(xs), map(typeof, rest))
    if fieldcount(typeof(xs)) > _WRITE_OUT_MAX_CHILDREN
        map((cs...) -> recurse(f, cs...), xs, rest...)
    else
        _map_zip_written(recurse, f, xs, rest...)
    end
end

@noinline _no_skipped_set_error() = throw(ArgumentError(string(
    "`mapparameters` and `mapstorage` build a new set and so have nothing to put where a `nothing` ",
    "stands; use `mapparameters!`, `mapstorage!` or `foreachparameters`, which skip that position.")))

@generated function _map_zip_written(recurse, f, xs, rest...)
    n = fieldcount(xs)
    calls = [Expr(:call, :recurse, :f, :(getfield(xs, $i)),
                  (:(getfield(rest[$j], $i)) for j in 1:length(rest))...) for i in 1:n]
    :(($(calls...),))
end

@generated function _foreach_zip(f, kind, xs, rest...)
    n = _children_arity(xs, rest)
    ks = _branch_keys(xs)
    calls = [Expr(:call, :_foreach_step, :f, :kind, :(getfield(xs, $i)),
                  (_child_expr(ks, rest[j], j, i) for j in 1:length(rest))...) for i in 1:n]
    quote
        $(calls...)
        nothing
    end
end

@inline _foreach_step(f, ::typeof(freeparameters), args::Vararg{
    Any, N}) where {N} = foreachparameters(f, args...)
@inline _foreach_step(f, ::typeof(_storage_recurse), args::Vararg{
    Any, N}) where {N} = _foreach_storage(f, args...)

# The fold across the children of one branch, at every arity. `foldparameters` and `foldstorage` are
# the same walk over the same generated body, and `kind` selects which of the two `_fold_step`
# continues with, exactly as it does for the `foreach` family above.
#
# A *left* fold, and the expansion keeps it one: `op` is the caller's and need not be associative, so
# the nesting is built inside out rather than halved.
#
# Written out for the reason the rest of them are, and this is the walk the point was made on from
# outside. `GeometricOptimizers` wrote three `Base.tail` folds for want of a zipped one here —
# `l2norm`, `solution_scale` and `_dot` — and they cost 26 to 71 s to compile at 369 children on Julia
# 1.12 and 1.13, against 0.65 to 1.47 s for the same code on 1.11.9 (issue #19).
#
# **Written out at every width, where `_map_zip` stops at 32 — and by that walk's own test rather than
# in spite of it.** The question is "in an inner loop or not", and those same three folds are what an
# optimizer computes once per iteration, which puts the fold on the side `_flatten_children!` and
# `_foreach_zip` are on and not the side `mapparameters` is. The sweep prints the price of the
# difference in one row: at 369 bare children on 1.13 `mapparams` reads 0.04 s against this body's
# 0.48 s, because `mapparameters` handed that branch back to `Base.map`. There is no such hand-off to
# be had for a reduction anyway. `Base.map` returns a container of the same shape where a fold returns
# one number, and folding `values(ps)` instead materialises per branch and per further set exactly the
# tuple the walks here were rewritten to stop materialising.
#
# `scripts/wide_branch_cost.jl` sweeps that comparison here, and its `tailfold` control reproduces it:
# on a 369-child branch the chain costs 1.94 s on 1.11.9, 28.55 s on 1.12.7 and 37.86 s on 1.13.0-rc3,
# where this body costs 2.11 s, 0.80 s and 0.47 s. The chain is the cheaper of the two on the compat
# floor — that is not the argument. The argument is that one of them holds still across versions.
#
# **`op` is not annotated `::F where {F}` here, and that was measured rather than assumed.** Every
# method above the leaf only passes `op` along, which is the shape the specialisation heuristic bites
# on — and #19 asks for the annotation on that ground, having measured it costing `_sumsq_leaves` 128
# bytes a leaf downstream. Annotating them changes nothing: a 369-leaf fold reached through a
# `@noinline` caller costs 3 088 bytes at arity one and 6 160 at arity two, with the annotation and
# without it, on 1.11.9 and 1.13.0-rc3 alike. The boxing is the *caller's* — annotating the caller
# takes both figures to zero — so the note belongs in the docstring, where a consumer will read it,
# rather than on signatures where it would look like it were doing something.
@noinline _no_folded_skip_error() = throw(ArgumentError(string(
    "`foldparameters` and `foldstorage` reduce every leaf they are given, so a set that is `nothing` ",
    "would make the result a partial sum without saying so; fill the gaps in, or walk it with ",
    "`foreachparameters` and accumulate into a `Ref`.")))

@generated function _fold_zip(op, kind, acc, xs, rest...)
    n = _children_arity(xs, rest)
    ks = _branch_keys(xs)
    any(r -> r === Nothing, rest) && _no_folded_skip_error()
    expr = :acc
    for i in 1:n
        expr = Expr(:call, :_fold_step, :op, :kind, expr, :(getfield(xs, $i)),
                    (_child_expr(ks, rest[j], j, i) for j in 1:length(rest))...)
    end
    expr
end

@inline _fold_step(op, ::typeof(freeparameters), args::Vararg{
    Any, N}) where {N} = foldparameters(op, args...)
@inline _fold_step(op, ::typeof(_storage_recurse), args::Vararg{
    Any, N}) where {N} = foldstorage(op, args...)

# `_map_zip`'s two arms return the children in different shapes: the written-out body a `Tuple`, which
# has to be keyed, and `Base.map` over a `NamedTuple` a `NamedTuple` that is keyed already. Selecting
# all `k` fields of the second out of itself is a second `k`-wide generated body and a copy for nothing
# — 6 272 bytes of the 57 120 a `mapparameters` costs at 369 children.
@inline _rewrap_children(::NamedTuple{Keys}, children::NamedTuple{Keys}) where {Keys} = children
@inline _rewrap_children(::NamedTuple{Keys}, children) where {Keys} = NamedTuple{Keys}(children)

@inline _check_keys(::Tuple, ::Tuple{}) = nothing
@inline function _check_keys(ks::Tuple, nts::Tuple)
    other = first(nts)
    if other !== nothing && keys(other) != ks
        throw(ArgumentError(string("parameter sets have different keys: ", ks, " and ", keys(other))))
    end
    _check_keys(ks, Base.tail(nts))
end
