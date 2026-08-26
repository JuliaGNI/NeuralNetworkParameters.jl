```@meta
CurrentModule = NeuralNetworkParameters
```

# Walking a parameter set

Nearly everything done to a parameter set has the same shape: recurse into the `NamedTuple`s, do
something at each leaf, put the result back in the same shape. Moving parameters to another device,
mapping them to the host, making them static, the elementwise arithmetic an optimizer cache needs,
writing them to a file — all of it.

Written once over [the leaf protocol](@ref "The leaf protocol"), the recursion never needs to know
which structured types exist, so each of those operations becomes a call rather than its own copy of
the traversal with a method per wrapper type.

```@docs
mapparameters
mapstorage
mapparameters!
mapstorage!
foreachparameters
foldparameters
foldstorage
```

## Whole leaves or their storage

The distinction between [`mapparameters`](@ref) and [`mapstorage`](@ref) is which level the function
sees, and both are needed.

Moving a parameter to another device wants the *whole* leaf: it has to rebuild the wrapper around the
moved array. Halving one wants only the *storage*: for a symmetric matrix that means halving the
``n(n+1)/2`` numbers it keeps, and broadcasting over the dense interface instead would do twice the
work — while for a skew-symmetric or triangular matrix there is no `setindex!` to broadcast through at
all.

```jldoctest walks
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0 5.0],)))
mapparameters(x -> x ./ 2, ps).L1.W

# output

1×2 Matrix{Float64}:
 0.5  1.0
```

## Combining two sets

Given more than one argument the trees are walked in lockstep and the function receives one leaf from
each:

```jldoctest walks
a = NetworkParameters((L = (x = [1.0, 2.0],),))
b = NetworkParameters((L = (x = [10.0, 20.0],),))
mapparameters(+, a, b).L.x

# output

2-element Vector{Float64}:
 11.0
 22.0
```

## Gaps

A branch or leaf that is `nothing` skips that position in the in-place and `foreach` walks. This is the
shape a gradient tree has when a layer was frozen or simply not differentiated, and skipping means such
a tree can be walked against the parameters it belongs to without filling the holes in first:

```jldoctest walks
dest = NetworkParameters((p = [1.0], q = [2.0]))
mapparameters!((d, s) -> (d .+= s), dest, NetworkParameters((p = [10.0], q = nothing)))
(dest.p, dest.q)

# output

([11.0], [2.0])
```

## Reductions

[`foldparameters`](@ref) visits leaves in the order [`flatten`](@ref) writes them, so a fold and a
flattening agree about the order:

```jldoctest walks
foldparameters((n, x) -> n + length(x), 0, ps)

# output

5
```

A fold takes further sets in lockstep as the other walks do, which is how an inner product or a
quadrature norm over a parameter set is computed without flattening it first — one number out, and no
flat vector of either set on the way. The walk itself allocates nothing at any width, depth or arity;
what `op` spends is `op`'s own, and the broadcast below builds a temporary per leaf:

```jldoctest walks
foldparameters((acc, x, y) -> acc + sum(x .* y), 0.0, a, b)

# output

50.0
```

[`foldstorage`](@ref) is the same walk over the [`freeparameters`](@ref) of each leaf, which is the
level `flatten` writes: the pairing of a symmetric matrix is over the ``n(n+1)/2`` numbers it stores,
where reading its dense interface would count every off-diagonal entry twice.

A set that is `nothing` is an error here rather than a skip. A fold reduces every leaf it is given, so
leaving one out would make the answer a partial sum without saying so; walk such a tree with
[`foreachparameters`](@ref) and accumulate into a `Ref` if that is what is wanted. A `nothing` in
place of a single *leaf* is not the same thing and still reaches `op`, which is what lets the caller
decide what a missing leaf contributes.
