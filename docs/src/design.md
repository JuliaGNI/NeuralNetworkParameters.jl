```@meta
CurrentModule = NeuralNetworkParameters
```

# Design notes and measurements

This page gives the reasons behind the internal design of the walks and the layouts, and the
measurements that support them. Each figure names the Julia version and the script or method it
was taken with. A figure is the *first-call* time of a cold process, which is compilation and a
negligible run, unless the row says otherwise. `scripts/wide_branch_cost.jl` and
`scripts/leaf_layout_cost.jl` run each row in a process of its own, because compilation is shared
across a session and a second row would measure less than the first.

The widest branch that a real consumer has is a flat set of 369 leaves: the parameter set of the
MNIST transformer in GMLDatasets.jl.

## The walk across the children of a branch

A walk down the tree is dispatch on `NetworkParameters`, `NamedTuple`, `Tuple` or a leaf. A walk
*across* the children of one branch is one call of a primitive of `src/unrolled.jl`: a map, a left
fold, and a map that carries a state from child to child. Each primitive is a generated body that
reads child `i` with `getfield` at the literal index `i`. A foreach is the map, called for its effect.
A walk binds its own function with the internal `_FixFirst` and not with `Base.Fix1`, because a
primitive passes one child of each set, and `Base.Fix1` takes only one further argument on Julia
1.11.

**Why not a `Base.tail` chain.** A chain makes a new tuple type at every level, so a branch of `k`
children costs `k` specialisations whose argument types are each `O(k)` long. `flatten` on a flat set
took 0.17 s at 32 children, 2.2 s at 64, 17.6 s at 128, and more than ten minutes at 369. Without
`@inline` the chain took 3.2 s at 64 and allocated 187 808 bytes. Julia 1.11, before NNP 0.2.2,
`scripts/wide_branch_cost.jl`.

**Why not a loop over `keys`.** A loop indexes a heterogeneous tuple at a runtime index, which is not
type stable, and costs a dynamic dispatch and an allocation per child.

**Why not `values` of the branch.** `values` materialises a temporary tuple per branch and per set.
`mapparameters!(f, dest, src)` on a flat set of 48 children allocated 800 bytes a call that way, and
6 144 at 369. Julia 1.13, NNP 0.2.2, bytes per call measured from inside a function; the 0.2.2
entry of `CHANGELOG.md` has the table. The primitives read every set in place.

**A fold reached through a caller that does not specialise.** A 369-leaf fold reached through a
`@noinline` caller that does not annotate `op::F where {F}` allocates 3 088 bytes at arity one and
6 160 at arity two, on Julia 1.11.9 and 1.13.0-rc3, and zero with the annotation.
`test/integration/wide_branches.jl` asserts both sides. The walks of this package do not annotate
`op`, because the boxing is the caller's.

**The `Base.tail` folds of GeometricOptimizers.** Three such folds (`l2norm`, `solution_scale` and
`_dot`) took 26 to 71 s to compile at 369 children on Julia 1.12 and 1.13, against 0.65 to 1.47 s on
1.11.9. The `tailfold` column of `scripts/wide_branch_cost.jl` reproduces the effect: on a branch of
369 children a chain takes 1.94 s on 1.11.9, 28.55 s on 1.12.7 and 37.86 s on 1.13.0-rc3, where the
written-out fold takes 2.11, 0.80 and 0.47 s.

## Two exceptions to the written-out walks

**`mapparameters` and `mapstorage` hand a branch wider than 32 children to `Base.map`.** They run once
per cache or per device move, not per iteration, and their compile time is paid per pair of function
and branch shape. On a flat set of 369, `map(zero, ps)` compiled in 0.01 s and a written-out
`mapparameters(zero, ps)` in 1.51 s; GeometricOptimizers reaches six such walks while it builds one
`OptimizerCache`, which took that from 1.57 s to 71.16 s. Julia 1.11.9, `scripts/wide_branch_cost.jl`.
`Base.map` past 32 fields drops to its `Any32` loop: its result type is not inferred, and it
allocates about 30 % more (19 824 bytes against 15 424 at 128 children). A closure costs nothing
against a plain function on Julia 1.11, 1.12 and 1.13 (6 240 bytes against 6 224 at 40 children,
19 824 against 19 808 at 128). 32 is the width to which `Base` unrolls a tuple itself.

**The layout carries the offset from child to child.** `_layout` of a branch is the map with a state:
each child is laid out after the offset of the one before it, and gives the offset after itself. Two
other forms do not work:

- A left fold whose accumulator holds the children laid out so far gives one tuple type per child,
  each longer than the last, so inference pays for a type that grows with every child.
- The lengths of the children first, then each child laid out at its own offset, passes the offsets
  as a tuple of `Int`s beside the branch. For a structured leaf whose storage has more blocks than its
  branch has children, that tuple grows from one level of the recursion to the next, and inference
  widens the layout type ("failed to optimize due to recursion" in `test/quality/jet.jl`).

## The layout type

`LeafLayout` keeps no prototype, so a layout type does not carry the array type of each leaf. With
the prototype as a type parameter, `parameterlayout` on a flat set of 369 leaves took 1.35 s as a
bare `NamedTuple` and 13.40 s inside a `NetworkParameters`; at 768 leaves 2.77 s against 87.77 s. The
type tree of the wrapped 369-leaf layout had 1 849 nodes, against 742 without the parameter. Without
it the two shapes cost the same: 1.05 s wrapped against 1.04 s bare at 369, 3.01 s against 3.07 s at
768, and 0.84 s against 0.85 s on a 16 × 24 set of 384 leaves. Julia 1.11, 1.12 and 1.13 agree: 1.05,
1.13 and 1.03 s at 369 wrapped. The whole 369-wrapped path of `parameterlayout`, `flatten` and
`unflatten` went from 21.90 s to 8.64 s. NNP 0.2.3, `scripts/leaf_layout_cost.jl` and
`scripts/wide_branch_cost.jl`, whose shared row agrees to 0.1 s.

A leaf is terminal or wrapped by dispatch on the type of its storage. An `===` test in one method
body gave each leaf layout a three-way union, which inference then carries through every child of a
branch.
A leaf whose `freeparameters` returns a distinct object of its own type is terminal by this rule;
such a leaf has no working layout either way, since the descent into its storage does not end.

## The walks on NNP 0.5.1

The rows below compare NNP 0.5.0 with the primitives of this page, on a flat set of 369 leaves of
2 × 2 inside a `NetworkParameters`. Julia 1.13.1, `julia --project=. scripts/wide_branch_cost.jl
--in-process wrapped 369`, one cold process per row, load average about 5.

| | layout | map(zero) | mapparams | fold | foldzip | flatten | unflatten | allocations |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| NNP 0.5.0 | 0.85 s | 0.01 s | 0.04 s | 0.16 s | 0.27 s | 1.67 s | 15.41 s | 0/0/0/0 |
| the primitives | 1.03 s | 0.01 s | 0.01 s | 0.04 s | 0.30 s | 2.08 s | 15.66 s | 0/0/0/0 |

The allocations are those of `parameter_eltype`, `flatten!`, `unflatten!` and `mapparameters!`.
