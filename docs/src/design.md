```@meta
CurrentModule = NeuralNetworkParameters
```

# Design notes and measurements

This page gives the reasons behind the internal design of the walks and the layouts, and the
measurements that support them. Each figure names the Julia version and the script it was taken
with. A figure is the *first-call* time of a cold process, which is compilation and a
negligible run, unless the row says otherwise. `scripts/wide_branch_cost.jl` and
`scripts/leaf_layout_cost.jl` run each row in a process of its own, because compilation is shared
across a session and a second row would measure less than the first.

The widest branch that a real consumer has is a flat set of 369 leaves: the parameter set of the
MNIST transformer in GMLDatasets.jl.

## The walk across the children of a branch

The layout and the flattening walk down the tree by dispatch on `NetworkParameters`,
`NamedTuple`, `Tuple` or a leaf. They walk *across* the children of one branch with one call of a
primitive of `src/unrolled.jl`: a map, a left fold, and a map that carries a state from child to
child. Each primitive is a generated body that reads child `i` with `getfield` at the literal index
`i`. A foreach is the map, called for its effect. Such a walk binds its own function with
`Base.Fix1`, and a primitive passes it one child of each set. The maps, the in-place walks and the
folds take another form, described in the next section.

**Why not a `Base.tail` chain.** A chain makes a new tuple type at every level, so a branch of `k`
children costs `k` specialisations whose argument types are each `O(k)` long. The 0.2.2 entry of
`CHANGELOG.md` gives the cold first-call times of the chain of NNP 0.2.1 against the written-out
walks, on Julia 1.11.9 and 1.13, from `scripts/wide_branch_cost.jl`. Without `@inline` the chain is
slower still, and it allocates, because the chain is what kept the per-leaf `copyto!` statically
dispatched.

**Why not a loop over `keys`.** A loop indexes a heterogeneous tuple at a runtime index, which is not
type stable, and costs a dynamic dispatch and an allocation per child.

**Why not `values` of the branch.** `values` materialises a temporary tuple per branch and per set,
so `mapparameters!(f, dest, src)` allocates on every call that way. The primitives read every set in
place.

**A fold reached through a caller that does not specialise.** A 369-leaf fold reached through a
`@noinline` caller that does not annotate `op::F where {F}` allocates, at arity one and at arity
two, and allocates nothing with the annotation. `test/integration/wide_branches.jl` asserts both
sides. The walks of this package do not annotate `op`, because the boxing is the caller's.

**The `Base.tail` folds of GeometricOptimizers.** Three such folds (`l2norm`, `solution_scale` and
`_dot`) took much longer to compile at 369 children on Julia 1.12 and 1.13 than on 1.11. The
`tailfold` column of `scripts/wide_branch_cost.jl` reproduces the effect: on a branch of
369 children a chain takes 1.94 s on 1.11.9, 28.55 s on 1.12.7 and 37.86 s on 1.13.0-rc3, where the
written-out fold takes 2.11, 0.80 and 0.47 s.

## The maps, the in-place walks and the folds are one body per tree

`mapparameters`, `mapstorage`, `foreachparameters`, `mapparameters!`, `mapstorage!`,
`foldparameters` and `foldstorage` read the whole tree off the types of their arguments when they are
generated, and each is one body with one call per leaf. The key and arity checks are the
generator's, as in the primitives. The maps check the keys of a branch before the number of its
children, so two keyed branches of different widths are reported by their keys; the other walks and
the primitives check the number first.

**Why not a walk that recurses through its own methods.** An optimizer that steps each leaf with its
own method runs a walk in the function of another one. A walk that recurses through its own methods
then puts the same method on the inference stack twice, the second time with a function argument
nested one level deeper, such as `Base.Fix1{typeof(foreachparameters), Step{…}}`. Inference reads
that as recursion that might not end and widens the inner call, so the inner walk dispatches
dynamically and boxes its function at every leaf. NNP 0.5.1 walks that way. With an inner walk on a
one-leaf set of each of the 3 leaves of a set, its `foreachparameters` in `foreachparameters` and
`mapstorage!` in `mapstorage!` allocate 96 B per step, or 192 and 240 B with five captured `Float32`
and `Float64` coefficients, and each of its folds in itself 144 B. The generated walks allocate
nothing in all these cases. Julia 1.13.1, `julia --project=. scripts/nested_walk_cost.jl nested`.

An optimizer that builds its cache and its state per leaf runs a map in the function of another map,
and a map that recurses through its own methods returns a type that inference widens the same way.
On NNP 0.5.2, `mapparameters` whose function runs `mapparameters` on a set of one leaf, of two leaves,
or of one leaf at arity two, does not infer a concrete result type in any of the three cases, in
`Float32` or `Float64`; the generated `mapparameters` infers all six. `mapstorage` in `mapstorage`
on a set of two leaves does not infer on NNP 0.5.2 either, and infers in both element types as a
generated body. Julia 1.13.2, `scripts/nested_walk_cost.jl nestedmap`. The `TrainingOptimizer` of a
`CompositeMethod` in GeometricOptimizers is such a caller: on NNP 0.5.2 its type is not inferred, and
a caller that builds it and steps in one function allocates 64 B per step.

**Why `f` at a leaf and not the leaf method.** A generated body that calls the leaf method of
`foreachparameters` or of `foldparameters` at each leaf puts that method on the inference stack
twice as well, both times called from the same generated method. Where the inner walk takes more
sets than the outer one, and at least four, inference shortens the longer argument list of the
second call to a `Vararg` and does not specialise it on its function. The per-leaf `Adam` step of
GeometricOptimizers walks six sets inside a walk of five, and then boxes its function at every leaf,
32 B in `Float32` and 48 B in `Float64`. So these two walks call `f` or `op` at a leaf directly, and
`test/walk.jl` runs a walk of four sets inside a walk of two for each. The storage walks call their
leaf method, which decides at run time with `freeparameters` whether a leaf is its own storage, so
`mapstorage!` and `foldstorage` of four or more sets inside a walk of fewer sets still allocate at
every leaf: on the 3 leaves of `scripts/nested_walk_cost.jl wide`, 192 B and 48 B per step
(`KNOWN_ISSUES.md` K11).

**What it costs.** The table compares NNP 0.5.1 with the generated walks: the first call in a cold
process, on a set of 3 leaves of depth 2, 64 leaves of depth 4, and 369 leaves of depth 1, at arity
two. Julia 1.13.1, `scripts/nested_walk_cost.jl <walk> <shape>`, one process per row, load average
about 10.

| | 3 leaves | 64 leaves | 369 leaves |
|:--|--:|--:|--:|
| `mapparameters!`, NNP 0.5.1 | 0.06 s | 0.09 s | 1.11 s |
| `mapparameters!`, generated | 0.03 s | 0.16 s | 1.24 s |
| `mapstorage!`, NNP 0.5.1 | 0.07 s | 0.09 s | 1.09 s |
| `mapstorage!`, generated | 0.04 s | 0.16 s | 1.21 s |
| `foldparameters`, NNP 0.5.1 | 0.06 s | 0.06 s | 0.28 s |
| `foldparameters`, generated | 0.05 s | 0.06 s | 0.11 s |
| `foldstorage`, NNP 0.5.1 | 0.07 s | 0.07 s | 0.32 s |
| `foldstorage`, generated | 0.05 s | 0.07 s | 0.15 s |

The in-place walks compile about 0.1 s slower on the two larger sets, and faster on the small one.
The folds compile as fast or faster. One call takes the same time within the resolution of
the clock, and allocates nothing on either side. The generators' helpers are precompiled, since a
generator runs at the first call of a walk on a tree type.

The first call of `mapparameters` at arity two, with a function that costs nothing to compile, on
the same three sets: the median of three cold processes per cell, NNP 0.5.2 and the generated map
alternating. Julia 1.13.2, `scripts/nested_walk_cost.jl map <shape>`, load average about 15.

| | 3 leaves | 64 leaves | 369 leaves |
|:--|--:|--:|--:|
| `mapparameters`, NNP 0.5.2 | 0.08 s | 0.11 s | 0.21 s |
| `mapparameters`, generated | 0.04 s | 0.06 s | 0.24 s |

The generated map compiles faster on the two smaller sets and about as fast on the flat set, whose
branch of 369 children both map across with `Base.map`. A later call allocates the same on both.

## Two walks that take another form

**`mapparameters` and `mapstorage` hand a branch wider than 32 children to `Base.map`.** They run once
per cache or per device move, not per iteration, and their compile time is paid per pair of function
and branch shape. On a flat set of 369, `map(zero, ps)` compiled in 0.01 s and a written-out
`mapparameters(zero, ps)` in 1.51 s. Julia 1.11.9, `scripts/wide_branch_cost.jl`. The generated
`mapparameters` maps across such a branch with `Base.map` too: written out in its body, with
`_WRITE_OUT_MAX_CHILDREN` raised above 369, its first call of `zero` on that set takes 10.4 s, against
0.03 s with `Base.map`. Julia 1.13.2, two cold processes each. GeometricOptimizers
reaches six such walks while it builds one `OptimizerCache`, so a written-out walk raises the compile
time of that cache. `Base.map` past 32 fields drops to its `Any32` loop: its result type is
not inferred, and it allocates. A closure passed to it costs about as
much as a plain function. 32 is the width to which `Base` unrolls a tuple itself.

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
`unflatten` took 21.90 s with the parameter and 8.64 s without it. Julia 1.11 where no other
version is named; NNP 0.2.2 for the figures with the prototype and NNP 0.2.3 for those without;
`scripts/leaf_layout_cost.jl` and `scripts/wide_branch_cost.jl`, whose shared row agrees to 0.1 s.

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
