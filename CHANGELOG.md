# Changelog

Notable changes to `NeuralNetworkParameters` are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-08-25

**A branch with many children is usable.** 0.2.1 finished making the walks type stable and allocation
free; this release makes them *compilable* on a branch wider than a couple of dozen children, which
they were not.

### Changed

- **Julia 1.11 is now the minimum**, up from 1.10. The walks below are `@generated` bodies, and Julia
  1.10 cannot inline one: neither `Expr(:meta, :inline)` in the generated body, nor `@inline` on the
  declaration, nor `@inline` at the call site has any effect there — the call stays out of line. The
  bodies are allocation-free on 1.10 too; what `flatten!` and `unflatten!` need from the inlining is
  that it lets the temporaries at a branch boundary be elided. Without it, on 1.10, against 0.2.1:

  | set | `flatten!` | `unflatten!` | `unflatten` |
  |---|---|---|---|
  | one with structured leaves | 0 → 192 | 0 → 608 | 576 → 1616 |
  | a two-level `NamedTuple` | 0 → 0 | 0 → 176 | 960 → 1408 |

  That is the guarantee 0.2.1 exists to provide, so on 1.10 the two cannot both be had. Julia 1.11 and
  later elide those temporaries without inlining, and 1.11, 1.12 and 1.13 were each checked at zero on
  a nested set, a set with structured leaves, a wide flat branch and a wide branch of branches.

### Fixed

- **Every walk across the children of one branch cost one specialisation per child.** The walk *down*
  the tree is ordinary dispatch and was never the problem. The walk *across* was an `@inline`d
  `Base.tail` chain, and `Base.tail` yields a new tuple type at every level — so a branch of `k`
  children produced `k` specialisations whose argument types were each `O(k)` long, and inference on
  that grew as `k³`.

  Nine walks were written this way and all nine are now written out instead, as a `@generated` flat
  body of `k` statements reading `getfield(xs, 1) … getfield(xs, k)` at *literal* indices: one
  specialisation per branch shape rather than `k`, no new tuple types at all, every index inferred at a
  constant, and the same straight-line code the chain used to inline down to. They are
  `_flatten_children!`, `_unflatten_children`, `_unflatten_children!` (`src/flatten.jl`), `_map_zip`,
  `_foreach_zip`, `_fold_children`, `_anynothing` (`src/walk.jl`), `_promote_eltypes`
  (`src/leaves.jl`) and `_layout_children` (`src/layout.jl`).

  First-call time, which is compilation plus a negligible run, on a flat `NamedTuple` of `k` leaves
  (Julia 1.13, `scripts/wide_branch_cost.jl`):

  | children | `flatten`, 0.2.1 | `flatten`, 0.2.2 | `mapparameters(zero, ·)` cold, 0.2.1 | 0.2.2 |
  |---|---|---|---|---|
  | 32 | 0.17 s | 0.09 s | 0.07 s | 0.00 s |
  | 64 | 2.16 s | 0.23 s | 1.32 s | 0.00 s |
  | 128 | 17.57 s | **0.56 s** | 7.39 s | 0.00 s |
  | 369 | not run to completion | **2.05 s** | not run to completion | 0.00 s |

  `map(zero, ·)` is unchanged throughout at 0.01 s, and is in the harness as the floor: it is what
  `Base` costs on the same branch, and what a consumer that wrote its own walk over `map` to get around
  this was paying instead.

  369 is the width that made this worth fixing rather than noting: it is the parameter set of the MNIST
  transformer in `scripts/geometric_optimizers/mnist.jl` of
  [GMLDatasets.jl](https://github.com/JuliaGNI/GMLDatasets.jl) — 3·7·16 attention projections, 2·16
  ResNet parameters and one classification weight, in one flat `NamedTuple` — written against this
  package alone. `flatten` is on the path of every `GeometricOptimizers.GradientAutodiff`, so that
  network could not be optimized through this package at all.

  Two answers that look obvious are both wrong, and the harness above is what says so. **Dropping the
  `@inline`** leaves the `k` specialisations exactly where they were: the same 64-child set went from
  2.16 s to 3.22 s, *and* began allocating, because the inlining is what had kept the per-leaf
  `copyto!` statically dispatched. **A `for` loop over the children**, which is what `Base.map` falls
  back to past 32 fields, indexes a heterogeneous tuple at a runtime `i` — type-unstable, so a dynamic
  dispatch and a boxed allocation per child, and the end of the property 0.2.1 exists to provide.

  Reported from the review of [GeometricOptimizers#68], which hit this when widening its elementwise
  primitives to a parameter container and had to write its own `map`-based walk to get around it.

- **`flatten!` and `unflatten!` allocated on a branch of more than about forty children**, against the
  guarantee in their own docstrings. Two temporary tuples were materialised per branch — `values(ps)`
  and `values(l.children)` — which is free while a branch is narrow enough to keep in registers and is
  not beyond that. Measured on the same flat sets on Julia 1.13, bytes per call from inside a
  function:

  | children | 32 | 48 | 64 | 128 | 369 |
  |---|---|---|---|---|---|
  | 0.2.1 | 0 | 81 488 | 187 808 | not reached | not reached |
  | 0.2.2 | 0 | **0** | **0** | **0** | **0** |

  The walks index the branch in place with `getfield` now and take no `values` at all, so nothing is
  materialised on the way in. This was found while fixing the entry above and is a separate defect:
  narrowing the branch made it disappear, which is why the docstrings' claim had held up in testing.

- **The reverse pass had the same two defects**, on the walk that turns a cotangent in the shape of the
  parameters back into a flat gradient. `_accumulate_named!` was a `Base.tail` chain over
  `values(l.children)` and `keys(l.children)`; it is written out now, with the branch's keys spliced in
  as *literal* `Symbol`s — which is what keeps `_cotangent_get`'s `haskey` a compile-time question, the
  property the chain had been relying on constant propagation for. `_matching_named` goes with it.

  Bytes per pullback call, on the same flat sets and again on Julia 1.13:

  | children | 16 | 32 | 48 | 64 | 128 | 369 |
  |---|---|---|---|---|---|---|
  | 0.2.1 | 320 | 576 | 70 592 | 161 504 | not reached | not reached |
  | 0.2.2 | 320 | 576 | **848** | **1 120** | **2 112** | **6 208** |

  Identical up to 32 children and 144× better at 64, which is the signature of the same cliff: below it
  the temporary tuples stay in registers, above it they do not.

  The two *positional* walks — `_accumulate_positional!` and `_matching_positional` — are deliberately
  left as chains, and the comment beside them now says why: their length is the number of blocks of a
  single leaf, two for a `StiefelLieAlgHorMatrix` and one for a Grassmann lift, so they are never wide
  the way a branch of layers is.

### Added

- **`ParameterSet`**, exported: `Union{NetworkParameters, NamedTuple}`, the type to dispatch on when a
  method takes *the parameters* and does not care which of the two forms it was handed.

  Every package in this ecosystem was spelling that union out inline — `AbstractNeuralNetworks` at
  eight sites, `GeometricMachineLearning` at fifteen, `GeometricOptimizers` at sixteen — and
  `SymbolicNeuralNetworks` had named it `EquationSet` for the equation sets that share the shape. One
  name means a reader meets the same type in all of them.

  Its docstring states the two things it is not. It is **not** `isparametertree`, which is also true of
  a `Tuple`, because a `Tuple` is a branch the walks recurse into — what `freeparameters` returns for a
  multi-block leaf — and never a set of parameters handed in whole. And it puts no bound on the element
  type or the depth, so it is **not** `GeometricOptimizers.ParameterContainer{T}`, which additionally
  requires the `NamedTuple` half to be flat and element-type-homogeneous.

- `test/wide_branch_tests.jl`, which walks a 369-child branch through `flatten`/`unflatten`,
  `mapparameters` at both arities, `mapparameters!`, `foreachparameters` with and without the `nothing`
  skip, `foldparameters` and `parameter_eltype`, and pins the in-place forms at zero allocations there.
  It also covers 48, either side of the 32 fields `Base` unrolls a tuple up to.

  A 48-wide branch whose children are themselves branches is covered as well, and separately, because
  the two shapes fail differently: a flat branch's children are arrays, which are heap objects
  already, so a walk that reaches one has nothing to keep off the heap, while a branch of branches has
  to be taken apart in place at every level. The width is a consumer's; the nesting is a network's.

  It asserts the properties and not the timings — a wall-clock bound would be a flake on a loaded
  machine — so the regression test is that the file *completes*, which before this release it did not.
  It costs the suite about 45 s on Julia 1.13 and about 1 m 45 s on 1.11, all of it compilation at the
  wide width, and that is the honest price of covering the width a consumer actually has.

- `scripts/wide_branch_cost.jl`, the harness behind both tables above.

[GeometricOptimizers#68]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/68

## [0.2.1] — 2026-08-25

### Fixed

- **The out-of-place `unflatten` walked its children with `map` over a closure.** It was the one walk
  in the package written as `map(c -> unflatten(c, v), values(l.children))` rather than as the
  `Base.tail` recursion `src/walk.jl` states the house rule to be, and which `flatten!`, `unflatten!`
  and `mapparameters` all follow. Both are equally inferable — nothing here was ever type unstable —
  but `map` leaves the closure over the flat vector to be elided, and not every version elides it
  ([#13]).

  Bytes per call, measured from inside a function on the three-leaf set of the `flatten` docstring, and
  per leaf on a flat set of four identical ones:

  | | 1.10 | 1.11 | 1.12 | 1.13 |
  |---|---|---|---|---|
  | three-leaf set, before | 720 | 240 | 240 | 240 |
  | three-leaf set, after | **400** | **112** | 240 | 240 |
  | per leaf, before | 176 | 96 | 96 | 96 |
  | per leaf, after | 176 | **48** | 96 | 96 |

  Julia 1.11 is where it tells: the recursion halves what every leaf costs, from two heap allocations
  to one. On 1.10 the saving depends on the shape of the set rather than on its size — the closure is
  elided for some trees and not others — and 1.12 and 1.13 elide it throughout, so there the change is
  neutral. It is neutral or better everywhere; no set measured costs more than it did.

  The same rewrite takes the walk off `Base`'s `Any32` fallback, which `map` drops to past 32 children
  and which returns a tuple with no concrete type — so a parameter set with more than 32 layers, or a
  layer with more than 32 entries, no longer unflattens through a type-unstable path. This holds for
  the `AbstractMatrix` overload set as well, which splits a Jacobian by parameter block.

  `SymbolicNeuralNetworks` splits the result of a generated function into the shape of a parameter set
  through this function, once per call, which is how the cost came to be noticed
  ([SymbolicNeuralNetworks#55](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/55)).
- **The `unflatten` rrule dispatched dynamically once per child, on every reverse pass.** Its branch
  walks read `for k in keys(l.children)` and indexed `l.children[k]`: a runtime `Symbol` into a
  heterogeneous `NamedTuple` gives back the union of the branch's child layout types, so nothing about
  the recursion was known at compile time. The cost climbed with the number of layers and with the
  depth of the tree, which is exactly what a gradient step pays most often. The walks are now the same
  `Base.tail` recursion as everything else ([#13]).

  Four leaves, one to a layer, against a gradient vector that is itself 128 bytes:

  | | 1.10 | 1.11 | 1.12 | 1.13 |
  |---|---|---|---|---|
  | before | 1664 | 1664 | 1664 | 1664 |
  | after | **128** | **128** | **128** | **128** |

  The pullback now allocates its answer and, on three of the four versions, nothing else. Grouping the
  same leaves into layers no longer costs anything at all, where it used to multiply the bill by five.
- `_matching_storage`, which narrows a cotangent to the storage of a multi-block structured leaf, built
  its positional case with `ntuple` over a *runtime* length. `Base` stops inferring that past ten, so a
  leaf whose `freeparameters` are more than ten blocks fell to a `Tuple` with no concrete type. Both of
  its cases are recursions now ([#13]).
- `parameterrange`, `length(::ParameterLayout)`, `_reshape_leaf` and the `parameter_eltype` family are
  marked `@inline` alongside, for the same reason the in-place walks are. None of these moved a
  measurement; they are consistency with the rule the walks follow, not a claimed effect ([#13]).

## [0.2.0] — 2026-08-24

### Changed

- **`NetworkParameters` carries the element type of its leaves as its first type parameter**, so that
  `T` binds in a method signature: `f(ps::NetworkParameters{T}) where {T}`, or a `Union` with
  `AbstractVector{T}`. `GeometricOptimizers` takes the element type from the *type* of the solution it
  is handed — eleven sites spell it `where {T, VT<:OptimizerSolution{T}}`, across every cache and
  state constructor, both `Optimizer` constructors and the `BFGSState` `update!` methods — and a
  parameter set could not join that union while it carried no such parameter ([#12]).

  The element type is derived by `parameter_eltype` at construction rather than chosen. Naming it, as
  `NetworkParameters{T}(nt)`, asserts it and raises if the leaves say otherwise, so a set whose type
  disagrees with what it holds has no inhabitants. Deriving it costs nothing: it folds to a constant
  and construction allocates as it did.

  It is a *promotion*, not a guarantee of uniformity — unlike `GeometricOptimizers`'
  `ArrayNamedTuple{T}`, `NetworkParameters{T}` does not license assuming every leaf is a `T` — and a
  set with nothing to promote reports `Union{}`.
- **The element type comes first, so the keys no longer fit in the braces.** Build a set from its keys
  and values with `NetworkParameters(NamedTuple{keys}(vals))`, which is what the removed
  `NetworkParameters{Keys}(values)` did anyway. The old spelling raises an `ArgumentError` saying so
  rather than a `MethodError` about a conversion nobody asked for ([#12]).
- **`parameter_eltype` is total, where `freeparameters` is not.** Every parameter set now runs its
  constructor through this function, including sets that hold no numbers at all: a gradient tree with
  `nothing` where an untouched layer's entries would be, or `SymbolicNeuralNetworks` wrapping
  generated functions in one. A leaf this package cannot read numbers out of contributes nothing to
  the promotion, exactly as an empty set does, and both report `Union{}` — where a leaf with no
  protocol used to raise. A set that cannot be flattened is still a set with an element type; the
  protocol error comes from `parameterlayout`, which is where it decides something ([#12]).

  The recursion follows `freeparameters` for an `AbstractArray` leaf, which every structured leaf in
  the ecosystem is; asking an arbitrary type where its storage is would mean raising, which this
  function must not do. A leaf that keeps its numbers behind another interface opts in with one method
  more, `parameter_eltype(x::MyLeaf) = parameter_eltype(freeparameters(x))`, and `flatten` says so if
  it is missing rather than guessing.
- `parameter_eltype(ps::NetworkParameters)` reads the type parameter instead of walking the leaves, so
  `flatten(ps)` and the `flatten` rrule are cheaper than they were.

### Added

- `NetworkParameters{T, Keys, ValueTypes}(nt)` and `NetworkParameters{T}(nt)` are written out
  ([#12]), since the derived element type makes the one that computes it an inner constructor and
  suppresses the defaults. The three-parameter form is not optional: `ChainRulesCore.construct`
  calls it from `+(::P, ::Tangent{P})`, which is how a parameter set and a cotangent add.
- `flatten(ps)` raises an `ArgumentError` naming `parameter_eltype` when the leaves promote to
  `Union{}` and the layout nevertheless found numbers to copy — a leaf keeping its storage behind an
  interface that is not an array's, with no `parameter_eltype` method of its own. It used to be a
  `Vector{Union{}}` failing on the first copy. An empty set still flattens to a `Vector{Union{}}`
  ([#12]).
- `Base.hash(::NetworkParameters)`, forwarding to the wrapped `NamedTuple` as `isequal` does. The
  default takes the type in, and the element type is part of that, so a `Float32` set and a
  `Float64` set holding the same numbers — `isequal`, and `==` — used to hash apart and behave as
  two different `Dict` keys ([#12]).

### Unchanged

- **The on-disk HDF5 format.** The element type is derived on read, so nothing new is written and a
  file from 0.1 loads with the element type its leaves imply.
- `Base.eltype` is still deliberately undefined on `NetworkParameters`. The type forwards the
  `NamedTuple` interface, for which `eltype` means the type of what iteration yields — a layer's
  `NamedTuple` — rather than the numeric element type.
- Equality, which compares the wrapped `NamedTuple`s: a `Float32` set and a `Float64` set holding the
  same numbers are still `==`, and have different element types.

## [0.1.1] — 2026-08-23

### Fixed

- The error raised for an unregistered type in a file `GeometricMachineLearning` wrote no longer
  offers a remedy that cannot work. Such a group carries a `gml_type` attribute and holds the type's
  fields under their own names, with no `storage` for `rebuild` to take, so registering the
  type with `register_parameter_type!` is the only way into it — the message now says that instead of
  suggesting a prototype ([#11]).
- Passing a prototype for one of those leaves raised `KeyError: "storage"` from inside the reader.
  `load` now checks for the storage before reaching for it and raises an `ArgumentError` naming the
  tagged type and the registration it needs ([#11]).

### Added

- The branch of the HDF5 reader that loads those files is covered by the test suite, including what
  it hands a registered reconstructor: the group's fields, the same object in both argument
  positions, reachable by name only. One registered name serves both that layout and the current
  one, which is the contract `GeometricOptimizers` is written against ([#11]).
- Coverage for a structured leaf that keeps everything in its storage and has no
  `parameter_metadata` — a manifold element, in the types this package is written for. The writer
  leaves the metadata group out of the file and the reader hands the reconstructor an empty
  `NamedTuple`; both halves of that agreement are now pinned ([#11]).

### Changed

- `docs/src/hdf5.md` says what the corrected errors say: that a `gml_type` file has to go through the
  registry, that the group's fields reach the reconstructor in both positions, and that the metadata
  group is written only when the type has metadata to record ([#11]).

## [0.1.0] — 2026-08-21

The initial release: the `NetworkParameters` container and its flat `FlatParameters` form, the
`freeparameters`/`rebuild` leaf protocol that structured parameter types extend, the tree walks,
`flatten`/`unflatten` with their derivative rules, and HDF5 storage through a package extension.

[#11]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/11
[#12]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/12
[#13]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/13
[0.2.1]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.1
[0.2.0]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.0
[0.1.1]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.1
[0.1.0]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.0
