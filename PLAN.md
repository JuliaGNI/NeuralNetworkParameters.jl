# NeuralNetworkParameters.jl — plan

This is a plan, not a record. It says what is still to be done, what a change must not break, and
how to check both. What has already been done lives in `CHANGELOG.md` and in the issue trackers,
which carry it in more detail than this file ever did.

Verified against the sibling checkouts on 2026-08-28. Paths written `ANN/…`, `SNN/…`, `GML/…`,
`GO/…`, `NLI/…`, `SS/…`, `GB/…` are relative to the corresponding checkout.

## Goals

1. Provide a structured type for storing neural network parameters across the GML ecosystem:
   `AbstractNeuralNetworks`, `SymbolicNeuralNetworks`, `GeometricOptimizers`,
   `GeometricMachineLearning`, `GMLDatasets`, `NonlinearIntegrators`.
2. Remove all type piracy in the GML ecosystem.
3. Remove code duplication across the GML ecosystem.

**All three are met.** Goal 1 is in use by every package below. Goal 3 has no remainder but one
exception kept on purpose (§4.1). Goal 2 closed with `GeometricOptimizers` 0.6.1, and closed on a
*property* rather than a count: `GO/test/aqua_tests.jl` runs `Aqua.test_piracies`, and
`Aqua.Piracy.hunt` reports zero for all five of `NeuralNetworkParameters`, `GeometricBase`,
`SimpleSolvers`, `GeometricOptimizers` and `AbstractNeuralNetworks`.

§4 is therefore about what the closing of goal 2 exposed rather than about goal 2 itself.

## Specification

The type wraps `NamedTuple`s of `NamedTuple`s of `AbstractArray`s and has a secondary flat
representation as an `AbstractArray`. Accessors and operations are type-stable and allocation-free.
The implementation is performance-oriented; §6 says how that is measured here.

## Where the ecosystem stands

Every package below has its wave of the migration merged; nothing is waiting on a branch.

| package | version | branch |
|---|---|---|
| `NeuralNetworkParameters` | 0.2.5 | `main` |
| `GeometricBase` | 0.14.10 | `main` |
| `SimpleSolvers` | 0.13.2 | `main` |
| `AbstractNeuralNetworks` | 0.7.3 | `main` |
| `SymbolicNeuralNetworks` | 0.7.1 | `main` — untracked `GMLPLAN.md`, kept on purpose, see §4.2 |
| `GeometricOptimizers` | 0.6.1 | `main` |
| `GeometricMachineLearning` | 0.6.1 | `main` |
| `NonlinearIntegrators` | 0.4.2 | `main` |
| `GMLDatasets` | 0.1.0 | `keep-the-mnist-weights-out-of-the-generated-docs` |

**`GeometricBase` 0.14.10, `SimpleSolvers` 0.13.2 and `GeometricOptimizers` 0.6.1 have to be
registered in that order**, because the third requires the first two by version: the methods it used
to pirate are only in those releases. See §4.1 for what that costs downstream.

**Julia 1.11 is the floor across all of these except `GeometricBase`**, which keeps 1.10 and can:
its new extension only loads beside a package that already requires 1.11, and a 1.10 caller who does
not load `NeuralNetworkParameters` never sees it. Every workaround the move retired was a *1.10*
workaround, in three places: `SimpleSolvers`' `HAS_PREALLOCATED_GETRF`, whose fallback allocated the
pivot vector on every factorization in the default linear solver; `SymbolicNeuralNetworks`'
`Base.tail` walk in `unflatten_batch`; and a set of 1.10-only allocation ceilings in `SNN` and
`NLI`. 1.10 also cannot inline a `@generated` body, which §3 needs. Nothing wants 1.12 as a floor;
it stays in every CI matrix instead.

## 1. What the package provides

For the networks defined in `AbstractNeuralNetworks`, `GeometricMachineLearning` and
`SymbolicNeuralNetworks`:

- **(a)** a structured parameter type — `NamedTuple`s of `NamedTuple`s of `AbstractArray`s following
  the architecture;
- **(b)** a flat `AbstractArray` form suited to computing derivatives;
- conversions, such that a derivative taken with respect to the flat form comes back structured;
- HDF5 read and write.

Dependencies must stay lean and must not pull a chain. The package adds exactly one,
`ChainRulesCore`, whose own dependencies are `Compat` and `LinearAlgebra`. **Any proposal that adds
a second dependency to the main package has to argue against this line.**

| file | contents |
|---|---|
| `src/parameters.jl` | `NetworkParameters`, `params`, `ParameterSet`, the `NamedTuple` forwarding |
| `src/leaves.jl` | `freeparameters`, `rebuild`, `parameter_metadata`, `parameter_eltype`, `isterminal`, `isparametertree` |
| `src/walk.jl` | `mapparameters`, `mapstorage`, the in-place and `foreach` forms, `foldparameters`, `foldstorage`, at any arity |
| `src/layout.jl` | `ParameterLayout` and its five concrete cases, `parameterlayout`, `parameterrange`, `flatlength` |
| `src/flatten.jl` | `flatten`, `flatten!`, `unflatten`, `unflatten!`, the Jacobian split |
| `src/flat_parameters.jl` | `FlatParameters`, the `AbstractVector` interface, layout-preserving `similar` and broadcasting |
| `src/io.jl` | the `h5save`/`h5load`/`save`/`load` generics and the type registry |
| `src/derivatives.jl` | `ChainRulesCore` rules for both conversions |
| `ext/HDF5Ext.jl`, `ext/ZygoteRulesExt.jl` | the storage methods; `pullback` on a `NetworkParameters` |

Two packages now carry a method on this package's types, and both are extensions on a weak
dependency, so neither costs anything to a caller who does not load this one:

| package | file | methods |
|---|---|---|
| `GeometricBase` 0.14.10 | `GB/ext/NeuralNetworkParametersExt.jl` | `L2norm(::ParameterSet)`, from which the generic `l2norm(x) = sqrt(L2norm(x))` follows |
| `SimpleSolvers` 0.13.2 | `SS/ext/SimpleSolversNeuralNetworkParametersExt.jl` | `GradientAutodiff(F, ::ParameterSet)`, `GradientFunction(F, ∇F!, ::ParameterSet)`, `alloc_h(::ParameterSet)` |

## 2. Why the package is needed

Two arguments still decide something. The rest — the inventory of traversals that were written many
times over and have since been deleted — is done, and each package's changelog records its own half.

**A parameter set has to be a type somebody owns.** A bare `NamedTuple` container means every method
taking a parameter set dispatches on a type nobody owns, which is global: any package that loads the
definer silently gets the new behaviour. `NetworkParameters` is that type, and it is what made the
eight methods of `GeometricOptimizers` issue
[#16](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16) movable — five of them upstream
to the packages that own their generics, two behind a `GeometricOptimizers`-owned
`RiemannianGradient <: SimpleSolvers.Gradient`, one narrowed onto that package's own `Hessian` types.
The issue is closed and `GO/test/aqua_tests.jl` is the ledger now.

**`ParameterHandling` cannot express these parameters.** See §5. It is why GO's remaining work was a
removal rather than a wrapping, and why it stays removed.

## 3. Design invariants

Properties a change has to preserve. Each is asserted somewhere, and where it is asserted is named,
because **a guarantee holds where it is asserted and nowhere else** — the allocation defect below
was fixed on the two walks the tests measured, then found again on three that were not measured, and
again on a tenth that had no column in the sweep.

- **`ParameterSet` is the dispatch type; `NetworkParameters` is the container.** `ParameterSet` is
  keyed only, and deliberately not `isparametertree`'s domain, which also admits a `Tuple`: a
  `Tuple` is a branch the walks recurse into, but never a set of parameters handed in whole.
  `test/parameters_tests.jl` asserts exactly that difference rather than the happy path.
- **`ParameterSet` bounds neither element type nor depth.** That is the whole difference from
  `GeometricOptimizers.ParameterContainer{T}`, and both have to exist: across GO's alias `T` means a
  flatness-and-uniformity guarantee for its `NamedTuple` member and a promotion over leaves at any
  depth for the container, so it is the right type exactly where `T` also appears elsewhere in the
  signature and the wrong type everywhere else.
- **The leaf protocol is the entire extension surface.** `freeparameters(x)` gives the
  differentiable storage; `rebuild(prototype, data)` puts a leaf back. `freeparameters` may return
  an array, a scalar, or a `Tuple`/`NamedTuple` for a multi-block type, and the recursion continues
  into it. Nothing here holds a list of structured types.
- **A walk over a set recurses through the *leaf's* generic, not through its own.** `GeometricBase`'s
  `L2norm(::ParameterSet)` folds `abs2(l2norm(leafᵢ))` and not `L2norm(leafᵢ)`, which is what leaves
  the leaf's own notion of norm to the package that defines the leaf: `GO`'s
  `l2norm(::AbstractLieAlgHorMatrix)` is over the lift's free parameters, where the generic
  `L2norm(::AbstractArray)` would read the dense ``n \times n`` interface and count a skew block's
  entries twice. A method moved upstream that stops calling back into the leaf's generic has silently
  changed what it computes. `GO/test/optimizer_status_tests.jl` is where that is caught.
- **`rebuild` takes a prototype, not a type.** Non-differentiable fields (`n`, `N`) come across for
  free, `data` may have a different element type, and the concrete type cannot drift — which is the
  bug class that turned a `GrassmannManifold` into a `StiefelManifold` on every round trip before
  the protocol existed.
- **A layout is a value**, not a closure: storable in an optimizer cache, comparable, inferable.
  This is the concrete difference from `ParameterHandling`.
- **Copies are range `copyto!`s.** No element is ever indexed individually, so GPU arrays work
  unchanged (`src/flatten.jl:43-44`).
- **`unflatten` must be able to return a different element type from the parameters**
  (`ForwardDiff.Dual`). A `Float64` buffer cannot be shared with a `Dual` view, so the flat form is
  copied rather than aliased, with allocation-free in-place variants for inner loops.
- **Every across-children walk is a `@generated` body that indexes the branch with `getfield` at
  literal indices.** Never `values(·)`: it materialises a temporary tuple, free while the branch
  stays in registers and not beyond, and past 32 children `Base` stops unrolling. Ten walks have had
  this treatment. A new one starts there rather than arriving at it — and note that the *outer*
  branch's width is what decides the cost, so a 48 × 2 set (a child per layer, which is what a deep
  network is) pays where a 16 × 24 one does not.
- **A `@generated` body may only call helpers defined above it in the file.** A generator runs in
  the world age of its own method definition. Precompilation hides a violation by giving every
  method in a module one world age, so `test/world_age_tests.jl` runs the walks in a subprocess
  under `--compiled-modules=no`, which is the only way to ask.
- **The HDF5 writer records key order** as a group attribute. Older files — `ANN`'s attribute-less
  groups, `GML`'s `gml_type` tagging — still load, falling back to a regex guess that is guarded so
  it leaves the order alone unless every key matches.
- **Loading has two paths**, because a file has no prototype to rebuild against: pass one explicitly
  (`load(NetworkParameters, h5, prototype)`, no registration needed) or let the owning package call
  `register_parameter_type!`.
- **`save`/`load`/`h5save`/`h5load` are not exported**, matching `AbstractNeuralNetworks`, since
  those names collide with most of the ecosystem.
- **A package cannot export a type sharing its own name** — the module binding wins at the `using`
  site. Hence `NetworkParameters`, and hence one type with one name: `AbstractNeuralNetworks` 0.7
  removed `NeuralNetworkParameters` outright rather than aliasing it, and
  `ANN/test/parameters_seam_tests.jl` pins that the name is not even defined. 0.7.3 applied the same
  rule the other way and renamed its own `ArrayNamedTuple` — network inputs and outputs — to
  `NamedTupleOfArrays`, since `GeometricOptimizers.ArrayNamedTuple` is a set of *parameters* and the
  two shared a name by coincidence.

## 4. Open work

### 4.1 The `SimpleSolvers` 0.12 pin, upstream of everything here

**This is the one thing blocking the ecosystem, and it is outside all nine checkouts.**
`GeometricIntegratorsBase` 0.6.3 declares `SimpleSolvers = "0.12.1 - 0.12"`, and
`GeometricIntegrators` 0.17 inherits it. `GeometricOptimizers` 0.6.1 requires `SimpleSolvers` 0.13.2,
because that is where three of the five upstreamed methods live. So:

| consumer | held at | by |
|---|---|---|
| `NonlinearIntegrators` | `GeometricOptimizers` 0.5.0, `SimpleSolvers` 0.12.2, `NeuralNetworkParameters` 0.2.1 | its own `GeometricIntegratorsBase = "0.6.3"` dependency |
| `GeometricMachineLearning`'s **test** environment | `GeometricOptimizers` 0.6.0, `SimpleSolvers` 0.12.2 | its test extra `GeometricIntegrators = "0.16 - 0.17"` |

Neither of these is new — both were already pinned back before this wave — but 0.6.0 could still
*resolve* against `SimpleSolvers` 0.12 and 0.6.1 cannot. `GeometricMachineLearning`'s own package
dependencies are unaffected; only its test environment blocks, and only through that one extra.

**The fix is a compat widening in `GeometricIntegratorsBase` to `SimpleSolvers = "0.13"`, followed by
one in `GeometricIntegrators`.** Nothing in this repository or its siblings can do it. Until then:

- `GML/test/reduced_system.jl` is the one test file that needs `GeometricIntegrators`. With it set
  aside, `GeometricMachineLearning`'s suite passes in full against `GeometricOptimizers` 0.6.1 —
  every optimizer test, every structured-parameter test, the HDF5 round trips and the docstring
  examples. That is how this wave was verified.
- `NonlinearIntegrators` cannot be tested against it at all, and does not need to be: nothing there
  reaches any of the eight methods. Its `ShallowNet` flattens to a plain `Vector` before `Optimizer`
  sees it, and `GradientAutodiff(F, ::NamedTuple)` — the one upstreamed method it does name, in a
  comment — is the same function with the same signature, reached through the same
  `GeometricOptimizers` export.

### 4.2 Goal 3 — duplication

Nothing is left to consolidate. One item is kept on purpose, and it is listed as a decision so that
it is not mistaken for outstanding work — three earlier revisions of this file filled this slot with
work that was already done.

`GML`'s `_tree_optim_step!` (`GML/src/optimizers/optimizer.jl:293`) stays hand-written, and the
comment above it at `:270-292` says why. `foreachparameters` cannot serve it for two reasons, both
load-bearing: the recursion is keyed on the *cache* tree and stops where the cache stops, so a
layer's `NamedTuple` arrives whole rather than being descended into; and `λY` is *broadcast* rather
than zipped, one `GlobalSection` standing in for a whole subtree. That is a different walk, not a
copy of this one.

`SNN`'s untracked `GMLPLAN.md` is kept as an untracked file, deliberately. The work it describes —
rewriting `SNN/docs/src/guide/training.md` against `GeometricOptimizers` instead of
`GeometricMachineLearning`, so that `SNN/docs/Project.toml` resolves again — is **not** done, and
`SNN/docs/Project.toml` still carries `GeometricMachineLearning = "0.6"` and a comment saying the
environment does not resolve. Note that its premise has moved on: the blocker it names is a
`GeometricMachineLearning` release tracking the 0.2 container, and the blocker today is §4.1.

### 4.3 Housekeeping

- **`GeometricOptimizers` has 139 method ambiguities**, which is what turning `Aqua` on turned up
  and what keeps `GO/test/aqua_tests.jl` at `test_piracies` rather than `test_all`. **35 of them have
  both methods owned by that package**, which is the tractable half; the rest are between its
  structured matrices (`SymmetricMatrix`, `SkewSymMatrix`, the triangulars, `StiefelProjection`) and
  `LinearAlgebra`/`ArrayLayouts` methods on `AbstractMatrix` — `mul!`, `vcat`, `hcat`. None was
  introduced by the de-piracy work.

  Four of the owned ones were **reachable**, all in `_copyto!`, and 0.6.1 closes them; see that
  changelog. The lesson for the rest is that the report is not a work list — a reported pair may
  intersect only at a shape nothing can build. The way to triage one is
  `typeintersect` on the two signatures, then an attempt to construct a witness:
  three pairs in the `_copyto!` family survive triage as unreachable and are documented in place
  rather than closed.

- **`GeometricBase`'s extension is asserted in `GeometricOptimizers`, not in `GeometricBase`.** Its
  test environment would have to resolve `NeuralNetworkParameters`, whose floor is Julia 1.11, while
  `GeometricBase` still supports 1.10 and tests on it. `GO/test/aqua_tests.jl` carries the
  `which`-assertion instead, where both are hard dependencies. If `GeometricBase` ever moves to 1.11
  the assertion should move with it — that is the only reason to.
- **`GeometricOptimizers` 0.6.1 changes what `gradient(opt)` returns** for a parameter set: a
  `RiemannianGradient` wrapping the `GradientAutodiff` rather than the `GradientAutodiff` itself. A
  downstream `isa` check on the inner type has to reach through `.gradient`. `GML` does not do this —
  it builds its own `_GMLGradient` and calls `GeometricOptimizers.update!` directly, never going
  through `Optimizer` — but it is the one visible behaviour change in the release.

### 4.3.1 Should `GeometricOptimizers.ArrayNamedTuple` exist?

Measured rather than argued, because it decides §4.1's wider question — `ParameterSet` was introduced
as a step towards cleaner dispatch, not as an end state, and its `NamedTuple` half is what keeps it
wide.

**It is the cause of all four reachable ambiguities above.** With
`ParameterContainer{T} = NetworkParameters{T}` and the four disambiguations deleted, the owned
ambiguity count falls from **35 to 26** and the `_copyto!` family from **11 to 1** — the section
methods stop overlapping because `(::NetworkParameters, ::GlobalSectionNamedTuple)` is then strictly
more specific than `(::NetworkParameters, ::NamedTuple)`, and the mixed-parameter pairs stop existing
because the shape does. So the disambiguations are a patch on the alias, not on the design.

**The cost of removing it is 13 of `GO`'s 37 test files**, which fail on `Optimizer`,
`OptimizerCache`, `GradientState`, `AdamState` and `_zero` — every entry point that takes a solution.
That is `GO`'s own public commitment that a bare `NamedTuple` is a parameter set, and
`GO/test/named_tuple_parameters.jl` exists to pin it.

**What that commitment is actually worth downstream, measured:**

| consumer | what it hands `GO` | could it hand a `NetworkParameters`? |
|---|---|---|
| `GML` | one **layer** `NamedTuple` per layer, to `OptimizerCache`/`update!` (`GML/src/optimizers/optimizer.jl:70-91`, `:293`) | not by wrapping — `GML`'s own note at `:60-69` says the end state is one cache for the *whole* network, which needs `_GMLGradient` to take a `NetworkParameters` and `_tree_optim_step!` to go. That changes behaviour (one `GlobalSection` tree and one `Q` across every layer) and is its own release. |
| `GMLDatasets` | a **flat 369-key** whole set (`scripts/geometric_optimizers/mnist.jl:127`, `mnist_metal_short.jl:275`) | yes, by wrapping: `NetworkParameters` forwards `keys`, `values`, `ps[:sym]`, `ps.field` and `flatlength`, so `regroup`, `F` and `∇F!` need no change. These are scripts, not library code. |

Note that **neither reaches `l2norm`**: `GML`'s per-layer path never builds an `OptimizerStatus` (0
hits over 15 `optimization_step!` calls across `Adam`, `MomentumMethod` and `GradientMethod`), and
`GMLDatasets`' loop drives `solver_step!` rather than `solve!`, which is the same (0 hits). Earlier
revisions of this file cited both as the reason the `NamedTuple` half had to stay for `l2norm`. They
are the reason it has to stay for the *elementwise primitives*, which is a different claim.

**So the order is:** convert `GMLDatasets`' two scripts (cheap, no library impact) → decide whether
`GO` drops the `ArrayNamedTuple` half of `ParameterContainer`, which is a breaking `GO` release and
the real gate → then `ParameterSet` can narrow in this package, and the ~35 `::ParameterSet`
signatures across `GO`, `ANN`, `GML`, `SNN` and the two extensions follow. `GML`'s
`_tree_optim_step!` removal is orthogonal to all of it.
### 4.4 Performance verification across Julia 1.11, 1.12 and 1.13

**Done for 0.2.5, on 2026-08-28.** A/B of `NeuralNetworkParameters` 0.2.4 against 0.2.5 with
everything else held fixed — `GeometricOptimizers` 0.6.0 (a clean worktree at the release commit, so
that this measures 0.2.5 and not the 0.6.1 work), `GeometricBase` 0.14.9, `SimpleSolvers` 0.13.1 —
one Julia at a time, on an Apple M4 Max. The toggle is the change and not two different inputs.

**Nothing regresses, and the run-time side does not move at all.**
`GO/scripts/optimizer_allocations.jl` is **byte-for-byte identical** between the two on all three
versions, which is the expected shape: 0.2.5's fix was a `@generated` promotion body, so it is
compile time and nothing else.

`solve!`, 20 iterations, bytes allocated — identical under 0.2.4 and 0.2.5:

| | Julia 1.11 | Julia 1.12 | Julia 1.13 |
|---|---|---|---|
| `BFGS` `Vector` | 22 536 | 22 520 | 22 600 |
| `BFGS` `Manifold` | 1 056 760 | 1 055 432 | 1 055 448 |
| `BFGS` `NamedTuple` | 1 559 832 | 1 558 456 | 1 558 472 |
| `BFGS` container | 1 566 072 | 1 564 696 | 1 564 712 |
| `DFP` container | 1 556 456 | 1 555 080 | 1 555 096 |

The compile-time side improves where 0.2.5 said it would and nowhere else. First-call seconds, 0.2.4
→ 0.2.5:

| harness, row | 1.11 | 1.12 | 1.13 |
|---|---|---|---|
| `leaf_layout_cost.jl`, 369 flat, bare | 1.04 → 0.97 | 1.07 → 1.03 | 1.06 → 0.99 |
| `leaf_layout_cost.jl`, 369 flat, wrapped | 1.04 → 0.95 | 1.08 → 1.08 | 1.07 → 0.99 |
| `leaf_layout_cost.jl`, 16 × 24 nested, bare | 0.88 → 0.83 | 0.63 → 0.62 | 0.59 → 0.54 |
| `walk_compile_cost.jl`, `parameterlayout`, flat bare | 1.03 → 1.03 | 1.15 → 1.03 | 1.06 → 1.01 |
| `walk_compile_cost.jl`, `parameterlayout`, nested wrapped | 0.87 → 0.88 | 0.67 → 0.62 | 0.58 → 0.54 |
| `walk_compile_cost.jl`, `NetworkParameters(·)`, nested bare | 0.19 → 0.18 | 0.13 → 0.08 | 0.19 → 0.11 |

`wide_branch_cost.jl` is a wash at every width and both shapes on all three versions, and every
allocation column reads **0** — including 0.2.5's new `parameter_eltype` column, which is the one the
release added. `adam_cache_attribution.jl` is a wash too, every row within 3 %.

The two `layout` columns agree, which is the reading §6 asks for: 0.97/0.95 on 1.11, 1.03/1.08 on
1.12, 0.99/0.99 on 1.13. And `GO`
[#73](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/73)'s baseline holds — the tree
column stays at 0 B, which `GO/test/flat_buffer_allocations.jl` asserts and which passed in the full
`GO` run.

**One methodology defect fell out of this and is fixed.** A row of `leaf_layout_cost.jl` reported
**-1.4 s**, where two re-runs read 0.53. `first_call` used `time()`, the wall clock, which steps when
the system adjusts it. All five harnesses across the two repositories now use `time_ns()`: a negative
reading is at least obvious, and a small positive step in the other direction would not have been.
§6 records the rule.

**The rule stands for the next release.** Every release after 0.2.5 takes its predecessor as the
baseline on the same terms, reported as three columns, and small regressions on 1.11 are acceptable
if 1.12/1.13 improve.

## 5. Why not an existing package

**ParameterHandling.jl** fails on four counts, each one something `GeometricOptimizers` hit in
practice before dropping it:

1. Its `flatten(::Type{T<:Real}, x)` methods cover the Base types, so extending it for `Float32`
   fidelity *is* piracy.
2. `unflatten` is a chain of closures: not type stable, not storable, not comparable.
3. `flatten(x)` defaults to `Float64`, silently promoting a `Float32` network.
4. No GPU method — a `CuVector` falls through to the element-mapping `AbstractVector` path.

It also cannot round-trip a `VectorStorageMatrix` at all, which GO's own docstring spells out
(`GO/src/special_matrices/vector_storage_matrix.jl:18-20`): the `AbstractMatrix` method reshapes the
flat vector back to ``n \times n``, and ``n(n\pm1)/2`` numbers do not reshape to that. And it pulls
`IterTools`, `LogExpFunctions`, `InverseFunctions` and — oddly — `Test` as hard dependencies.

**ComponentArrays.jl**: leaves come back as `ReshapedArray`/`SubArray` views, which cannot represent
the structured types, and there is nowhere for their non-array fields (`n::Int`, `N::Int`). Adds
`ArrayInterface` and `StaticArrayInterface`.

**Optimisers.jl `destructure`**: lean, and it handles parameter tying, but it drags in the
optimiser-rule machinery `GeometricOptimizers` exists to replace, and `Functors.fmap` would need
teaching the structured types anyway — the same extension work plus a dependency.

## 6. How to measure here

```bash
julia --project -e 'using Pkg; Pkg.test()'
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'

julia --project=. scripts/wide_branch_cost.jl    # the walks, both shapes, six widths
julia --project=. scripts/leaf_layout_cost.jl    # the layout type's own cost
```

Rules, each of which cost a wrong committed figure to learn:

- **One process per row, cold.** A sweep in one session reads what the previous row compiled.
  `wide_branch_cost.jl` forks per (shape, width); `--in-process` exists for running one row alone,
  which is also how to check the fan-out is really forking — a row run alone and the same row inside
  the sweep have to read the same.
- **Time the first call through `Base.invokelatest`.** Compiling the harness function infers through
  a direct call, so the inference being timed is spent before the clock starts.
- **Time it on `time_ns()` and not on `time()`.** `time()` is the wall clock and steps when the
  system adjusts it; §4.4's sweep produced a reported **-1.4 s** that way. `time_ns()` is monotonic,
  so a row is the elapsed time or it is nothing.
- **Measure the shape a consumer holds** — bare `NamedTuple` *and* inside a `NetworkParameters`. The
  two entry points reach `parameterlayout` through different methods. The two `layout` columns
  agreeing (about 1.0 s at 369 children on every Julia) is the reading; a cliff re-appearing in the
  wrapped column is the regression to act on.
- **A `Vararg` walk needs a zero-argument closure.** `@allocated f(a, b)` lowers to
  `Base.allocated(f, a, b)`, whose own splat allocates, so the walk's own figure is invisible
  otherwise.
- **Assert `@allocated` as a bound, not an equality.** It reports the process-wide counter over its
  window, and Windows CI jitters tens of bytes in both directions. Pick a bound that separates the
  jitter from the defect by orders of magnitude.
- **Two columns earn their keep by making a difference visible, not by explaining it.** A gap
  between two measurements is where to look, not what to conclude.

The suite is about 45 s on Julia 1.13 and 1 m 45 s on 1.11, nearly all of it compiling the 369-child
case — the width of GMLDatasets' MNIST transformer, which is the width a consumer has rather than
the width that is convenient. It asserts properties and not timings, because a wall-clock bound
would flake on a loaded machine; the regression test is that `test/wide_branch_tests.jl`
*completes*.

Beyond round trips it covers: `Float32` fidelity; zero allocation for the in-place walks measured
from inside a function, at every width, depth and arity; a structured leaf and a two-block leaf
whose types survive the round trip; scalar, empty and tuple leaves; `Dual`-valued unflattening;
agreement between `ForwardDiff` on the flat form and `Zygote` on the structured one; a
structurally-zero gradient block and a set the reverse pass never touched; the `nothing`-branch
skip; HDF5 key ordering, both load paths, and files in the two older formats.

**Two durable measurements**, so they need not be re-derived:

- **Flattening is cheap.** 1.3M parameters, five 512-wide dense layers, `Float64`: `flatten` 0.144
  ms, `flatten!` into a preallocated buffer 0.133 ms, `unflatten` 0.149 ms — against 0.946 ms for
  one forward pass at batch 32. A round trip is about 10 % of a single forward pass.
- **Reshaped views keep the BLAS fast path**, so the argument against shared storage is element type
  and not speed: `reshape(view(v, r), m, n)` is a `StridedArray` and `mul!` runs at 0.089 ms either
  way. The decisive constraint is the `Dual` case in §3.

## 7. Revising this file

The failure mode on record, five times over, is a status claim that was true when it was written.
Three specific traps:

- **Check the tree, not the pull request.** GO's #68 and #69 bodies describe the state *before* the
  commits inside them; three entries were recorded as open on that basis after they had been fixed.
- **A design claim is worth what the last consumer that tried to use it found.** "One implementation
  covers all of them" was true in intent and false in practice for two years of releases, and the
  package that noticed was the one that had to write its own walk instead.
- **A count is not a property.** This file listed four piracy sites, and a fifth it was unsure about,
  where `Aqua.Piracy.hunt` found eight. The list was not merely short — GO's own audit had it wrong
  in *both* directions at once, recording two as open after they were fixed. Where a tool can decide
  the question, cite the tool and delete the list; where none can, expect the list to be wrong.

So: before this file is next revised, `grep` every `file:line` it cites, `gh issue view` every issue
number, and check the version table against `Project.toml` and `git branch --show-current` in all
nine checkouts. Anything that reads as finished comes out.
