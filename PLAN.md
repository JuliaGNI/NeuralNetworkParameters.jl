# NeuralNetworkParameters.jl — analysis and plan

Analysis last revised 2026-08-26, against these working trees. The whole family moves in one wave and
only this package's dependency, `ChainRulesCore`, is outside it. Two rows have landed on `main` since
the wave began — this package's 0.2.3 and `GeometricOptimizers`' 0.6.0 — and the rest are still
branches, so read the third column before quoting a version as released.

| package | version | branch |
|---|---|---|
| `NeuralNetworkParameters` | 0.2.3 | this repository, `main` |
| `GeometricBase` | 0.14.9 | `l2norm-over-any-array-and-julia-1.11` |
| `SimpleSolvers` | 0.13.1 | `retire-the-1.10-getrf-fallback` |
| `AbstractNeuralNetworks` | 0.7.2 | `adopt-parameterset` |
| `SymbolicNeuralNetworks` | 0.7.1 | `adopt-parameterset` |
| `GeometricOptimizers` | 0.6.0 | `main` (#68 and #69 merged) |
| `GeometricMachineLearning` | 0.6.1 | `adopt-parameterset` |
| `NonlinearIntegrators` | 0.4.2 | `julia-1.11-and-geometricoptimizers-0.6` |
| `GMLDatasets` | 0.1.0 | `repair-the-mnist-scripts-against-geometricoptimizers-0.5` |

Paths written `ANN/…`, `SNN/…`, `GML/…`, `GO/…`, `NLI/…` are relative to the corresponding sibling
checkout.

**Julia 1.11 is the floor across all nine.** Every workaround the move retires was a *1.10*
workaround, and they were in three places: `SimpleSolvers`' `HAS_PREALLOCATED_GETRF`, whose fallback
allocated the pivot vector on every factorization in the default linear solver (3360 bytes at
``n = 384``, measured on 1.10.11); `SymbolicNeuralNetworks`' `Base.tail` walk in `unflatten_batch`,
which existed for a 1.10-only saving and had this package's D12 in exchange; and a set of 1.10-only
allocation ceilings in `SNN` and `NLI`. Nothing wanted 1.12, and `GeometricOptimizers`' open issue D1
is a 140× compile-time cliff *on* 1.12 that 1.13 does not have — so 1.12 would have been the wrong
floor to choose. It stays in every CI matrix instead.

**Status.** Phase 1 is released and registered in General. Phase 2 is complete: this package is where
the parameter container lives, and `AbstractNeuralNetworks` consumes it. **Phase 3 is done** —
`GeometricOptimizers` 0.5.0 supplies the leaf protocol for its structured matrices *and* dropped
`ParameterHandling`, and its 0.6.0 branch takes a `NetworkParameters` through the whole optimizer;
`GeometricMachineLearning` 0.6.0 handed the HDF5 traversal over. `SymbolicNeuralNetworks` adopted the
layout type, which §8 had not asked it to.

Two things happened in 0.2.2 that no earlier draft of this document anticipated, and both came from
*outside* — from the review of `GeometricOptimizers` #68 and #69 rather than from this plan:

- **D12, the walks were unusable on a wide branch**, which is a defect of this package that the
  consolidation argument in §2 depended on not existing. It is fixed; see §4.
- **`ParameterSet`**, one name for `Union{NetworkParameters, NamedTuple}`, which every consumer had
  been spelling out for itself. See §6 and §8.

The lesson §3 draws — that a status claim here is worth what the last check against the sibling
checkouts was worth — now has a companion: a *design* claim is worth what the last consumer that tried
to use it found. §2's "written once against the leaf protocol, one implementation covers all of them"
was true in intent and false in practice for two years' worth of releases, and the package that noticed
was the one that had to write its own walk instead.

---

## 1. Goal

For the networks defined in `AbstractNeuralNetworks`, `GeometricMachineLearning` and
`SymbolicNeuralNetworks`, provide:

- **(a)** a structured parameter type — `NamedTuple`s of `NamedTuple`s of `AbstractArray`s following the
  architecture;
- **(b)** a flat `AbstractArray` form suited to computing derivatives;
- conversions, such that a derivative taken with respect to the flat form can be returned structured;
- HDF5 read and write.

Additional dependencies must be lean and must not pull a long chain. The package adds exactly one,
`ChainRulesCore`, whose own dependencies are `Compat` and `LinearAlgebra`.

## 2. Why the package is needed

The parameter type used to live *downstream*, in `ANN/src/parameters.jl`, and the rest of this
functionality was implemented separately in `GeometricOptimizers` and `GeometricMachineLearning`. Three
things drove the consolidation. The first two are now partly demonstrated rather than argued, so each
records what has actually come of it.

**The same traversal was written many times over.** Recurse into the `NamedTuple`s, do something at
each leaf, put it back: that is `flatten`/`unflatten`, `h5save`/`h5load`, `changebackend`, `map_to_cpu`,
`_statify`, the optimizer's elementwise primitives, and `_eltype`/`apply_toNT` — each re-declaring a
method per structured type, in four packages. Written once against the leaf protocol, one
implementation covers all of them and needs to know none of the types. **What that has come to, checked
against the working trees rather than remembered:** ANN's copies are gone, its `changebackend` and
`_statify` each one `mapparameters` call; GML's `h5save` and its "natural sort" are gone, its
`map_to_cpu` is one `mapstorage` call, and its `apply_toNT` and `_eltype` are gone with them; SNN's
`FlatSlice` is gone; GO's `apply_toNT` and its `ParameterHandling` flattening went in 0.5.0 and its
`_mapleaves` in 0.6.0.

**One is left, and it is left on purpose.** GML's `_tree_optim_step!` is still hand-written, and
`GML/src/optimizers/optimizer.jl:270-292` now gives two reasons that `foreachparameters` cannot serve:
the recursion is keyed on the *cache* tree and stops where the cache stops, so a layer's `NamedTuple`
arrives whole rather than being descended into; and `λY` is *broadcast* rather than zipped, one
`GlobalSection` standing in for a whole subtree. That is a different walk, not a copy of this one.
Earlier revisions of this section listed four more items here that had already been done.

**A parameter set needs to be a type somebody owns.** Most of the type piracy `GeometricOptimizers`
issue #16 tracks is not about flattening: the container is a bare `NamedTuple` (aliased
`ArrayNamedTuple`), so `outer!` (`GO/…/bfgs_cache.jl:90`), `(grad::Gradient)(::ArrayNamedTuple)`
(`GO/…/named_tuple_wrapper.jl:97`), `Base.copyto!` (`:141`) and two sites in
`GO/src/optimizers/optimizer_status.jl` (`:163`, `:185`) all dispatch on types nobody owns. Its
comments ask for *"a wrapper struct"*. `NetworkParameters` is that struct, and it exists — but GO has
not adopted it yet, so these five sites still stand. Downstream of it, `NetworkParameters` is already
the container in `AbstractNeuralNetworks` 0.7.0, `GeometricMachineLearning` 0.6.0,
`SymbolicNeuralNetworks` 0.6.0 and `NonlinearIntegrators` 0.4.0.

**`ParameterHandling` cannot express these parameters.** See §7. Still true, and still the reason GO's
remaining work is a removal rather than a wrapping.

## 3. What changed over the course of the analysis

Recorded because each of these invalidated an earlier draft of this document, and because the same
thing keeps happening.

**The GML/GeometricOptimizers type fork is resolved.** GO 0.4.0 made the manifold geometry public API;
GML#239 deleted GML's own copies of eleven types and `go_bridges.jl` with them — about 2500 lines, all
of `src/manifolds/`, and all of `src/arrays/` bar `poisson_tensor.jl`. GML now imports them from GO, and
`GML/src/arrays/gml_extensions.jl` is 43 lines holding only `add!`. Earlier drafts of this plan had an
"eight duplicate walks across four packages" inventory and a plan to collapse `go_bridges.jl`; both are
obsolete, and §2 above is the reduced version.

**GeometricOptimizers already has the free-storage accessor, and it is `Base.parent`.** GO 0.4.0 added
the `VectorStorageMatrix` alias (`GO/src/special_matrices/vector_storage_matrix.jl:35`) for the four
types keeping their free parameters in one vector, and writes each optimizer primitive once over it.
`Base.parent` is defined across the family:

| type | `parent` | file |
|---|---|---|
| `SkewSymMatrix` | `A.S` | `GO/src/special_matrices/skew_symmetric.jl:103` |
| `SymmetricMatrix` | `A.S` | `GO/src/special_matrices/symmetric.jl:143` |
| `AbstractTriangular` | `A.S` | `GO/src/special_matrices/triangular.jl:8` |
| `Manifold` | `A.A` | `GO/src/manifolds/abstract_manifold.jl:147` |
| `StiefelLieAlgHorMatrix` | `(A.A, A.B)` | `GO/src/lie_algebras/stiefel_lie_algebra_horizontal.jl:81` |
| `GrassmannLieAlgHorMatrix` | `(A.B,)` | `GO/src/lie_algebras/grassmann_lie_algebra_horizontal.jl:80` |

Its docstring states the principle this package arrived at independently: *"For every one of these types
the free parameters are the coordinates the optimizer should work in, so each primitive is the
corresponding operation on `parent`."*

So `freeparameters` is deliberately **not** `Base.parent`, but adoption is one method rather than one
per type. `parent` of a `SubArray` is the whole underlying buffer rather than the view's own entries, so
Base's relation and this one do not agree in general, and flattening on `parent` would silently take in
too much.

**The ecosystem moved while this document sat still, for the second time.** Every version in the table
above changed between the 2026-08-19 revision and this one, and §8 said "Phases 2 and 3 are not
started" while `AbstractNeuralNetworks` had already shipped Phase 2 as 0.7.0 and both
`GeometricOptimizers` and `GeometricMachineLearning` had shipped half of Phase 3. Two of the plan's own
predictions came out differently: the compatibility alias §8 asked for was deliberately not added, and
`SymbolicNeuralNetworks`, which §8 said needed no change, adopted `ParameterLayout` and deleted
`FlatSlice`. Both are recorded where they happened. The lesson for whoever revises this next is that a
status claim here is worth exactly as much as the last time somebody checked it against the sibling
checkouts, so §8 is now written as *where each package stands* rather than as a list of phases to do.

## 4. Defects on record

D1, D2, D3, D4, D7, D8, D9, D10, D11, D12, D13 and D14 are fixed — which is all of them but D5 and
D6, and those two are open *in `GeometricOptimizers`* rather than here. D15 through D21 are new in this
revision and are all seven defects of this package's own 0.2.2 work, found by running it rather than by
reading it.

**D17 and D18 came from reviewing 0.2.2 itself**, and the lesson is narrower than D16's and worth as
much: D13 was fixed on the two walks the tests measured, and the three that take a second set were
neither fixed nor measured — while being the walks that run every iteration rather than once. A
guarantee holds where it is asserted and nowhere else.

**D16, D19 and D20 are one lesson from three sides: a harness reports what it is pointed at.** D16
swept its widths in one process, so every row but the first measured what its predecessors had
compiled. D19 timed a direct call, so the inference being timed was spent before the clock started.
D20 is the argument rather than the clock — every set the sweep and the wide-branch tests build is a
bare `NamedTuple`, and a consumer holds a `NetworkParameters`. On Julia 1.11 that is 1.35 s against
13.40 s for `parameterlayout` on the same 369 leaves, so the committed table reports the cheap column
and calls it the cost.

**D21 is the fourth side, and it is what the other three are for.** Pointing the harness at the shape
a consumer holds is what made a 13× gap visible; D20 then closed on the gap as a property of the two
entry points, because the bare column looked healthy and nothing in it moved. It was one unused type
parameter, and the bare column could not have shown it — only the wrapped path has a caller that has
to infer through the layout type. A two-column sweep earns its keep by making a difference visible,
not by explaining it, and D20 stopped a step early.

**Three entries in this table were stale when this revision began**, and all three in the same
direction: D3, the `changebackend` half of D8, and D11 were recorded as open and had been fixed in
`GeometricOptimizers` 0.5.0 and `GeometricMachineLearning` 0.6.0. §3's lesson had one revision of
practice and then failed again.

**D12, D13 and D14 are this package's own, and none of them was found here.** All three came from the
review of `GeometricOptimizers` #68, which needed `mapparameters` on the one parameter shape it has a
named consumer for and could not compile it. They are grouped because they are one cliff seen from
three sides — compile time, forward allocations, reverse allocations — and because two of them
contradicted docstrings in this repository that the test suite was affirming, since every set it tested
was narrow enough to stay below the cliff. `test/wide_branch_tests.jl` is the width a consumer actually
has, and `scripts/wide_branch_cost.jl` is the harness.

| # | defect | location | status |
|---|---|---|---|
| D1 | `Aversion = "0.1.0"` typo — the package had **no `version` field at all** | `Project.toml:4` | **fixed, phase 1** |
| D2 | Stale `Manifest.toml` pinning `AbstractNeuralNetworks v0.3.0` | `Manifest.toml` | **fixed, phase 1** — the file is untracked and `.gitignore`d |
| D3 | `changebackend` for the five wrapper types is defined **only inside GML's HDF5 extension**, so `changebackend(GPU(), nn)` on a `PSDLayer`/`LASympNet` network MethodErrors unless HDF5 happens to be loaded | was `GML/ext/HDF5Ext.jl` | **fixed** — `GO/ext/AbstractNeuralNetworksExt.jl` carries one method over `Manifold`/`VectorStorageMatrix`/`AbstractLieAlgHorMatrix`, delegating to `mapstorage`, and GML's copies are gone. `changebackend(GPU(), nn)` no longer depends on HDF5 being loaded |
| D4 | HDF5 returns group members sorted, worked around by a regex "natural sort" that silently falls back to lexicographic order unless *every* key matches `^\D+\d+$` | was `GML/ext/HDF5Ext.jl:87-97` | **fixed** — this package records key order as a group attribute, and GML's `_natural_sort_keys` is gone. `ext/HDF5Ext.jl:217-221` keeps the regex guess for files that record nothing better, guarded so it leaves the order alone unless every key matches |
| D5 | `unflatten` rebuilds a manifold via `Base.typename(typeof(x)).wrapper`; the comment records that hardcoding `StiefelManifold` previously turned a `GrassmannManifold` into one on every round trip | `GO/…/named_tuple_wrapper.jl:16-23, 78-79` | open in GO — step 2 of the GO work in §8. Not a defect here: `rebuild` takes a prototype, so the concrete type cannot drift |
| D6 | `ParameterHandling.flatten` defaults to `Float64`, silently promoting `Float32` networks | `GO/…/named_tuple_wrapper.jl:14` | open in GO — step 2 of the GO work in §8. Not a defect here: `flatten(ps)` takes its element type from `parameter_eltype(ps)` |
| D7 | With the type moved out of ANN, `ZygoteRules.pullback(f::Function, ::NeuralNetworkParameters)` owns neither argument type and becomes piracy there | was `ANN/src/pullback_for_applychain.jl:10-17` | **fixed** — the generic method is `ext/ZygoteRulesExt.jl:17`, and `ANN/src/pullback_for_applychain.jl:10-14` records the move |
| D8 | GML's `h5save(::H5DataStore, ::StiefelManifold, ::AbstractString)` and `changebackend(::NeuralNetworkBackend, ::StiefelManifold)` are genuine type piracy — the generics are ANN's and the types are GO's, so GML owns nothing in either signature | `GML/ext/HDF5Ext.jl` | **fixed, both halves.** The `h5save` half went with GML's extension, which defines no `h5save` at all; the `changebackend` half went to `GO/ext/AbstractNeuralNetworksExt.jl` — see D3. `GML/ext/HDF5Ext.jl` is 86 lines and is `save`/`load` entry points and nothing else |
| D9 | `Base.NamedTuple(p::NeuralNetworkParameters)` — Base's constructor and ANN's type, so the package defining it owns neither. Surfaced by GML [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207) | was `GML/src/layers/forcing_dissipation_layers.jl` | **fixed, phase 1** — `src/parameters.jl:162` |
| D10 | `h5save(::HDF5.Group, ::NeuralNetworkParameters, ::AbstractString)` — ANN's generic and ANN's type, from GML. Nothing existed for a parameter set nested at a path, which the parameter-dependent architectures produce. Also GML [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207) | was `GML/ext/HDF5Ext.jl` | **fixed** — `src/io.jl` owns the generic HDF5 path, so a nested parameter set serialises without anybody committing piracy |
| D11 | `GeometricOptimizers.GlobalSection(ps::NetworkParameters)` — `GlobalSection` is GO's and `NetworkParameters` is this package's, so GML owns neither. Not in the original survey; it appeared when GML adopted the container while GO had not | was `GML/src/optimizers/optimizer.jl` | **fixed** — GO owns it, at `GO/src/global_sections/global_sections.jl:45`, and deliberately returns a plain `NamedTuple` tree rather than a container: a section is not a parameter |
| D12 | **The walks were superlinear in the width of one branch.** Every across-children walk was an `@inline`d `Base.tail` chain, and `Base.tail` yields a new tuple type per level — so `k` children cost `k` specialisations over `O(k)`-long argument types and inference grew as `k³`. `flatten` on a flat 369-leaf set, the MNIST transformer of GMLDatasets, did not finish | was `src/flatten.jl`, `src/walk.jl`, `src/leaves.jl`, `src/layout.jl`, `src/derivatives.jl` | **fixed, 0.2.2** — written out as `@generated` flat bodies at literal indices. 369 leaves: `flatten` 2.05 s, `mapparameters` 0.00 s. Julia 1.11 is the compat floor as a consequence: 1.10 cannot inline a `@generated` body, and that inlining is what D13 needs. **Two things `GeometricOptimizers` [#70](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/70) adds, one confirming this account and one qualifying it.** Its three quadrature folds carry **no `@inline` at all**, deliberately, and hit the same cliff at the same width — so the cliff is not conditional on the `@inline`, which is independent confirmation from the other end of 0.2.2's finding that removing it did not help. But the **Julia version matters more than this entry implies**: an un-`@inline`d `Base.tail` fold at 369 children is 0.65 s on 1.11.9 and 26 to 35 s on 1.12.7 and 1.13.0-rc3. D12 was diagnosed on walks that were written out for every version, so the interaction never showed here. Anyone reasoning from this entry to a *new* `Base.tail` walk must not conclude that going without the `@inline` is safe on 1.11 evidence |
| D13 | `flatten!`/`unflatten!` allocated past about forty children — 81 488 bytes at 48, 187 808 at 64 — against the guarantee in their own docstrings, because `values(ps)` and `values(l.children)` materialise a temporary tuple per branch | was `src/flatten.jl` | **fixed, 0.2.2** — the walks index the branch in place with `getfield` and take no `values`. Zero at every width and every depth, on Julia 1.11, 1.12 and 1.13. See D17: the fix reached the two forward walks and not the three that take a second set |
| D14 | The reverse pass had D13's defect too: 70 592 bytes per pullback call at 48 children, 161 504 at 64 | was `src/derivatives.jl` | **fixed, 0.2.2** — `_accumulate_named!` splices the branch's keys in as literals, which is also what keeps `_cotangent_get`'s `haskey` a compile-time question |
| D15 | **The `@generated` walks raised `MethodError: … may be too new` when the sources are evaluated rather than loaded from a cache.** `_map_zip` and `_foreach_zip` called `_children_arity` from their *generator* bodies, and it was defined below them in the same file; a generator runs in the world age of its own method definition, so it was invisible. Precompilation hides this by giving every method in a module one world age, which is why the whole suite passed with it present | was `src/walk.jl:277,284` vs `:311` | **fixed, 0.2.2** — the helper is defined above every caller, and `test/world_age_tests.jl` runs the walks in a subprocess under `--compiled-modules=no`, which is the only way to ask |
| D16 | **`scripts/wide_branch_cost.jl` swept its widths in one process, so every row but the first measured what its predecessors had compiled.** Every figure the 0.2.2 work first quoted was wrong — `flatten` on 0.2.1 at 128 children was recorded as 17.57 s against 2.08 s cold, and at 369 as "not run to completion" against 12.82 s, and `mapparameters` read as `0.00 s` where cold it was 1.51 s. `GeometricOptimizers` closed an issue of its own on that `0.00 s` and deleted a file on the strength of it | was `scripts/wide_branch_cost.jl` | **fixed, 0.2.2** — one process per width, and the `--cold-map` flag deleted with the need for it. A measurement you have to ask for in a special way is one the default run is getting wrong |
| D17 | **D13 was fixed on the two walks that were measured and not on the three that were not.** `mapparameters!`, `mapstorage!` and `foreachparameters` turned each *further* argument into a `values(...)` tuple per branch, and a skipped one into a fresh tuple of `nothing`s — 800 bytes a call on a flat 48-child set, 6 144 at 369, and three to four times that on a branch of branches. These are the walks an optimizer runs every iteration, where `flatten!` runs once or twice | was `src/walk.jl:152,196,293-297` | **fixed** — `_foreach_zip` reads every argument with `getfield` at a literal index; `_values_for` and `_tuple_for` are gone. Zero at every width, depth and arity. Not caught because measuring a `Vararg` method needs a zero-argument closure: `@allocated f(a, b)` lowers to `Base.allocated(f, a, b)`, whose own splat allocates |
| D18 | **The in-place walks paired the children of two keyed branches by *position*.** `_values_for(x, ks)` ignored its `ks`, so two same-shaped sets whose keys were ordered differently wrote every leaf into the wrong parameter without a word, where `mapparameters` has always raised | was `src/walk.jl:293` | **fixed** — the generated body checks a named argument's keys against the branch's own, in the generator, so the guard is free at run time and stricter than `_check_keys`. A behaviour change: positional pairing of differently-keyed sets now raises |
| D19 | **`scripts/wide_branch_cost.jl` still was not measuring compile time after D16.** `first_call` timed a direct `f(args...)`, and compiling `first_call` infers through that call — so the inference being timed was spent before its own `t = time()` ran. On Julia 1.13 the committed harness printed 0.00 s for every width in every column, where an opaque call measures 1.61 s for `parameterlayout` and 3.95 s for `flatten` at 369 children | was `scripts/wide_branch_cost.jl:38-42` | **fixed** — the call goes through `Base.invokelatest`. D16 and this are the same lesson from two ends: arrange the harness so the cost cannot have been paid somewhere the clock is not looking |
| D20 | **The wide-branch tests and the sweep script measure a bare `NamedTuple`, and a consumer holds a `NetworkParameters`.** The two reach `parameterlayout` through different methods and, on Julia 1.11, cost wildly different amounts on the same leaves: 1.35 s bare against 13.40 s wrapped at 369, 2.77 s against 87.77 s at 768. So the 0.2.2 table reports the cheap column, and issue [#16](https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/16) read the gap between the two as a cost of *nesting* — which it is not: a flat 369-leaf set and a 16 × 24 one cost the same 1.35 s bare and the same 13.8 s wrapped | was `test/wide_branch_tests.jl:27`, `scripts/wide_branch_cost.jl:32` | **fixed** — the sweep runs both shapes at every width, each in its own process, and `test/wide_branch_tests.jl` asserts the round trip and the zero allocations on the wrapped shape. This entry closed on the measurement without a cause, and recorded that no code needed to change on the grounds that the whole path cost nearly the same either way (19.96 s bare, 21.90 s wrapped at 369) so the wrapper only decided which entry point paid. **That was where to look and not what to conclude**: the gap had a cause, D21 is it, and 0.2.3 removed it. The two columns agree now |
| D21 | **`LeafLayout` carried a `prototype` field that nothing read, and it was the cost D20 measured.** A `LeafLayout` is by construction the terminal case, so `rebuild` on it is the identity and is never called — but the field's type parameter put each leaf's *concrete array type* into the layout type of every branch above it, 1849 nodes in the type tree of a 369-leaf wrapped layout against 742 without. `_layout(::NetworkParameters, ::Int)` is where a caller paid for it, inferring through the child walk's whole return type to wrap its result, and inference on such a type grows faster than the type does: 2.5 times the type was 13 times the time. The bare column never showed it, because only the wrapped shape has such a caller — which is exactly why D20's two-column sweep was needed to find it and D20's own conclusion was not | was `src/layout.jl:29` | **fixed, 0.2.3** — the struct is `LeafLayout{N}` over `range` and `size`. `parameterlayout` on a 369-leaf set inside a `NetworkParameters` goes 13.40 s → 1.05 s and at 768 leaves 87.77 s → 3.01 s, the whole 369-wrapped path 21.90 s → 8.64 s, and Julia 1.11, 1.12 and 1.13 agree at 1.05/1.13/1.03 s where the compat floor used to be the one that behaved differently. `scripts/leaf_layout_cost.jl` is the harness. Issue [#15](https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/15) filed it as hygiene and said explicitly it was not the cause: it had compared two leaf *sets* under one layout type, never the layout type with and without the parameter |
| D22 | **`scripts/wide_branch_cost.jl` did not sweep the fold.** It timed `parameterlayout`, `map(zero, ·)`, `mapparameters`, `flatten` and `unflatten` and the in-place allocations, at both shapes and every width, and not `foldparameters` at either. That was the one walk whose downstream analogue had just produced a two-orders-of-magnitude compile cliff, and the only thing standing in for a figure was the total wall clock of a suite that folds 369 children while asserting the value alone. D16 and D19 are the same lesson about the clock; this is it about an argument | was `scripts/wide_branch_cost.jl` | **fixed, 0.2.4** — three columns, `fold`, `foldzip` and a `tailfold` control that is the `Base.tail` recursion a consumer writes without a zipped fold. Filed as part 2 of issue [#19](https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/19) |

D8's two halves are the clearest evidence for the protocol, and they were fixed by different packages
at different times. With the structured types upstream of the package that trains with them, a
*generic* HDF5 path driven by `freeparameters`/`rebuild` is the only way to serialise them without
somebody committing piracy — and once that path existed, GML's copy could simply be deleted while GO
registered its own types. The `changebackend` half is the same shape and is still waiting, because
`changebackend` is `AbstractNeuralNetworks`' generic rather than this package's: the methods belong in
a `GeometricOptimizers` extension on `AbstractNeuralNetworks`, which is a different release chain.
`GML/ext/HDF5Ext.jl:17-20` records that reasoning in place.

**The flat-ordering question is settled.** Earlier drafts left open whether this package's leaf
ordering matches GO's, since `GO/test/named_tuple_parameters.jl:21-22` asserts specific flat ranges.
It does — verified by running both flattenings over one of each leaf family, and explicable rather than
lucky: `Base.vec` and `Base.parent` return the same storage for every `VectorStorageMatrix`, and this
package copies leaves in linear index order (`src/flatten.jl:103-104`). The table in §8 works through
it family by family. So the migration is byte-for-byte behaviour-preserving on the flat vector and
GO's hardcoded ranges stand — which is worth pinning with a test while both packages are still present,
since nothing else would catch a later drift.

## 5. Evidence

Verified by running code rather than by reading, so it can be rechecked.

**A package cannot export a type sharing its own name.** The module binding wins at the `using` site:
a scratch package `Foo` exporting `struct Foo` gives `using Foo; Foo(1)` →
`ERROR: objects of type Module are not callable`. (`Foo.Foo` *is* the struct; the clash is only in the
importer's namespace.) Hence the type is `NetworkParameters`. An earlier draft concluded from this that
`AbstractNeuralNetworks` should keep a `const NeuralNetworkParameters = NetworkParameters` alias; it
removed the name instead, so that one type has one name — see §8.

**Flattening is cheap.** 1.3M parameters, five 512-wide dense layers, `Float64`: `flatten` 0.144 ms,
`flatten!` into a preallocated buffer 0.133 ms, `unflatten` 0.149 ms — against **0.946 ms for one
forward pass at batch 32**. A round trip is about 10 % of a single forward pass.

**Reshaped views keep the BLAS fast path**, so the argument against shared storage is *element type*,
not speed: `reshape(view(v, r), m, n)` is a `StridedArray` and `mul!` runs at 0.089 ms either way. An
earlier draft had this backwards. The decisive constraint is that `unflatten` must be able to return
arrays of a different element type from the parameters — `ForwardDiff.Dual`, noted at
`GO/…/named_tuple_wrapper.jl:19-22` — and a `Float64` buffer cannot be shared with a `Dual` view. Hence
copy semantics, with allocation-free in-place variants for inner loops.

## 6. The package, as built

| file | contents |
|---|---|
| `src/parameters.jl` | `NetworkParameters`, `params`, `ParameterSet`, and the `NamedTuple` forwarding |
| `src/leaves.jl` | `freeparameters`, `rebuild`, `parameter_metadata`, `parameter_eltype`, `isterminal`, `isparametertree` |
| `src/walk.jl` | `mapparameters`, `mapstorage`, the in-place and `foreach` forms, `foldparameters` and `foldstorage`, all of them at any arity |
| `src/layout.jl` | `ParameterLayout` and its five concrete cases, `parameterlayout`, `parameterrange`, `flatlength` |
| `src/flatten.jl` | `flatten`, `flatten!`, `unflatten`, `unflatten!`, and the Jacobian split |
| `src/flat_parameters.jl` | `FlatParameters`, the `AbstractVector` interface, layout-preserving `similar` and broadcasting |
| `src/io.jl` | the `h5save`/`h5load`/`save`/`load` generics and the type registry |
| `src/derivatives.jl` | `ChainRulesCore` rules for both conversions |
| `ext/HDF5Ext.jl` | the storage methods |
| `ext/ZygoteRulesExt.jl` | `pullback` on a `NetworkParameters`, D7's new home |

Design points worth keeping in view:

- **`ParameterSet` is the dispatch type; `NetworkParameters` is the container.** New in 0.2.2, and it
  is the answer to a question this document never asked: what should a method take when it wants *the
  parameters* and does not care which of the two forms it was handed? Every consumer had answered it
  separately — `Union{NamedTuple, NetworkParameters}` inline at eight sites in `AbstractNeuralNetworks`
  and fifteen in `GeometricMachineLearning`, `const EquationSet` in `SymbolicNeuralNetworks`, and in
  `GeometricOptimizers` a *narrower* alias, `ParameterContainer{T}`, plus sixteen inline fallbacks
  where that one did not fit.

  It is keyed only, and deliberately not `isparametertree`'s domain, which also admits a `Tuple`: a
  `Tuple` is a branch the walks recurse into — it is what `freeparameters` returns for a multi-block
  leaf, and why `TupleLayout` exists — but never a set of parameters handed in whole. The two agree on
  everything except the `Tuple`s, and `test/parameters_tests.jl` asserts exactly that rather than the
  happy path.

  It also puts **no bound on the element type and none on the depth**, which is the whole difference
  from `GeometricOptimizers.ParameterContainer{T}` — see §8 for why that one has to exist as well.

- **The leaf protocol is the whole extension surface.** `freeparameters(x)` gives the differentiable
  storage, `rebuild(prototype, data)` puts a leaf back. `freeparameters` may return an array, a scalar,
  or a `Tuple`/`NamedTuple` for multi-block types, and the recursion continues into it. Nothing in the
  package holds a list of structured types.
- **`rebuild` takes a prototype, not a type.** Non-differentiable fields (`n`, `N`) come across for
  free; `data` may have a different element type; and the concrete type cannot drift, which is D5's bug
  class.
- **A layout is a value**, not a closure: storable in an optimizer cache, comparable, inferable. This is
  the concrete difference from `ParameterHandling`. It also subsumes `FlatSlice`, which
  `SymbolicNeuralNetworks` used to define and has since deleted — see §8.
- **Copies are range `copyto!`s**, so no element is ever indexed individually and GPU arrays work
  unchanged — the failure documented at `GO/scripts/mnist_cuda.jl:5-15`.
- **The HDF5 writer records key order** as a group attribute, which fixes D4 for new files. Older files
  — `ANN`'s attribute-less groups, and GML's `gml_type` tagging — still load, the latter falling back to
  the regex guess because they record nothing better.
- **HDF5 loading has two paths**, because a file has no prototype to rebuild against: pass one
  explicitly (`load(NetworkParameters, h5, prototype)`, needs no registration) or let the owning package
  call `register_parameter_type!`. `parameter_metadata` exists for the second path and was not in the
  original design — a prototype carries `n` and `N` across, but a file has to be told.
- **`save`/`load`/`h5save`/`h5load` are not exported**, matching `AbstractNeuralNetworks`, since those
  names collide with most of the ecosystem.

### Verification

```bash
julia --project -e 'using Pkg; Pkg.test()'                                            # 471 tests
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'   # doctests
julia --project=. scripts/wide_branch_cost.jl                                         # D12/D13/D17/D20/D22
julia --project=. scripts/leaf_layout_cost.jl                                         # D21
```

The sweep prints two shapes at every width, bare and inside a `NetworkParameters`, each in a process of
its own (D20). **The two `layout` columns now agree on every Julia — about 1.05 s wrapped against 1.04 s
bare at 369 children — and that is the reading rather than a warning.** Until 0.2.3 the wrapped column
showed a cliff the bare one did not, about 13.4 s at 369 on Julia 1.11, and D21 is what that was. A
cliff re-appearing there is the regression to act on.

That leaves the fork itself to be checked some other way, since agreeing columns no longer distinguish
a working fan-out from a broken one. Check it directly: `fan_out` `run`s one child process per row and
each prints its own line, so `ps aux` during a run, or simply invoking one row on its own —
`julia --project=. scripts/wide_branch_cost.jl --in-process wrapped 369` — is what says the isolation
holds. A row run alone and the same row inside the sweep have to read the same.

It also covers a **369-child branch** — the width of GMLDatasets' MNIST transformer, and the shape D12,
D13 and D14 were about — through `flatten`/`unflatten`, `mapparameters` at both arities,
`mapparameters!`, `foreachparameters` with and without the `nothing` skip, `foldparameters` and
`foldstorage` at both arities, the `unflatten` pullback, and the in-place forms at zero allocations,
and a 48-child one inside a `NetworkParameters` for the same round trip and the same zero
allocations. It asserts the properties and not the timings, because a wall-clock bound would be a
flake on a loaded machine; the regression test is that the file *completes*, which before 0.2.2 it
did not. It costs the suite about 45 s on Julia 1.13 and
about 1 m 45 s on 1.11, all of it compilation at that width, and that is the price of covering the
width a consumer has rather than the width that is convenient.

The suite covers, beyond round trips: `Float32` fidelity (D6); zero allocation for `flatten!` and
`unflatten!` — measured from inside a function, which is the claim that matters, an optimizer's inner
loop rather than the top level of a testset; a structured leaf and a two-block leaf whose types
survive the round trip (D5); scalar, empty and tuple leaves; `Dual`-valued
unflattening; agreement between `ForwardDiff` on the flat form and `Zygote` on the structured one;
Zygote through both conversions; a structurally-zero gradient block; the `nothing`-branch skip in the
walks; ten-layer HDF5 key ordering (D4); both HDF5 load paths; and files in the two older formats.

## 7. Why not an existing package

**ParameterHandling.jl**, the tempting choice since `GeometricOptimizers` already depends on it, fails
on four counts, each one something GO has hit in practice:

1. Its `flatten(::Type{T<:Real}, x)` methods cover the Base types, so extending it for `Float32`
   fidelity *is* piracy. `GO/…/named_tuple_wrapper.jl:25-29` says so, citing issue #16.
2. `unflatten` is a chain of closures: not type stable, not storable, not comparable.
3. `flatten(x)` defaults to `Float64` (D6).
4. No GPU method — a `CuVector` falls through to the element-mapping `AbstractVector` path
   (`GO/scripts/mnist_cuda.jl:5-15`).

It also cannot round-trip a `VectorStorageMatrix` at all, which GO's own docstring spells out
(`vector_storage_matrix.jl:18-20`): the `AbstractMatrix` method reshapes the flat vector back to
``n \times n``, and ``n(n\pm1)/2`` numbers do not reshape to that. And it pulls `IterTools`,
`LogExpFunctions`, `InverseFunctions` and — oddly — `Test` as hard dependencies.

**ComponentArrays.jl**: leaves come back as `ReshapedArray`/`SubArray` views, which cannot represent the
structured types, and there is nowhere for their non-array fields (`n::Int`, `N::Int`). Adds
`ArrayInterface` and `StaticArrayInterface`.

**Optimisers.jl `destructure`**: lean, and it handles parameter tying, but it drags in the optimiser-rule
machinery `GeometricOptimizers` exists to replace, and `Functors.fmap` would need teaching the
structured types anyway — the same extension work plus a dependency.

## 8. Where each package stands

### `AbstractNeuralNetworks` 0.7.1 — done

The container moved. `src/parameters.jl` and the whole `ext/` directory are gone; the package depends
on this one (`Project.toml:10`, compat `:19`) and imports the five names it extends or reaches through
(`src/AbstractNeuralNetworks.jl:32-33`). `changebackend` is one `mapparameters` call
(`src/utils/changebackend.jl:16-18`), which also gave it the `Tuple`-branch case the two methods it
replaces did not have; `_statify` is another (`src/static_cpu_backend.jl:30`). D7's generic `pullback`
moved here. `test/parameters_tests.jl` and `test/parameters_hdf5_tests.jl` became
`test/parameters_seam_tests.jl`, which tests the seam rather than the container.

**`ParameterSet` replaced eight inline unions** (`chain.jl:52`, `model.jl:35`, `losses.jl:90,95,99`,
`static_cpu_backend.jl:30`, `utils/changebackend.jl:16`). Note that ANN's own `ArrayNamedTuple`
(`src/utils/array_named_tuple.jl:23`) is a *different* thing and stays: it is about network inputs and
outputs rather than parameters, and it is the one name it shares with `GeometricOptimizers` by
coincidence. Worth renaming to leave `ArrayOrNamedTuple` alone as the only name there, but that is
cosmetic and not done.

**The compatibility alias was deliberately not added.** Earlier drafts of this section called for
`const NeuralNetworkParameters = NetworkParameters` and a re-export, and listed the downstream uses it
would have to support. 0.7 removed the name outright instead, so that one type has one name across the
ecosystem — `src/AbstractNeuralNetworks.jl:24-27` says so, and
`test/parameters_seam_tests.jl:16-25` pins it by asserting the name is not even defined. Every call
site the survey found now names `NetworkParameters`: `src/chain.jl:52,57` and
`src/neural_network.jl:12,32-35` here, `SNN/src/codegen/equation_sets.jl:35,72`,
`GML/src/GeometricMachineLearning.jl:8` and `NLI/src/NonlinearIntegrators.jl:47-53`. The two SNN sites
that used to build the container by keys went further and no longer build one at all: the nested walk
in `SNN/src/derivatives/derivative.jl:26` is `mapparameters`.

### `SymbolicNeuralNetworks` 0.7.0 — done, and more than was asked

**`EquationSet` is gone.** It was `Union{NamedTuple, NetworkParameters}`, and this was the only package
in the ecosystem to have given that union a name — which is why `ParameterSet` is named after the shape
rather than after the contents. `EquationSetFunction` and `EquationSetArrayFunction` keep their names;
they are named for what they carry.

Earlier drafts said this package needed no change, on the grounds that its `flatten_equations` solves a
different problem. That was half right: restoring batch dimensions is still its own
(`unflatten_batch(::LeafLayout, ::AbstractMatrix)`, `src/codegen/equation_sets.jl:192`), but the
*layout* was not. `FlatSlice` is gone, and `flatten_equations` is one line over this package —
`flatten(Num, mapparameters(scalar_expressions, eqs))` (`:146`) — with `split_result` dispatching on
`ParameterLayout` (`:159-160`). The design point in §6 that called `ParameterLayout` a generalisation
of `FlatSlice` turned out to be the plan for a package this document said to leave alone.

### `GeometricOptimizers` 0.5.0, and 0.6.0 on the branch — done

All three steps this section used to set out have shipped, and not in the order it predicted.

**Steps 1 and 2 shipped together in 0.5.0**, as one breaking release rather than the patch this section
proposed: `ParameterHandling` is gone, `ext/NeuralNetworkParametersExt.jl` supplies the whole leaf
protocol for GO's structured matrices, the flat buffers are preallocated on the quasi-Newton caches, and
`NetworkParameters{T}` is a member of `OptimizerSolution{T}`. D5 and D6 went with it: `rebuild` takes a
prototype, so the concrete type cannot drift, and `flatten(ps)` takes its element type from the
parameters.

**Step 3 is the 0.6.0 branch** (#68 and #69). The elementwise primitives and the ~40 sites around them
take a container, and a nested container now runs `solve!`, `BFGS` and `DFP` — which is what this
section meant by "the swap". Two things it predicted came out differently:

- **The trap did not fire.** This section warned that adding `NetworkParameters` to `OptimizerSolution`
  would invert GML's `_use_go_cache` test and `MethodError` on the first training step. GML had already
  put the `NetworkParameters` branch *ahead* of `_use_go_cache` in anticipation, so the inversion is
  harmless. What was wrong was the comment beside it, which still said the ordering was invisible —
  0.5.0 had made it load-bearing a release earlier. Corrected in GML.

- **#16 is not closed, but one of its five sites is gone.** Closing the group needs the `NamedTuple`
  methods *removed*, and they cannot be: GML hands GO one bare layer `NamedTuple` per layer, and
  GMLDatasets' MNIST scripts pass a flat `NamedTuple` directly. But `outer!` on a parameter set turned
  out to be *internal* to the quasi-Newton caches once #69 routed it through the flat buffers — no
  consumer calls it — so it could simply be deleted. That is the distinction to apply to the other four:
  ask whether a consumer reaches the generic, not whether the signature is pirated.

**What `ParameterContainer{T}` is for, and why `ParameterSet` does not replace it.** GO keeps a narrower
alias, `Union{ArrayNamedTuple{T}, NetworkParameters{T}}`, and it has to. `T` means two different things
across it — for `ArrayNamedTuple{T}` a guarantee that every leaf is an `AbstractArray{T}` *and* that the
set is flat, for `NetworkParameters{T}` a promotion over leaves at any depth — so it is the right type
exactly where `T` also appears elsewhere in the signature, beside a `Matrix{T}` or an `f̄::T`, and the
wrong type everywhere else. The sixteen sites that had to fall back to the inline union are the
measurement of that, and they take `ParameterSet` now.

**There is no remainder. Three things this section listed as outstanding had already been done**, all
three in GO commit `b4c710c`, which landed inside #69 rather than #68 — which is why reading the two
pull request bodies does not find them:

- **`src/parameter_walks.jl` is deleted** and `_mapleaves`/`_mapleaves!` are `mapparameters`/
  `mapparameters!` at 44 call sites. This section said the file "can go" once D12 was fixed; it went.
- **The `_as_walkable` catch-all is answered, and the answer is stronger than the question.** This
  section had it as an open question — a tree zipped against something that is neither a branch nor
  `nothing` walks there and raises here — with the implication that the catch-all might be a capability
  worth keeping. It was not. `map` over a `NamedTuple` and a bare `Vector` matches neither of Base's
  specific methods, falls through to the generic iterator `map`, and zips the branch's *entries*
  against the leaf's *elements*: `map(f, (a = [1.0], b = [2.0]), [9.0, 9.0])` returns
  `[([1.0], 9.0), ([2.0], 9.0)]`, and a leaf shorter than the branch is silently truncated. So the
  catch-all turned a caller's bug into a wrong answer where `_as_namedtuple`'s three exhaustive methods
  raise a `MethodError` naming the type. Deleting the file fixed a latent bug of GO's own.
- **The pirated `l2norm`s are gone**, upstream. `l2norm(::AbstractMatrix)` and
  `l2norm(::AbstractFloat)` were piracy on *Base* types; `GeometricBase` 0.14.9 takes
  `L2norm(x::AbstractArray)`, which is the matrix method with the `vec` removed, and its
  `L2norm(x::Real) = x^2` had always given `abs` for the second. GO's `Project.toml` pins `0.14.9` for
  it. Removing the `vec` was not only ownership: `vec` of a `Matrix` allocates a 32-byte wrapper, so
  `l2norm` of a parameter set had cost 32 bytes per matrix leaf on every stopping criterion of every
  iteration.

**One thing to carry back into D13 and D17.** GO 0.6.0's allocation table reported a 20-iteration
`solve!` going from 2 431 912 bytes to 1 559 832 on its flat `NamedTuple` problem and read the whole of
it as its own flat-buffer work. It is not: its 0.5.0 column resolved **0.2.1** and its 0.6.0 column
0.2.2, so **737 840 of those 872 080 bytes are D13 and D17** and 134 240 are GO's. Verified by pinning
0.2.1 against GO's own 0.5.0 worktree, which returns the old column byte for byte. Corrected in GO with
a three-column table holding the dependency fixed. Worth knowing here because it is the largest measured
consequence either defect has, and neither entry above records one.

**§3's lesson, a third time in one revision.** Three entries of the defect table were stale when that
revision began — D3, half of D8, D11 — all recorded as open after `GeometricOptimizers` 0.5.0 and
`GeometricMachineLearning` 0.6.0 had fixed them. These three are the same failure in the same
direction, and they survived a revision that had just written the lesson down. The specific trap here
is worth naming because it is not the general one: **the work was done in a commit under a pull request
whose body describes the state before it.** #68's body still argues for keeping `_mapleaves` and cites
GO's issue D9 as the reason; #69's body does not mention the walk at all. A status claim checked against
the pull requests rather than against the tree gets all three of these wrong.

### `GeometricMachineLearning` 0.6.1 — done

`ext/HDF5Ext.jl` is **86 lines** and defines no `h5save` and no `changebackend`: it is `save`/`load`
entry points dispatching on GML's own `NeuralNetwork`, delegating the traversal here.
`_natural_sort_keys` is gone (D4), the pirated `h5save` is gone (half of D8), and the five
`changebackend` methods moved to `GO/ext/AbstractNeuralNetworksExt.jl` in GO 0.5.0, which closed D3
and the other half of D8. `GlobalSection(::NetworkParameters)` went the same way, closing D11. The
package depends on this one (`Project.toml:19`) and re-exports `NetworkParameters`
(`src/GeometricMachineLearning.jl:106`).

**`ParameterSet` replaced fifteen inline unions**, across `loss/`, `optimizers/optimizer.jl`,
`pullbacks/zygote_pullback.jl` and `data_loader/optimize.jl`.

**What remains, and it is one thing kept on purpose.** `src/map_to_cpu.jl` is one `mapstorage` call;
`apply_toNT` and `_eltype` are gone. The step dispatcher `_tree_optim_step!`
(`src/optimizers/optimizer.jl:270-292`) stays hand-written, and now says why: it recurses on the
*cache* tree and stops where the cache stops, so a layer's `NamedTuple` reaches `_leaf_optim_step!`
whole, which is what one `GeometricOptimizers` cache is for; and `λY` is broadcast rather than zipped,
one `GlobalSection` standing in for a whole subtree. `foreachparameters` does neither. Three earlier
revisions of this section listed `map_to_cpu`, `apply_toNT`/`_eltype` and the `changebackend` methods
here after they had been done.

The comment on `_make_optimizer_cache`'s branch ordering was **wrong for a release** and is corrected:
it said the ordering was invisible because `NetworkParameters` is not one of the types
`OptimizerSolution` unions, and GO 0.5.0 had made it one, so the ordering is load-bearing. The code
was already right; only the comment was not.

### `NonlinearIntegrators` 0.4.1 — a consumer

Not a target of any phase, but it pins both this package (`Project.toml:16`) and
`AbstractNeuralNetworks = "0.7"` (`:26`), and reaches for `NetworkParameters` directly
(`src/NonlinearIntegrators.jl:47-53`, used in `src/nvi/`). It is here so the version table above is
the whole set of packages a change to this one has to consider.
