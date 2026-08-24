# NeuralNetworkParameters.jl — analysis and plan

Analysis last revised 2026-08-24, against these working trees:

| package | version | commit |
|---|---|---|
| `NeuralNetworkParameters` | 0.2.0 | this repository |
| `AbstractNeuralNetworks` | 0.7.0 | `c324789` |
| `SymbolicNeuralNetworks` | 0.6.0 | `1d568c0` |
| `GeometricMachineLearning` | 0.6.0 | `6db0b98` |
| `GeometricOptimizers` | 0.4.3 | `a8707f3` |
| `NonlinearIntegrators` | 0.4.0 | `1288174` |

Paths written `ANN/…`, `SNN/…`, `GML/…`, `GO/…`, `NLI/…` are relative to the corresponding sibling
checkout.

**Status.** Phase 1 is released and registered in General; 0.2.0 carries the element type of the
leaves on the type, which is what `GeometricOptimizers` needs of a parameter set to take it as a
solution — step 3 of §8. Phase 2 is complete: this package is where the parameter container lives,
and `AbstractNeuralNetworks` 0.7.0 consumes it. Phase 3 is half done — `GeometricOptimizers` 0.4.3
supplies the leaf protocol for its structured matrices, and `GeometricMachineLearning` 0.6.0 has
handed the HDF5 traversal over — with a named remainder in both, set out in §8.
`SymbolicNeuralNetworks` 0.6.0 adopted the layout type, which §8 had not asked it to.

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
implementation covers all of them and needs to know none of the types. What that has come to: ANN's
copies are gone, its `changebackend` and `_statify` each one `mapparameters` call; GML's `h5save` and
its "natural sort" are gone; SNN's `FlatSlice` is gone. What is left is GML's `map_to_cpu`, its
`apply_toNT`/`_eltype` and its step dispatcher, and GO's `apply_toNT` and its `ParameterHandling`
flattening — §8 has both.

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

D1, D2, D4, D7, D9, D10 and half of D8 are fixed. The rest are noted so they are not lost; the status
column says where each stands. D11 is new since the last revision — the piracy inventory grows as
packages adopt the container ahead of the ones they depend on.

| # | defect | location | status |
|---|---|---|---|
| D1 | `Aversion = "0.1.0"` typo — the package had **no `version` field at all** | `Project.toml:4` | **fixed, phase 1** |
| D2 | Stale `Manifest.toml` pinning `AbstractNeuralNetworks v0.3.0` | `Manifest.toml` | **fixed, phase 1** — the file is untracked and `.gitignore`d |
| D3 | `changebackend` for the five wrapper types is defined **only inside GML's HDF5 extension**, so `changebackend(GPU(), nn)` on a `PSDLayer`/`LASympNet` network MethodErrors unless HDF5 happens to be loaded | `GML/ext/HDF5Ext.jl:23-41` | open — see the GML remainder in §8 |
| D4 | HDF5 returns group members sorted, worked around by a regex "natural sort" that silently falls back to lexicographic order unless *every* key matches `^\D+\d+$` | was `GML/ext/HDF5Ext.jl:87-97` | **fixed** — this package records key order as a group attribute, and GML's `_natural_sort_keys` is gone. `ext/HDF5Ext.jl:217-221` keeps the regex guess for files that record nothing better, guarded so it leaves the order alone unless every key matches |
| D5 | `unflatten` rebuilds a manifold via `Base.typename(typeof(x)).wrapper`; the comment records that hardcoding `StiefelManifold` previously turned a `GrassmannManifold` into one on every round trip | `GO/…/named_tuple_wrapper.jl:16-23, 78-79` | open in GO — step 2 of the GO work in §8. Not a defect here: `rebuild` takes a prototype, so the concrete type cannot drift |
| D6 | `ParameterHandling.flatten` defaults to `Float64`, silently promoting `Float32` networks | `GO/…/named_tuple_wrapper.jl:14` | open in GO — step 2 of the GO work in §8. Not a defect here: `flatten(ps)` takes its element type from `parameter_eltype(ps)` |
| D7 | With the type moved out of ANN, `ZygoteRules.pullback(f::Function, ::NeuralNetworkParameters)` owns neither argument type and becomes piracy there | was `ANN/src/pullback_for_applychain.jl:10-17` | **fixed** — the generic method is `ext/ZygoteRulesExt.jl:17`, and `ANN/src/pullback_for_applychain.jl:10-14` records the move |
| D8 | GML's `h5save(::H5DataStore, ::StiefelManifold, ::AbstractString)` and `changebackend(::NeuralNetworkBackend, ::StiefelManifold)` are genuine type piracy — the generics are ANN's and the types are GO's, so GML owns nothing in either signature | `GML/ext/HDF5Ext.jl` | **half fixed.** The `h5save` half is gone: GML's extension defines no `h5save` at all, and GO's `ext/NeuralNetworkParametersExt.jl` serves the types through the protocol. The `changebackend` half is open — see D3 |
| D9 | `Base.NamedTuple(p::NeuralNetworkParameters)` — Base's constructor and ANN's type, so the package defining it owns neither. Surfaced by GML [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207) | was `GML/src/layers/forcing_dissipation_layers.jl` | **fixed, phase 1** — `src/parameters.jl:162` |
| D10 | `h5save(::HDF5.Group, ::NeuralNetworkParameters, ::AbstractString)` — ANN's generic and ANN's type, from GML. Nothing existed for a parameter set nested at a path, which the parameter-dependent architectures produce. Also GML [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207) | was `GML/ext/HDF5Ext.jl` | **fixed** — `src/io.jl` owns the generic HDF5 path, so a nested parameter set serialises without anybody committing piracy |
| D11 | `GeometricOptimizers.GlobalSection(ps::NetworkParameters)` — `GlobalSection` is GO's and `NetworkParameters` is this package's, so GML owns neither. Not in the original survey; it appeared when GML adopted the container while GO had not | `GML/src/optimizers/optimizer.jl:5-6` | open. Fixed by step 3 of the GO work in §8, which gives it a home next to `GO/src/global_sections/global_sections.jl:29` |

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
| `src/parameters.jl` | `NetworkParameters`, `params`, and the `NamedTuple` forwarding |
| `src/leaves.jl` | `freeparameters`, `rebuild`, `parameter_metadata`, `parameter_eltype`, `isterminal`, `isparametertree` |
| `src/walk.jl` | `mapparameters`, `mapstorage`, the in-place and `foreach` forms, `foldparameters` |
| `src/layout.jl` | `ParameterLayout` and its five concrete cases, `parameterlayout`, `parameterrange`, `flatlength` |
| `src/flatten.jl` | `flatten`, `flatten!`, `unflatten`, `unflatten!`, and the Jacobian split |
| `src/flat_parameters.jl` | `FlatParameters`, the `AbstractVector` interface, layout-preserving `similar` and broadcasting |
| `src/io.jl` | the `h5save`/`h5load`/`save`/`load` generics and the type registry |
| `src/derivatives.jl` | `ChainRulesCore` rules for both conversions |
| `ext/HDF5Ext.jl` | the storage methods |
| `ext/ZygoteRulesExt.jl` | `pullback` on a `NetworkParameters`, D7's new home |

Design points worth keeping in view:

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
julia --project -e 'using Pkg; Pkg.test()'                                            # 255 tests
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'   # doctests
```

The suite covers, beyond round trips: `Float32` fidelity (D6); zero allocation for `flatten!` and
`unflatten!` — measured from inside a function, since a top-level `@allocated` reports a few tens of
bytes per leaf on Julia 1.10, where the recursion is not specialised; a structured leaf and a two-block
leaf whose types survive the round trip (D5); scalar, empty and tuple leaves; `Dual`-valued
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

### `AbstractNeuralNetworks` 0.7.0 — done

The container moved. `src/parameters.jl` and the whole `ext/` directory are gone; the package depends
on this one (`Project.toml:10`, compat `:19`) and imports the five names it extends or reaches through
(`src/AbstractNeuralNetworks.jl:32-33`). `changebackend` is one `mapparameters` call
(`src/utils/changebackend.jl:16-18`), which also gave it the `Tuple`-branch case the two methods it
replaces did not have; `_statify` is another (`src/static_cpu_backend.jl:30`). D7's generic `pullback`
moved here. `test/parameters_tests.jl` and `test/parameters_hdf5_tests.jl` became
`test/parameters_seam_tests.jl`, which tests the seam rather than the container.

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

### `SymbolicNeuralNetworks` 0.6.0 — done, and more than was asked

Earlier drafts said this package needed no change, on the grounds that its `flatten_equations` solves a
different problem. That was half right: restoring batch dimensions is still its own
(`unflatten_batch(::LeafLayout, ::AbstractMatrix)`, `src/codegen/equation_sets.jl:192`), but the
*layout* was not. `FlatSlice` is gone, and `flatten_equations` is one line over this package —
`flatten(Num, mapparameters(scalar_expressions, eqs))` (`:146`) — with `split_result` dispatching on
`ParameterLayout` (`:159-160`). The design point in §6 that called `ParameterLayout` a generalisation
of `FlatSlice` turned out to be the plan for a package this document said to leave alone.

### `GeometricOptimizers` 0.4.3 — the leaf protocol shipped

`ext/NeuralNetworkParametersExt.jl` supplies the whole protocol for GO's structured matrices: one
`freeparameters(x) = parent(x)` covering all three families, eight `rebuild` methods, a guard method on
the abstract types that raises rather than letting `rebuild(::AbstractArray, data) = data` hand back
bare storage for a subtype added later, `parameter_metadata`, and `register_parameter_type!` for all
eight types — including normalisation for the older layout GML used to write, which is the only place
that can do it, since it is where the types are.

**What remains.** GO still depends on `ParameterHandling` (`Project.toml:13`), still carries the
pirated Base-type `flatten` methods and the `Float64`-defaulting one-arg method
(`src/optimizers/named_tuple_wrapper.jl:12-66`), and still dispatches on `ArrayNamedTuple` — a bare
`NamedTuple` alias (`src/optimizer_solution.jl:48-52`) — rather than on a type it owns. The work, in
order:

**1. Preallocate the flat buffers.** A layout is a value, so it can live on the cache; the
`ParameterHandling` closure could not. Three sites flatten purely to obtain a length —
`iterative_hessians/bfgs/bfgs_cache.jl:33`, `.../dfp/dfp_cache.jl:37`, `.../bfgs/bfgs_state.jl:46` —
where `flatlength(x)` suffices and allocates nothing. The rest flatten per call, and become `flatten!`
into cache scratch: `outer!` (`bfgs_cache.jl:90-93`, and the lift form at `:109-112`),
`bfgs_cache.jl:161`, `dfp_cache.jl:113`, `bfgs_state.jl:91`, `_mul!` (`named_tuple_wrapper.jl:228-234`
and `:240-246`, twice per call each) and `_dot` (`:279-280`, both arguments per inner product, once per
line-search trial slope — the hottest of them). The write-back becomes `unflatten!`, which
`ParameterHandling` had no equivalent of at all.

**2. Drop `ParameterHandling`.** With the call sites moved,
`src/optimizers/named_tuple_wrapper.jl:12-93` goes: the one-arg method at `:14` (D6), the five
Base-type methods at `:30-66`, and the four GO-owned methods at `:16-23, 76-93`, which
`ext/NeuralNetworkParametersExt.jl` already supersedes. That retires `manifold_constructor`'s use in
flattening (D5) — `rebuild` takes a prototype, so the concrete type cannot drift — though
`manifold_constructor` itself still has a caller in `_similar` (`:126`). Then the dependency, the
import at `src/GeometricOptimizers.jl:49`, and the four docstrings that describe its behaviour
(`manifolds/abstract_manifold.jl:158,164`, `special_matrices/vector_storage_matrix.jl:18`,
`named_tuple_wrapper.jl:6`).

The ordering agrees leaf family by leaf family, which is what makes this behaviour-preserving:

| leaf | `ParameterHandling`, as GO wrote it | this package |
|---|---|---|
| plain array | `vec(x)` | `copyto!` in linear index order (`src/flatten.jl:103-104`) |
| `Manifold` | `flatten(T, x.A)` | `parent(x) = A.A` (`GO/…/abstract_manifold.jl:147`) — the same array |
| `VectorStorageMatrix` | `vec(s)` | `parent(x) = A.S`, and `Base.vec` *is* `.S` for all four (`symmetric.jl:297`, `skew_symmetric.jl:293`, `triangular.jl:124`) |
| `StiefelLieAlgHorMatrix` | `flatten(T, (g.A, g.B))` | `parent = (A.A, A.B)`, recursed in tuple order; `g.A` is a `SkewSymMatrix` and reduces to `.S` on both sides |
| `GrassmannLieAlgHorMatrix` | `flatten(T, g.B)` | `parent = (A.B,)` — the same numbers, one tuple level deeper |
| `NamedTuple` | `flatten(T, values(x))`, depth first | `NestedLayout` over `values`, depth first (`src/layout.jl:130-133`) |

So `GO/test/named_tuple_parameters.jl:21-22` keeps its hardcoded ranges and `∇F!` keeps indexing the
flat vector the same way. Pin it with an elementwise-equality test while both packages are still
present, before the deletion.

**3. Adopt `NetworkParameters`.** Everything funnels through `OptimizerSolution`
(`src/optimizer_solution.jl:59`), so this is one alias plus the methods behind it. Note that
`ArrayNamedTuple` is *flat*, so GO does not today accept a nested per-layer tree at all. Adding
`NetworkParameters` to the union is non-breaking, and most of the ~28 `ArrayNamedTuple` methods in
`named_tuple_wrapper.jl` are `apply_toNT` calls that become `mapparameters` — one method covering both
containers and nested branches, so the file shrinks rather than doubling. That also retires the
`apply_toNT` duplicated at `src/utils.jl:44`. Narrowing the piracy sites to `NetworkParameters`
afterwards is what closes the non-`flatten` half of issue #16:
`(grad::Gradient{T})(::ArrayNamedTuple{T})` (`named_tuple_wrapper.jl:97`),
`Base.copyto!(::GlobalSectionNamedTuple, ::ArrayNamedTuple)` (`:141`), `outer!`
(`bfgs_cache.jl:90`), `solution_scale` (`optimizer_status.jl:163`) and `l2norm` (`:185`).

One caveat: `l2norm(::AbstractMatrix)` and `l2norm(::AbstractFloat)` (`optimizer_status.jl:181-182`)
are piracy on *Base* types, and no parameter container fixes them. The comment at `:178-180` already
says where they belong, which is upstream in `GeometricBase`.

This is a breaking release — GO 0.5.0, by the convention its own `CHANGELOG.md:5-7` states — because
the `ParameterHandling.flatten` methods are observable, and GO's own tests call them
(`test/lie_algebras/grassmann_lie_algebra_horizontal.jl:86`,
`test/special_matrices/optimizer_primitives.jl:96`).

**Steps 1 and 2 are separable from step 3, and should ship first.** Nothing in them widens a
signature or moves an exported name, so they can go out on GO's own schedule — as a patch, even, since
the flat representation is unchanged. Step 3 cannot: it has to be co-released with GML.

**The trap in step 3.** GML does not hand a parameter set to GO's container machinery today. It walks
the tree itself (`GML/src/optimizers/optimizer.jl:252-264`) and calls GO's primitives per *leaf*
(`:206-243`), and which of the two happens is decided by `_use_go_cache`
(`:52-53`) — `x isa GeometricOptimizers.OptimizerSolution` — tested *before* the
`x isa NamedTuple || x isa NetworkParameters` recursion (`:56-58`, `:67-69`). Since
`NetworkParameters` is not in `OptimizerSolution` today, a whole network takes the recursion branch and
each *layer* — a bare `NamedTuple` of arrays, i.e. an `ArrayNamedTuple` — gets its own GO cache.

Adding `NetworkParameters` to `OptimizerSolution` inverts that: the top-level parameter set matches
`_use_go_cache` instead, the recursion never runs, one cache is built for the whole network, and
`_leaf_optim_step!` hands a `NetworkParameters` to `_GMLGradient`, which has methods only for
`ArrayNamedTuple` (`:21`) and `AbstractArray` (`:23`). That is a `MethodError` on the first training
step of every GML run — from a change that looks purely additive on GO's side. So step 3 needs a GML
release that adds `(g::_GMLGradient)(::NetworkParameters)` and then either drops `_tree_optim_step!` in
favour of the single whole-network cache, which is the better end state, or reorders the two tests.

Two smaller couplings to respect: `GeometricOptimizers.apply_toNT` is called qualified from
`GML/src/optimizers/optimizer.jl:19`, so the *name* has to survive even though GO stops using it
internally — reimplement it as `mapparameters` rather than delete it. And
`GML/src/optimizers/optimizer.jl:5-6` defines `GeometricOptimizers.GlobalSection(::NetworkParameters)`,
which is GO's function on this package's type: GML owns neither, so it is piracy of the same shape as
D8 (see D11). Step 3 is what gives it a legal home, next to
`GO/src/global_sections/global_sections.jl:29`.

**Two differences that are not ordering**, and are the only behaviour changes in step 2:
`ParameterHandling`'s `flatten(T, ::Vector{R})` returned an `unflatten` that converted back to `R`
(`GO/…/named_tuple_wrapper.jl:56-60`), where `unflatten` here keeps `eltype(v)` — identical for a
homogeneous set, which `ArrayNamedTuple{T}` guarantees, and a reason to prefer `unflatten!` on the hot
paths anyway, since writing through `copyto!` into an existing leaf cannot change its type. And an
empty set flattens to `Vector{Union{}}` rather than `T[]`, because `_promote_eltypes(()) = Union{}`
(`src/leaves.jl:204`); GO never constructs one. Since the container carries its element type, that
`Union{}` is visible on the type as well as in `flatten`'s output: an empty set is a
`NetworkParameters{Union{}}`, which binds and dispatches like any other.

`GO/test/named_tuple_parameters.jl` is the gate for all three steps: it covers the heterogeneous
`NamedTuple` (`:100-107`), `Float32` fidelity (`:109-123`) and manifold-kind preservation
(`:125-147`), which are D6 and D5 seen from GO's side. Its `ParameterHandling.flatten(ps)` calls become
`flatten`/`unflatten(layout, v)`; the assertions do not change.
`GO/test/neural_network_parameters_protocol.jl` already exercises the protocol and `flatlength`, so
extend that rather than starting a file. Worth adding for step 1: an `@allocated == 0` assertion on the
`_dot` and `_mul!` paths, measured from inside a function as `test/flatten_tests.jl:85-102` does here,
since that is the only way to know the preallocation landed.

### `GeometricMachineLearning` 0.6.0 — the HDF5 half shipped

`ext/HDF5Ext.jl` is 119 lines and defines no `h5save`: it is `save`/`load` entry points dispatching on
GML's own `NeuralNetwork`, delegating the traversal here. `_natural_sort_keys` is gone (D4), and so is
the pirated `h5save` (half of D8). The package depends on this one (`Project.toml:19`) and re-exports
`NetworkParameters` (`src/GeometricMachineLearning.jl:106`).

**What remains.** The five `changebackend` methods for GO's types are still inside the HDF5 extension
(`ext/HDF5Ext.jl:23-41`) — D3 and the other half of D8. `ext/HDF5Ext.jl:17-20` records why they are
still there and where they belong: a `GeometricOptimizers` extension on `AbstractNeuralNetworks`, which
is GO's release chain rather than GML's, so it is a separate change. Beyond that, `src/map_to_cpu.jl`
still hand-recurses with six per-type methods where one `mapparameters` call would do;
`src/utils.jl` still carries `apply_toNT` (`:31-36`, still exported at
`src/GeometricMachineLearning.jl:174`) and `_eltype` (`:225-237`); and the step dispatcher
`_tree_optim_step!` (`src/optimizers/optimizer.jl:252-264`) is one more hand-written recursion over the
same shape, which `foreachparameters` covers — including the `nothing`-branch skip it open-codes at
`:256`.

### `NonlinearIntegrators` 0.4.0 — a consumer

Not a target of any phase, but it pins both this package (`Project.toml:16`) and
`AbstractNeuralNetworks = "0.7"` (`:26`), and reaches for `NetworkParameters` directly
(`src/NonlinearIntegrators.jl:47-53`, used in `src/nvi/`). It is here so the version table above is
the whole set of packages a change to this one has to consider.
