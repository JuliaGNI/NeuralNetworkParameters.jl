# NeuralNetworkParameters.jl — analysis and plan

Analysis last revised 2026-08-19, against these working trees:

| package | version | commit |
|---|---|---|
| `NeuralNetworkParameters` | 0.1.0 | this repository |
| `AbstractNeuralNetworks` | 0.6.4 | `0a3119f` |
| `SymbolicNeuralNetworks` | 0.5.0 | `7e4610a` |
| `GeometricMachineLearning` | 0.5.0-DEV | `e1d439cc` |
| `GeometricOptimizers` | 0.4.0 | `d27c0d2` |

Paths written `ANN/…`, `SNN/…`, `GML/…`, `GO/…` are relative to the corresponding sibling checkout.

**Status: Phase 1 is implemented** — the package below exists and its suite passes (226 tests, plus
doctests). Phases 2 and 3 are not started; they are breaking changes to released packages and belong in
their own pull requests.

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

The parameter type lived *downstream*, in `ANN/src/parameters.jl`, and the rest of this functionality
was implemented separately in `GeometricOptimizers` and `GeometricMachineLearning`. Three things drive
consolidation.

**The same traversal is written many times over.** Recurse into the `NamedTuple`s, do something at each
leaf, put it back: that is `flatten`/`unflatten` (`GO/src/optimizers/named_tuple_wrapper.jl`, via
`ParameterHandling`), `h5save`/`h5load` (`GML/ext/HDF5Ext.jl`, `ANN/ext/HDF5Ext.jl`), `changebackend`
(`ANN/src/utils/changebackend.jl` and `GML/ext/HDF5Ext.jl:59-77`), `map_to_cpu`
(`GML/src/map_to_cpu.jl`), `_statify` (`ANN/src/static_cpu_backend.jl:35-42`), the optimizer's
elementwise primitives, the cache/state builders and step dispatcher in
`GML/src/optimizers/optimizer.jl`, and `_eltype`/`apply_toNT` (`GML/src/utils.jl`, duplicated in
`GO/src/utils.jl`). Each re-declares a method per structured type. GML's own changelog for 0.5.0 says
what remains of GML is *"walking a `NeuralNetworkParameters` tree, and the manifold layer types"* — the
first half of that sentence is this package's job.

**A parameter set needs to be a type somebody owns.** Most of the type piracy `GeometricOptimizers`
issue #16 tracks is not about flattening: the container is a bare `NamedTuple` (aliased
`ArrayNamedTuple`), so `outer!` (`GO/…/bfgs_cache.jl`), `(grad::Gradient)(::ArrayNamedTuple)`
(`GO/…/named_tuple_wrapper.jl:86-90`), `Base.copyto!` (`:130-133`) and two sites in
`GO/src/optimizers/optimizer_status.jl` all dispatch on types nobody owns. Its comments ask for *"a
wrapper struct"*. `NetworkParameters` is that struct.

**`ParameterHandling` cannot express these parameters.** See §7.

## 3. What changed over the course of the analysis

Recorded because it invalidated earlier drafts of this document, and because the same thing may happen
again.

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

## 4. Defects on record

D1 and D2 are fixed. The rest are noted so they are not lost; the phase column says where each belongs.

| # | defect | location | phase |
|---|---|---|---|
| D1 | `Aversion = "0.1.0"` typo — the package had **no `version` field at all** | `Project.toml:4` | **1, fixed** |
| D2 | Stale `Manifest.toml` pinning `AbstractNeuralNetworks v0.3.0` | `Manifest.toml` | **1, fixed** |
| D3 | `changebackend` for the five wrapper types is defined **only inside GML's HDF5 extension**, so `changebackend(GPU(), nn)` on a `PSDLayer`/`LASympNet` network MethodErrors unless HDF5 happens to be loaded | `GML/ext/HDF5Ext.jl:59-77` | 3 |
| D4 | HDF5 returns group members sorted, worked around by a regex "natural sort" that silently falls back to lexicographic order unless *every* key matches `^\D+\d+$` | `GML/ext/HDF5Ext.jl:87-97` | **1 for new files**; 3 |
| D5 | `unflatten` rebuilds a manifold via `Base.typename(typeof(x)).wrapper`; the comment records that hardcoding `StiefelManifold` previously turned a `GrassmannManifold` into one on every round trip | `GO/…/named_tuple_wrapper.jl:18-24` | **1 by design**; 3 |
| D6 | `ParameterHandling.flatten` defaults to `Float64`, silently promoting `Float32` networks | `GO/…/named_tuple_wrapper.jl:12-14` | **1 by design**; 3 |
| D7 | With the type moved out of ANN, `ZygoteRules.pullback(f::Function, ::NeuralNetworkParameters)` owns neither argument type and becomes piracy there | `ANN/src/pullback_for_applychain.jl:10-17` | **1 provides the new home**; 2 |
| D8 | Introduced by the separation: GML's `h5save(::H5DataStore, ::StiefelManifold, ::AbstractString)` and `changebackend(::NeuralNetworkBackend, ::StiefelManifold)` are now genuine type piracy — the generics are ANN's and the types are GO's, so GML owns nothing in either signature | `GML/ext/HDF5Ext.jl:17-50, 59-77` | 3 |
| D9 | `Base.NamedTuple(p::NeuralNetworkParameters)` — Base's constructor and ANN's type, so the package defining it owns neither. Surfaced by GML [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207), which needed it to write a nested parameter set to HDF5 | `GML/src/layers/forcing_dissipation_layers.jl` | **1 provides the home**; 2 |
| D10 | `h5save(::HDF5.Group, ::NeuralNetworkParameters, ::AbstractString)` — ANN's generic and ANN's type, from GML. ANN's own extension has `h5save(::H5DataStore, ::NamedTuple, …)` and `save(::H5DataStore, ::NeuralNetworkParameters)`, but nothing for a parameter set nested at a path, which the parameter-dependent architectures produce. Also GML [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207) | `GML/ext/HDF5Ext.jl` | 2 |

D9 is one line of Phase 1 work that is not there yet: `src/parameters.jl` forwards `keys`, `values`,
`getindex` and `pairs`, and has `params`, but no `Base.NamedTuple(p::NetworkParameters) = params(p)`.
Adding it makes the conversion something this package owns, and Phase 2 makes it reachable from the
`NeuralNetworkParameters` alias. D10 is the same shape one level up: with `src/io.jl` owning the
generic HDF5 path, a nested parameter set serialises without anybody committing piracy.

D8 is the sharpest argument for the protocol. With the structured types upstream of the package that
trains with them, a *generic* HDF5 path driven by `freeparameters`/`rebuild` is the only way to
serialise them without somebody committing piracy.

**Open question for Phase 3.** GO's `flatten(T, ::VectorStorageMatrix)`
(`GO/…/named_tuple_wrapper.jl:76`) now round-trips through free storage, but
`GO/test/named_tuple_parameters.jl:20` asserts specific flat ranges. Whether this package's leaf
ordering matches GO's has to be checked against that test before GO adopts.

## 5. Evidence

Verified by running code rather than by reading, so it can be rechecked.

**A package cannot export a type sharing its own name.** The module binding wins at the `using` site:
a scratch package `Foo` exporting `struct Foo` gives `using Foo; Foo(1)` →
`ERROR: objects of type Module are not callable`. (`Foo.Foo` *is* the struct; the clash is only in the
importer's namespace.) Hence the type is `NetworkParameters`, and Phase 2 supplies
`const NeuralNetworkParameters = NetworkParameters` for compatibility.

**Flattening is cheap.** 1.3M parameters, five 512-wide dense layers, `Float64`: `flatten` 0.144 ms,
`flatten!` into a preallocated buffer 0.133 ms, `unflatten` 0.149 ms — against **0.946 ms for one
forward pass at batch 32**. A round trip is about 10 % of a single forward pass.

**Reshaped views keep the BLAS fast path**, so the argument against shared storage is *element type*,
not speed: `reshape(view(v, r), m, n)` is a `StridedArray` and `mul!` runs at 0.089 ms either way. An
earlier draft had this backwards. The decisive constraint is that `unflatten` must be able to return
arrays of a different element type from the parameters — `ForwardDiff.Dual`, noted at
`GO/…/named_tuple_wrapper.jl:19-22` — and a `Float64` buffer cannot be shared with a `Dual` view. Hence
copy semantics, with allocation-free in-place variants for inner loops.

## 6. Phase 1, as built

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
  the concrete difference from `ParameterHandling`, and generalises `FlatSlice` from
  `SNN/src/codegen/equation_sets.jl:117-161`.
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
julia --project -e 'using Pkg; Pkg.test()'                                            # 226 tests
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'   # doctests
```

The suite covers, beyond round trips: `Float32` fidelity (D6); zero allocation for `flatten!` and
`unflatten!` — measured from inside a function, since a top-level `@allocated` reports a few tens of
bytes per leaf on Julia 1.10, where the recursion is not specialised; a structured leaf and a two-block leaf whose types survive the round trip (D5); scalar,
empty and tuple leaves; `Dual`-valued unflattening; agreement between `ForwardDiff` on the flat form and
`Zygote` on the structured one; Zygote through both conversions; a structurally-zero gradient block; the
`nothing`-branch skip in the walks; ten-layer HDF5 key ordering (D4); both HDF5 load paths; and files in
the two older formats.

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

## 8. Follow-up phases

**Phase 2 — `AbstractNeuralNetworks` 0.6.4.** Depend on this package; delete `src/parameters.jl` and
`ext/HDF5Ext.jl`; add `const NeuralNetworkParameters = NetworkParameters` and re-export; rewrite
`changebackend` (`src/utils/changebackend.jl:17-19`) and `_statify`
(`src/static_cpu_backend.jl:35-42`) as `mapparameters` calls; move the generic `ZygoteRules.pullback`
method out (D7 — it is already implemented here); move `test/parameters_tests.jl` and
`test/parameters_hdf5_tests.jl` down. Gate on the SNN and GML suites. The compatibility alias has to
support every downstream use found in the survey: `NeuralNetworkParameters{keys}(vals)`
(`ANN/src/chain.jl:52`), `PT <: NeuralNetworkParameters` (`ANN/src/neural_network.jl:12`),
`x isa NeuralNetworkParameters` (`GML/src/optimizers/optimizer.jl`), and
`NeuralNetworkParameters{keys(...)}(...)` in `SNN/src/derivatives/derivative.jl:28-43` and
`SNN/src/codegen/equation_sets.jl:151-189`.

**Phase 3 — `GeometricOptimizers` 0.4.0.** Drop `ParameterHandling`. Delete the pirated Base-type
`flatten` methods (`named_tuple_wrapper.jl:30-66`) and the `Float64`-defaulting one-arg method (line 14).
Add `freeparameters(x) = parent(x)` for the family plus per-type `rebuild`. Adopt `NetworkParameters` as
the parameter container, which is what retires the non-`flatten` half of issue #16. Preallocate the flat
buffers in `_dot` (flattens both arguments per inner product), `_mul!` (twice per call), `outer!`, and
the BFGS/DFP caches — including `bfgs_cache.jl:33`, `dfp_cache.jl:37` and `bfgs_state.jl:46`, which
`flatten(_zero(x))` purely to obtain a length, where `flatlength(x)` suffices. Check the flat ordering
against `test/named_tuple_parameters.jl:20` first.

**Phase 3 — `GeometricMachineLearning` 0.5.0-DEV.** Reduce `ext/HDF5Ext.jl` to `register_parameter_type!`
calls, deleting the pirating `h5save`/`changebackend` methods (D8) and `_natural_sort_keys` (D4); move
`changebackend` out of the HDF5 extension (D3); rewrite `src/map_to_cpu.jl` as one `mapparameters` call;
replace `_eltype` and `apply_toNT` (`src/utils.jl`). The wrapper-type methods belong in GO now, not GML,
since GO owns the types.

**Registration is now on the critical path.** GML
[#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207) — parametric generalized
Hamiltonian neural networks — is the first consumer of `flatten`/`unflatten` outside
`GeometricOptimizers`. It flattens the parameters of the *system* into the network input, and reached
for `ParameterHandling` first; that fails, because GO's pirated `flatten(x)`
(`GO/…/named_tuple_wrapper.jl:14`) has an unbound type parameter and is the method that wins — D6 and
§7.1, hit in practice. GML depends on this package instead, so it cannot merge until this package is
in the General registry: GML supports `julia = "1.10"`, where `[sources]` in `Project.toml` does not
exist.

**`SymbolicNeuralNetworks` 0.5.0** needs no change. Its `flatten_equations`/`unflatten`
(`src/codegen/equation_sets.jl`) solves a different problem — flattening symbolic equation sets and
restoring batch dimensions (`unflatten(::FlatSlice, ::AbstractMatrix)` and the 3-array method at lines
194-201). Revisit only if `ParameterLayout` turns out to subsume `FlatSlice` cleanly.
