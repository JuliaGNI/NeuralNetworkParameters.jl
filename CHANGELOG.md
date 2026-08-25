# Changelog

Notable changes to `NeuralNetworkParameters` are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — unreleased

### Fixed

- **The out-of-place `unflatten` allocated a closure per nesting level on Julia 1.10.** It was the one
  walk in the package written as `map(c -> unflatten(c, v), values(l.children))` rather than as the
  `Base.tail` recursion `src/walk.jl` states the house rule to be, and which `flatten!`, `unflatten!`
  and `mapparameters` all follow. Both are equally inferable — nothing here was ever type unstable —
  but `map` leaves the closure over the flat vector to be elided, and Julia 1.10 does not always elide
  it.

  `SymbolicNeuralNetworks` splits the result of a generated function into the shape of a parameter set
  through this function, once per call, so the cost landed on a hot path: 800 bytes per call on 1.10
  against 352 on 1.11 and later, which `NonlinearIntegrators` measured as 1.85x the allocations in its
  Newton residual and had to ship a version-conditional allocation ceiling for
  ([SymbolicNeuralNetworks#55](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/55)).
  Rewritten as `_unflatten_children`, it costs 512 on 1.10 and is unchanged on 1.11, 1.12 and 1.13 —
  what is left between the versions is Julia 1.10's `reshape`, which allocates 64 bytes where later
  versions allocate none.

  The same rewrite takes the walk off `Base`'s `Any32` fallback, which `map` drops to past 32 children
  and which returns a tuple with no concrete type — so a parameter set with more than 32 layers, or a
  layer with more than 32 entries, no longer unflattens through a type-unstable path.

  `parameterrange`, `length(::ParameterLayout)`, `_reshape_leaf` and the `parameter_eltype` family are
  marked `@inline` alongside, for the same reason the in-place walks are.

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
[0.2.1]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.1
[0.2.0]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.0
[0.1.1]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.1
[0.1.0]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.0
