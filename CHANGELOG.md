# Changelog

Notable changes to `NeuralNetworkParameters` are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-08-24

### Changed

- **`NetworkParameters` carries the element type of its leaves as its first type parameter**, so that
  `T` binds in a method signature: `f(ps::NetworkParameters{T}) where {T}`, or a `Union` with
  `AbstractVector{T}`. `GeometricOptimizers` takes the element type from the *type* of the solution it
  is handed — eleven sites spell it `where {T, VT<:OptimizerSolution{T}}`, across every cache and
  state constructor, both `Optimizer` constructors and the `BFGSState` `update!` methods — and a
  parameter set could not join that union while it carried no such parameter.

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
  rather than a `MethodError` about a conversion nobody asked for.
- **`parameter_eltype` is total, where `freeparameters` is not.** A leaf with no protocol reports
  `eltype(x)` rather than raising, and a gap in a gradient tree — `nothing`, or a `ChainRules`
  structural zero — contributes nothing to the promotion. Every parameter set now runs its constructor
  through this function, including sets that hold no numbers at all: a gradient tree with `nothing`
  where an untouched layer's entries would be, or `SymbolicNeuralNetworks` wrapping generated
  functions in one. A set that cannot be flattened is still a set with an element type; the protocol
  error comes from `parameterlayout`, which is where it decides something.

  A leaf that keeps numbers behind a non-array interface opts into the recursion with one more method,
  `parameter_eltype(x::MyLeaf) = parameter_eltype(freeparameters(x))`. Every structured leaf in the
  ecosystem is an `AbstractArray` subtype and needs nothing.
- `parameter_eltype(ps::NetworkParameters)` reads the type parameter instead of walking the leaves, so
  `flatten(ps)` and the `flatten` rrule are cheaper than they were.

### Added

- `NetworkParameters{T, Keys, ValueTypes}(nt)` and `NetworkParameters{T}(nt)` are written out, since
  the derived element type makes the one that computes it an inner constructor and suppresses the
  defaults. The three-parameter form is not optional: `ChainRulesCore.construct` calls it from
  `+(::P, ::Tangent{P})`, which is how a parameter set and a cotangent add.
- `parameter_eltype` methods for `nothing` and `ChainRulesCore.AbstractZero`, both `Union{}`.

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
[0.1.1]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.1
[0.1.0]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.0
