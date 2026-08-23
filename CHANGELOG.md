# Changelog

Notable changes to `NeuralNetworkParameters` are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows
[semantic versioning](https://semver.org/spec/v2.0.0.html).

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
