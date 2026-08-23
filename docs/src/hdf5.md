```@meta
CurrentModule = NeuralNetworkParameters
```

# Reading and writing HDF5

Loading `HDF5` brings in the storage methods through a package extension, so the binary chain behind
HDF5 is only paid for by users who ask for it. The four entry points are not exported — `save` and
`load` are names that would collide with almost anything — so they are reached as
`NeuralNetworkParameters.save`, or imported explicitly.

```@docs
save
load
h5save
h5load
```

## Round trip

```julia
using NeuralNetworkParameters, HDF5
using NeuralNetworkParameters: save, load

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
save("parameters.h5", ps)
load(NetworkParameters, "parameters.h5") == ps      # true
```

The element type is preserved: a `Float32` network reloads as `Float32`.

## Key order is recorded

HDF5 hands back the members of a group sorted, so a network with ten layers would read back as
`L1, L10, L2, …` — and since key order is part of the identity of a parameter set, that set no longer
compares equal to the one that was written. Guessing the order back from the names, by sorting on a
trailing integer, only works for names that happen to look like `L10`.

The writer therefore records the order explicitly, as an attribute on each group, and the reader uses
it. Files that predate this — see below — still fall back to the guess, because for them there is
nothing better available.

## Structured parameters

A structured leaf is written as a group holding its [`freeparameters`](@ref) and its
[`parameter_metadata`](@ref) — the latter only when there is any, so a type that keeps everything in
its storage, as a manifold element does, writes the storage alone. Reading it back needs to know how
to reconstruct the type, and there are two ways to supply that.

**Against a prototype.** Pass a parameter set of the right shape and the leaves are rebuilt with
[`rebuild`](@ref), exactly as unflattening does. Nothing needs registering. Where the architecture is
to hand — and when loading a network's parameters it usually is — this is the simpler path:

```julia
load(NetworkParameters, "parameters.h5", prototype)
```

**From the registry.** For a file that must be readable on its own, the package owning the type
registers a reconstructor:

```@docs
register_parameter_type!
parameter_metadata
parameter_type_name
```

```julia
NeuralNetworkParameters.parameter_metadata(A::SymmetricMatrix) = (n = A.n,)

register_parameter_type!("SymmetricMatrix", (S, md) -> SymmetricMatrix(S, md.n))
```

Loading a file with an unregistered type raises an error naming the type and both remedies, rather than
failing obscurely.

## Older files

Parameter files written before this package still load:

- those written by `AbstractNeuralNetworks`, which are plain nested groups with no attributes;
- those written by `GeometricMachineLearning`, whose structured matrices carry a `gml_type` attribute.
  Such a group holds the type's fields under their own names and no `storage` for [`rebuild`](@ref) to
  take, so the registry is the only way in: the type has to have been registered, and a prototype is
  no substitute. Passing one says so rather than failing obscurely.

The group's fields reach the registered reconstructor in both argument positions, storage and
metadata alike, since a file in that layout records nothing that could tell them apart. A
reconstructor that means to read these files sorts that out itself — and reaches for its fields by
name, the layout having recorded no key order either.
