```@meta
CurrentModule = NeuralNetworkParameters
```

# The two representations

## Structured

```@docs
NetworkParameters
params
```

A whole set of parameters is a `NetworkParameters` and nothing else. A *branch* of one — a layer — is
the plain `NamedTuple` it wraps, and that is a different question, answered by
[`isparametertree`](@ref): it is what the walks recurse into, and its domain also admits a `Tuple`,
which is what [`freeparameters`](@ref) returns for a multi-block leaf.

There is deliberately no alias unioning the two. One would be a method on `Base.NamedTuple` wherever
it were used — a type nobody owns, colliding with every other `NamedTuple` alias in the same method
table — and it would say the same thing about a whole set and about a branch. Where a function
genuinely takes both, it has a method for each.

`NetworkParameters{T}` wraps a `NamedTuple` and forwards `getproperty`, `getindex`, `keys`, `values`,
`length`, `iterate` and `pairs` to it, so it reads like the `NamedTuple` it holds. `NamedTuple(ps)`
unwraps it again, and is the same thing as [`params`](@ref); the conversion is defined here rather
than downstream because a package that owns neither `Base.NamedTuple` nor the type cannot define it
without committing piracy on both counts.

The wrapper is not decoration. A bare `NamedTuple` belongs to `Base`, so any package wanting to give a
parameter set its own behaviour — saving it, flattening it, stepping an optimizer over it — must write
methods whose every argument type is somebody else's. That is type piracy, and two packages doing it
can silently disagree about the same call. Owning the type removes the problem at the root.

The `T` is the element type the leaves promote to, carried on the type so that a method signature can
bind it — `f(ps::NetworkParameters{T}) where {T}`, or a `Union` with `AbstractVector{T}`. A set is
built from its keys and values with `NetworkParameters(NamedTuple{keys}(vals))`, since the braces name
the element type rather than the keys, and it is derived rather than chosen: writing
`NetworkParameters{T}(nt)` asserts `T` and raises if the leaves say otherwise. See
[`parameter_eltype`](@ref) for what the promotion does and does not guarantee.

Note that key *order* is part of the identity of a parameter set:

```jldoctest
using NeuralNetworkParameters

NetworkParameters((a = [1.0], b = [2.0])) == NetworkParameters((b = [2.0], a = [1.0]))

# output

false
```

which is why the HDF5 writer records it explicitly — see [Reading and writing HDF5](@ref).

## Flat

```@docs
FlatParameters
flatlayout
```

`FlatParameters` is an `AbstractVector`, so it works with anything that takes one; `parent` gives the
bare `Vector` for code that would rather not see a wrapper.

`similar` deliberately *keeps* the layout, so scratch space derived from a flat parameter set — a
gradient, a momentum buffer — stays self-describing:

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
fp = FlatParameters(ps)
g = similar(fp)
flatlayout(g) == flatlayout(fp)

# output

true
```

Individual layers can be read back off the flat form, which is mostly useful interactively:

```jldoctest
using NeuralNetworkParameters

fp = FlatParameters(NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),)))
fp.L1.b

# output

1-element Vector{Float64}:
 3.0
```
