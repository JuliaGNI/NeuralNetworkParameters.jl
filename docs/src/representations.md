```@meta
CurrentModule = NeuralNetworkParameters
```

# The two representations

## Structured

```@docs
NetworkParameters
params
```

`NetworkParameters` wraps a `NamedTuple` and forwards `getproperty`, `getindex`, `keys`, `values`,
`length`, `iterate` and `pairs` to it, so it reads like the `NamedTuple` it holds.

The wrapper is not decoration. A bare `NamedTuple` belongs to `Base`, so any package wanting to give a
parameter set its own behaviour — saving it, flattening it, stepping an optimizer over it — must write
methods whose every argument type is somebody else's. That is type piracy, and two packages doing it
can silently disagree about the same call. Owning the type removes the problem at the root.

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
