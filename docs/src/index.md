```@meta
CurrentModule = NeuralNetworkParameters
```

# NeuralNetworkParameters

The parameters of a neural network, in two shapes, with conversions between them.

The package serves the networks defined in
[AbstractNeuralNetworks](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl),
[GeometricMachineLearning](https://github.com/JuliaGNI/GeometricMachineLearning.jl) and
[SymbolicNeuralNetworks](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl), and holds nothing but
the parameters: it depends on none of them, and adds one dependency of its own
(`ChainRulesCore`).

## The two shapes

[`NetworkParameters`](@ref) follows the architecture — a `NamedTuple` of `NamedTuple`s of arrays, one
entry per layer:

```jldoctest shapes
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0; 3.0 4.0], b = [5.0, 6.0]),
                        L2 = (W = [7.0 8.0], b = [9.0])))
ps.L1.W

# output

2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0
```

[`FlatParameters`](@ref) is the same numbers as one vector, which is the shape a derivative, a linear
solver or a quasi-Newton method wants:

```jldoctest shapes
fp = FlatParameters(ps)
collect(fp)

# output

9-element Vector{Float64}:
 1.0
 3.0
 2.0
 4.0
 5.0
 6.0
 7.0
 8.0
 9.0
```

Neither is derived from the other on the fly: [`flatten`](@ref) and [`unflatten`](@ref) convert
between them through a [`ParameterLayout`](@ref) that is built once and then reused.

## Differentiating with respect to the flat form

The point of the flat shape is that `ForwardDiff` — or any method that wants a vector — can work on
it, while the answer still comes back laid out like the network:

```julia
using ForwardDiff

v, layout = flatten(ps)
loss(p) = sum(model(x, p))

g = ForwardDiff.gradient(w -> loss(unflatten(layout, w)), v)
unflatten(layout, g)          # the gradient, one entry per layer
```

[`unflatten`](@ref) is generic in the element type of its vector, which is what makes this work:
a `Dual`-valued vector produces `Dual`-valued parameters. Reverse mode is covered too — there are
`ChainRulesCore` rules for both conversions, so `Zygote` can differentiate through them, and with
`ZygoteRules` loaded the gradient of a `NetworkParameters` is a `NetworkParameters` rather than a
tangent nobody can consume.

## Structured parameters

A parameter need not be a plain array. A symmetric matrix keeps ``n(n+1)/2`` numbers behind an
``n \times n`` interface; a manifold element or a horizontal lift keeps its own. Those numbers — not
the dense entries — are what belongs in the flat vector, and what an optimizer should move.

Two methods teach the package about such a type, and everything else follows: flattening, the tree
walks, and HDF5. See [The leaf protocol](@ref).

## Pages

```@contents
Pages = ["representations.md", "conversions.md", "protocol.md", "walks.md", "hdf5.md", "library.md"]
Depth = 1
```
