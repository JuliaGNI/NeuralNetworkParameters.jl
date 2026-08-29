"""
    NeuralNetworkParameters

The parameters of a neural network, in two shapes and with conversions between them.

- [`NetworkParameters`](@ref) is the structured shape: a `NamedTuple` of `NamedTuple`s of arrays
  following the architecture, wrapped in a type of its own so that the parameter set is something a
  package can dispatch on without piracy.
- [`FlatParameters`](@ref) is the flat shape: one `AbstractVector` of every number in the set, which
  is what a derivative, a linear solver or a quasi-Newton method wants to work with.

[`flatten`](@ref) and [`unflatten`](@ref) convert between them through a [`ParameterLayout`](@ref)
that is built once and reused, so a derivative taken with respect to the flat vector comes back in the
shape of the network:

```julia
v, layout = flatten(ps)
g = ForwardDiff.gradient(w -> loss(unflatten(layout, w)), v)
unflatten(layout, g)          # the gradient, laid out like `ps`
```

Structured leaves — a symmetric matrix keeping ``n(n+1)/2`` numbers behind an ``n \\times n``
interface, a manifold element, a horizontal lift — plug in through two methods,
[`freeparameters`](@ref) and [`rebuild`](@ref). Everything else in the package, including the HDF5
support, is written against that protocol rather than against a list of types.

Loading `HDF5` brings in [`save`](@ref) and [`load`](@ref) through a package extension.

`GeometricBase.L2norm` of a whole parameter set is here rather than behind an extension, because
most of this ecosystem depends on `GeometricBase` already and its own sole dependency is `Unicode`.
See [`L2norm`](@ref).
"""
module NeuralNetworkParameters

using ChainRulesCore

# `import` and not `using`: these two are `GeometricBase`'s functions and `src/norms.jl` adds a method
# to one of them. `l2norm` is imported as well because that is what the fold calls at each leaf, and a
# downstream package's method for a structured leaf is the one that has to be reached.
import GeometricBase.Utils: L2norm, l2norm

export NetworkParameters, params

include("parameters.jl")

export freeparameters, rebuild, parameter_metadata, parameter_eltype

include("leaves.jl")

export mapparameters, mapparameters!, mapstorage, mapstorage!, foreachparameters,
       foldparameters, foldstorage

include("walk.jl")

export ParameterLayout, parameterlayout, parameterrange, flatlength

include("layout.jl")

export flatten, flatten!, unflatten, unflatten!

include("flatten.jl")

export FlatParameters, flatlayout

include("flat_parameters.jl")

export register_parameter_type!

include("io.jl")

include("norms.jl")

include("derivatives.jl")

end
