# NeuralNetworkParameters

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/NeuralNetworkParameters.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNI.github.io/NeuralNetworkParameters.jl/dev/)
[![Build Status](https://github.com/JuliaGNI/NeuralNetworkParameters.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/NeuralNetworkParameters.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/NeuralNetworkParameters.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/NeuralNetworkParameters.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/N/NeuralNetworkParameters.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/N/NeuralNetworkParameters.html)

`NeuralNetworkParameters` holds the parameters of the networks defined in
[`AbstractNeuralNetworks`](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl),
[`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) and
[`SymbolicNeuralNetworks`](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl), in two shapes, with conversions between them.

## Installation

```julia
using Pkg
Pkg.add("NeuralNetworkParameters")
```

<!-- Delete this paragraph once the package has been registered via Register.yml. -->
`NeuralNetworkParameters` is not in the General registry yet. Until it is, install it with
`Pkg.develop(url = "https://github.com/JuliaGNI/NeuralNetworkParameters.jl")`, or point a `[sources]`
entry at a local checkout.

## Quickstart

`NetworkParameters` follows the architecture — a `NamedTuple` of `NamedTuple`s of arrays, one entry
per layer:

```julia
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0; 3.0 4.0], b = [5.0, 6.0]),
                        L2 = (W = [7.0 8.0], b = [9.0])))

ps.L1.W                 # 2×2 Matrix{Float64}: [1.0 2.0; 3.0 4.0]
```

`FlatParameters` is the same numbers as one vector, which is the shape a derivative, a linear solver
or a quasi-Newton method wants:

```julia
fp = FlatParameters(ps)

collect(fp)             # [1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
fp.L2.b                 # [9.0] — a layer read back off the flat form
```

Neither shape is derived from the other on the fly: `flatten` and `unflatten` convert between them
through a `ParameterLayout` that is built once and then reused. The layout is an ordinary value, so it
can be stored in an optimizer cache, handed to a solver, or compared with `==`.

### Differentiating with respect to the flat form

The point of the flat shape is that `ForwardDiff` — or any method that wants a vector — can work on
it, while the answer still comes back laid out like the network:

```julia
using ForwardDiff

v, layout = flatten(ps)
loss(p) = sum(model(x, p))

g = ForwardDiff.gradient(w -> loss(unflatten(layout, w)), v)
unflatten(layout, g)          # the gradient, one entry per layer
```

`unflatten` is generic in the element type of its vector, which is what makes this work: a
`Dual`-valued vector produces `Dual`-valued parameters. Reverse mode is covered too — there are
`ChainRulesCore` rules for both conversions, so `Zygote` can differentiate through them, and with
`ZygoteRules` loaded the gradient of a `NetworkParameters` is a `NetworkParameters` rather than a
tangent nobody can consume.

For inner loops there are in-place forms. `flatten!(v, ps, layout)` and `unflatten!(ps, layout, v)`
allocate nothing when called from compiled code, so an optimizer that flattens once per step need not
allocate a fresh vector each time.

### Structured parameters

A parameter need not be a plain array. A symmetric matrix keeps `n(n+1)/2` numbers behind an `n × n`
interface; a manifold element or a horizontal lift keeps its own. Those numbers — not the dense
entries — are what belongs in the flat vector, and what an optimizer should move. Two methods teach
the package about such a type, and flattening, the tree walks and HDF5 all follow:

```julia
NeuralNetworkParameters.freeparameters(A::SymmetricMatrix) = A.S
NeuralNetworkParameters.rebuild(A::SymmetricMatrix, data)  = SymmetricMatrix(data, A.n)
```

`GeometricOptimizers` already exposes exactly this relation as `Base.parent` for its manifolds,
`VectorStorageMatrix`es and horizontal lifts, so one delegating method covers all of them:

```julia
NeuralNetworkParameters.freeparameters(x::Union{Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix}) =
    parent(x)
```

Nothing in the package knows which structured types exist, which is the point: the types can live
upstream of the package that trains with them without anybody committing piracy.

See the [documentation](https://JuliaGNI.github.io/NeuralNetworkParameters.jl/dev/) for the full
picture: the two representations, the conversions and what a layout records, the leaf protocol, the
tree walks (`mapparameters`, `mapstorage`, `foreachparameters`, `foldparameters`), reading and writing
HDF5, and the library reference.

## Development

We are using git hooks, e.g., to enforce that all tests pass before pushing.
In order to activate these hooks, the following command must be executed once:
```
git config core.hooksPath .githooks
```

The test suite and the doctests are run with
```
julia --project -e 'using Pkg; Pkg.test()'
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'
```
