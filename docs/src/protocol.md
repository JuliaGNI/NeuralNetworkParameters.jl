```@meta
CurrentModule = NeuralNetworkParameters
```

# The leaf protocol

Not every parameter is a plain array. `GeometricOptimizers` has a family of matrices that keep
``n(n\pm1)/2`` numbers behind an ``n \times n`` interface, manifold elements, and horizontal lifts
that keep their freedom in two blocks. Those stored numbers are the parameters: they are what belongs
in the flat vector, and what an optimizer should move.

Two methods say so.

```@docs
freeparameters
rebuild
```

With them defined, the type flattens, walks and saves — nothing in this package holds a list of which
structured types exist. That matters because the types live *upstream* of the package that trains with
them: `GeometricOptimizers` owns them, `GeometricMachineLearning` uses them, and a serialiser driven by
a list rather than a protocol would have to be written by somebody who owns neither the types nor the
generic it dispatches on.

## An example

```jldoctest protocol
using NeuralNetworkParameters
import NeuralNetworkParameters as NNP

struct Sym{T} <: AbstractMatrix{T}
    S::Vector{T}      # n(n+1)/2 numbers
    n::Int
end

Base.size(A::Sym) = (A.n, A.n)
Base.getindex(A::Sym, i::Int, j::Int) = A.S[(max(i,j)*(max(i,j)-1))÷2 + min(i,j)]

NNP.freeparameters(A::Sym) = A.S
NNP.rebuild(A::Sym, data) = Sym(data, A.n)

ps = NetworkParameters((L1 = (S = Sym([1.0, 2.0, 3.0], 2),),))
v, layout = flatten(ps)
v

# output

3-element Vector{Float64}:
 1.0
 2.0
 3.0
```

Three numbers, not four: the flat vector holds the free parameters, and the round trip returns the same
type, with its `n` intact.

```jldoctest protocol
back = unflatten(layout, [10.0, 20.0, 30.0])
(typeof(back.L1.S) === typeof(ps.L1.S), back.L1.S.n)

# output

(true, 2)
```

## Why `rebuild` takes a prototype

`rebuild` is handed the leaf the storage came from, not just its type. That carries the
non-differentiable information across — the `n` above — and it is what lets `data` have a different
element type from the prototype, as it does under forward-mode differentiation.

It also keeps the concrete type honest. Reconstructing from a type name instead has a failure mode
that has actually been hit upstream: a hardcoded reconstructor turned every manifold element into a
`StiefelManifold`, quietly converting a `GrassmannManifold` on each round trip.

## Adopting it for a whole family at once

`GeometricOptimizers` already exposes this relation, as `Base.parent`: `parent` of its
`VectorStorageMatrix`es is the vector they store, of a `Manifold` its matrix, of a horizontal lift the
tuple of its blocks. So one method covers the family:

```julia
NeuralNetworkParameters.freeparameters(
    x::Union{Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix}) = parent(x)
```

`Base.parent` is deliberately not used as the protocol here. `parent` of a `SubArray` is the whole
underlying buffer rather than the view's own entries, so Base's relation and this one do not agree in
general, and flattening on `parent` would silently take in far too much.

## Multi-block storage

`freeparameters` may return a `Tuple` or `NamedTuple`, for a type whose freedom lives in several
places; the recursion simply continues into it, and the blocks may be structured themselves.

```julia
NNP.freeparameters(g::StiefelLieAlgHorMatrix) = (A = g.A, B = g.B)
NNP.rebuild(g::StiefelLieAlgHorMatrix, data)  = StiefelLieAlgHorMatrix(data.A, data.B, g.N, g.n)
```

## Element type

```@docs
parameter_eltype
```

[`flatten`](@ref) uses this, so a `Float32` network flattens to a `Vector{Float32}`. Defaulting to
`Float64` instead would silently double the width of every single-precision network passing through,
and of everything computed from the flat vector afterwards.
