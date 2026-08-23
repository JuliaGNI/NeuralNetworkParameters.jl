# Stand-ins for the structured parameter types that live downstream, so that the protocol is tested
# without depending on GeometricOptimizers.
#
# `Sym` mirrors `SymmetricMatrix`: n(n+1)/2 numbers behind an n×n interface, and an `n` that is *not*
# part of the differentiable storage. `TwoBlock` mirrors `StiefelLieAlgHorMatrix`: freedom in two
# blocks, one of them structured itself. `Manifold` mirrors `StiefelManifold`: the storage is the whole
# matrix and there is nothing else to carry, so it has no metadata.

import NeuralNetworkParameters as NNP

struct Sym{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int
end

Base.size(A::Sym) = (A.n, A.n)

function Base.getindex(A::Sym, i::Int, j::Int)
    p, q = max(i, j), min(i, j)
    A.S[((p - 1) * p) ÷ 2 + q]
end

NNP.freeparameters(A::Sym) = A.S
NNP.rebuild(A::Sym, data) = Sym(data, A.n)
NNP.parameter_metadata(A::Sym) = (n = A.n,)

struct TwoBlock{T, ST} <: AbstractMatrix{T}
    A::ST
    B::Matrix{T}
    N::Int
end

TwoBlock(A::Sym{T}, B::Matrix{T}, N::Int) where {T} = TwoBlock{T, typeof(A)}(A, B, N)

Base.size(g::TwoBlock) = (g.N, g.N)
Base.getindex(::TwoBlock{T}, ::Int, ::Int) where {T} = zero(T)

NNP.freeparameters(g::TwoBlock) = (A = g.A, B = g.B)
NNP.rebuild(g::TwoBlock, data) = TwoBlock(data.A, data.B, g.N)
NNP.parameter_metadata(g::TwoBlock) = (N = g.N,)

struct Manifold{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
end

Base.size(Y::Manifold) = size(Y.A)
Base.getindex(Y::Manifold, i::Int, j::Int) = Y.A[i, j]

NNP.freeparameters(Y::Manifold) = Y.A
NNP.rebuild(::Manifold, data) = Manifold(data)
# and no `parameter_metadata`: the default, an empty `NamedTuple`, is the whole truth about this one

"A three-number symmetric matrix and the five-number two-block lift built on it."
sample_sym() = Sym([1.0, 2.0, 3.0], 2)
sample_twoblock() = TwoBlock(Sym([4.0, 5.0, 6.0], 2), [7.0 8.0], 3)

"A four-number matrix that is its own storage."
sample_manifold() = Manifold([1.0 2.0; 3.0 4.0])

"A parameter set with an ordinary layer, a structured leaf and a multi-block leaf: 3 + 3 + 5 numbers."
function sample_parameters()
    NetworkParameters((L1 = (W = [10.0 20.0], b = [30.0]),
        L2 = (S = sample_sym(),),
        L3 = (G = sample_twoblock(),)))
end
