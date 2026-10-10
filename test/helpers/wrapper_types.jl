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

# `Padded` stands in for no downstream type. It keeps the same three numbers as `Sym` behind *five*
# fields of metadata rather than one, and it is here so that the reverse pass can be asked whether
# finding the storage among a leaf's fields costs anything per field — a question a single field count
# cannot answer.
struct Padded{T} <: AbstractVector{T}
    a::Int
    b::Int
    c::Int
    d::Int
    e::Int
    S::Vector{T}
end

Base.size(x::Padded) = size(x.S)
Base.getindex(x::Padded, i::Int) = x.S[i]

NNP.freeparameters(x::Padded) = x.S
NNP.rebuild(x::Padded, data) = Padded(x.a, x.b, x.c, x.d, x.e, data)
NNP.parameter_metadata(x::Padded) = (a = x.a, b = x.b, c = x.c, d = x.d, e = x.e)

struct Manifold{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
end

Base.size(Y::Manifold) = size(Y.A)
Base.getindex(Y::Manifold, i::Int, j::Int) = Y.A[i, j]

NNP.freeparameters(Y::Manifold) = Y.A
NNP.rebuild(::Manifold, data) = Manifold(data)
# and no `parameter_metadata`: the default, an empty `NamedTuple`, is the whole truth about this one

# `Lift` mirrors `StiefelLieAlgHorMatrix` more closely than `TwoBlock`: a dense interface
# `[A -Bᵀ; B 0]` over a `Sym` block and a plain one, its storage the two blocks in order. Its storage
# gradient, defined beside `Sym`'s in `test/derivatives.jl`, converts the `Sym` block itself, as
# GeometricOptimizers' lift does.
struct Lift{T, ST <: Sym{T}} <: AbstractMatrix{T}
    A::ST
    B::Matrix{T}
    N::Int
end

Base.size(g::Lift) = (g.N, g.N)

function Base.getindex(g::Lift{T}, i::Int, j::Int) where {T}
    n = g.A.n
    i ≤ n && j ≤ n && return g.A[i, j]
    i ≤ n && return -g.B[j - n, i]
    j ≤ n && return g.B[i - n, j]
    zero(T)
end

NNP.freeparameters(g::Lift) = (g.A, g.B)
NNP.rebuild(g::Lift, data) = Lift(data[1], data[2], g.N)
NNP.parameter_metadata(g::Lift) = (N = g.N,)

"A three-number symmetric matrix and the five-number two-block lift built on it."
sample_sym() = Sym([1.0, 2.0, 3.0], 2)
sample_twoblock() = TwoBlock(Sym([4.0, 5.0, 6.0], 2), [7.0 8.0], 3)

"The same three numbers as `sample_sym`, behind five fields of metadata rather than one."
sample_padded() = Padded(1, 2, 3, 4, 5, [1.0, 2.0, 3.0])

"A four-number matrix that is its own storage."
sample_manifold() = Manifold([1.0 2.0; 3.0 4.0])

"A 4 × 4 lift over a three-number `Sym` block and a 2 × 2 block: 3 + 4 numbers."
sample_lift() = Lift(Sym([0.7, -0.8, 0.9], 2), [0.1 -0.2; 0.3 0.4], 4)

"A parameter set with an ordinary layer, a structured leaf and a multi-block leaf: 3 + 3 + 5 numbers."
function sample_parameters()
    NetworkParameters((L1 = (W = [10.0 20.0], b = [30.0]),
        L2 = (S = sample_sym(),),
        L3 = (G = sample_twoblock(),)))
end
