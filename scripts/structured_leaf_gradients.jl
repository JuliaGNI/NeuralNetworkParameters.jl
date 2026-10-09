# The flat and the structured Zygote gradient of a parameter set with `GeometricOptimizers`' leaf
# types, against a central difference on the flat storage.
#
# NeuralNetworkParameters does not depend on GeometricOptimizers, so this runs in an environment of
# its own that has both, Zygote and ForwardDiff:
#
#     julia --startup-file=no --project=<environment> scripts/structured_leaf_gradients.jl
#
# The environment develops GeometricOptimizers and either this repository or a released
# NeuralNetworkParameters, so the same script compares two versions.
#
# The first testset is the lift probe: a set with one `StiefelLieAlgHorMatrix` leaf and a loss that
# reads its dense form. The second runs each leaf type with a loss that uses it once, twice, and
# through a direct read of its storage field. In each case the flat gradient,
# `Zygote.gradient(v -> loss(unflatten(layout, v)), v)`, and the structured one,
# `Zygote.gradient(loss, ps)` followed by `flatten`, are equal, and both equal the central
# difference. The third gives a `SymmetricMatrix` leaf a cotangent of the other precision, and the
# gradient keeps the leaf's element type on both paths.

using NeuralNetworkParameters
using GeometricOptimizers
using GeometricOptimizers: SymmetricMatrix, SkewSymMatrix, StiefelLieAlgHorMatrix,
                           StiefelManifold
using Zygote
using LinearAlgebra
using Random
using Test

# The loss is a polynomial of degree two in the storage, so the central difference has no truncation
# error and only the rounding of the two loss values, about `eps · |L| / h`.
function central_difference(loss, layout, v::AbstractVector{Float64})
    h = cbrt(eps(Float64))
    map(eachindex(v)) do i
        vp = copy(v)
        vm = copy(v)
        vp[i] += h
        vm[i] -= h
        (loss(unflatten(layout, vp)) - loss(unflatten(layout, vm))) / 2h
    end
end

flat_gradient(loss, layout, v) = Zygote.gradient(w -> loss(unflatten(layout, w)), v)[1]
structured_gradient(loss, ps) = first(flatten(Zygote.gradient(loss, ps)[1]))

# `Float64` against a central difference at `rtol = 1e-6`, far above its rounding and far below the
# factor of 2 a second conversion of a block puts in. `Float32` against the `Float64` central difference
# of the same numbers at `sqrt(eps(Float32))`, the rounding of a `Float32` forward pass.
tolerance(::Type{Float64}) = 1e-6
tolerance(::Type{Float32}) = sqrt(eps(Float32))

# `ps64` is the set in `Float64`; the `Float32` set holds the same numbers, rounded.
function check_gradients(ps64, loss, T)
    ps = mapstorage(x -> T.(x), ps64)
    v, layout = flatten(ps)
    v64, layout64 = flatten(ps64)
    reference = central_difference(loss, layout64, v64)
    gflat = flat_gradient(loss, layout, v)
    gstruct = structured_gradient(loss, ps)
    @test eltype(gflat) === T
    @test eltype(gstruct) === T
    @test gflat == gstruct
    @test isapprox(gflat, reference; rtol = tolerance(T))
    @test isapprox(gstruct, reference; rtol = tolerance(T))
end

function random_lift()
    Random.seed!(1)
    N, n = 4, 2
    StiefelLieAlgHorMatrix(SkewSymMatrix(randn(n * (n - 1) ÷ 2), n), randn(N - n, n), N, n)
end

function lift_probe()
    @testset "the lift probe ($T)" for T in (Float32, Float64)
        check_gradients(NetworkParameters((L = (B = random_lift(),),)),
            p -> sum(abs2, Matrix(p.L.B)), T)
    end
end

function leaf(kind)
    Random.seed!(2)
    kind === :dense && return randn(3, 3)
    kind === :symmetric && return SymmetricMatrix(randn(6), 3)
    kind === :skew && return SkewSymMatrix(randn(3), 3)
    kind === :stiefel && return StiefelManifold(Matrix(qr(randn(3, 3)).Q))
    kind === :lift && return random_lift()
end

# The storage a loss reads directly, by field, as a loss that bypasses the interface would.
storage_read(X::Matrix) = sum(abs2, X)
storage_read(X::Union{SymmetricMatrix, SkewSymMatrix}) = sum(abs2, X.S)
storage_read(X::StiefelManifold) = sum(abs2, X.A)
storage_read(X::StiefelLieAlgHorMatrix) = sum(abs2, X.A.S) + 2sum(abs2, X.B)

# A weight that is not symmetric, so that the two places a stored number appears in the dense
# interface are weighted differently. It is a constant of the loss, so it is no part of the derivative.
function weight(X)
    T = eltype(X)
    reshape(collect(T, 1:length(X)), size(X)) ./ T(length(X))
end

once(p) = sum(abs2, Matrix(p.L.X) .* Zygote.@ignore(weight(p.L.X)))
twice(p) = once(p) + sum(Matrix(p.L.X) .* Zygote.@ignore(weight(p.L.X))')
stored(p) = storage_read(p.L.X)

function both_paths()
    @testset "both paths, one gradient ($kind, $use, $T)" for kind in (
            :dense, :symmetric, :skew, :stiefel, :lift),
        (use, loss) in (("once", once), ("twice", twice), ("storage", stored)),
        T in (Float32, Float64)
        check_gradients(NetworkParameters((L = (X = leaf(kind),),)), loss, T)
    end
end

# A loss that multiplies the leaf by a constant of the other precision gives the leaf a cotangent of
# that precision; the gradient has the leaf's element type on both paths.
function other_precision()
    @testset "a cotangent of another precision ($T)" for T in (Float32, Float64)
        S = T === Float32 ? Float64 : Float32
        ps = NetworkParameters((L = (X = SymmetricMatrix(T.(leaf(:symmetric).S), 3),),))
        v, layout = flatten(ps)
        loss(p) = sum(abs2, one(S) .* Matrix(p.L.X))
        gflat = flat_gradient(loss, layout, v)
        g = Zygote.gradient(loss, ps)[1]
        @test gflat isa Vector{T}
        @test g isa NetworkParameters{T}
        @test g.L.X isa SymmetricMatrix{T}
        @test first(flatten(g)) == gflat
    end
end

@testset "structured leaf gradients" begin
    lift_probe()
    both_paths()
    other_precision()
end
