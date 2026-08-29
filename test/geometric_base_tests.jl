# `GeometricBase.L2norm` over a parameter set, which `src/norms.jl` supplies.
#
# This file exists because it *can*: the method dispatches on this package's type and walks this
# package's leaf protocol, so this is the package that breaks it and therefore the package that has to
# catch it. `GeometricBase` supports Julia 1.10 and cannot resolve this package at all, so a method
# living there is one nobody can exercise.

using GeometricBase
using GeometricBase.Utils: L2norm, l2norm
using NeuralNetworkParameters
using NeuralNetworkParameters: NetworkParameters, flatten
using Test

# The method is in the main module and not behind an extension, so it is there for anyone who has
# loaded this package -- there is nothing to trigger. `GeometricBase` is a hard dependency, which most
# of this ecosystem takes anyway and whose own sole dependency is `Unicode`.
@testset "the method is this package's own" begin
    ps = NetworkParameters((L1 = (W = [3.0 0.0; 0.0 4.0], b = [0.0, 0.0]), L2 = (W = [0.0 0.0], b = [12.0])))
    @test which(L2norm, Tuple{typeof(ps)}).module === NeuralNetworkParameters

    # ... and it is not piracy, which is what makes that legal: `L2norm` is `GeometricBase`'s, but the
    # argument type dispatched on is this package's, and one owned side is enough.
    @test which(L2norm, Tuple{typeof(ps)}).sig.parameters[2] === NetworkParameters
end

# The leaves combine **in quadrature** and not by summing their norms. The two leaves that carry
# anything have norms `5` and `12`, which give `13` in quadrature and `17` summed — chosen so that the
# two cannot be confused, because the difference is inherited by every stopping criterion computed
# from this.
@testset "the leaves combine in quadrature" begin
    ps = NetworkParameters((L1 = (W = [3.0 0.0; 0.0 4.0], b = [0.0, 0.0]), L2 = (W = [0.0 0.0], b = [12.0])))
    leaves = (ps.L1.W, ps.L1.b, ps.L2.W, ps.L2.b)

    @test l2norm(ps) ≈ 13.0
    @test sum(l2norm, leaves) ≈ 17.0
    @test l2norm(ps) ≉ sum(l2norm, leaves)
    # ... and the quadrature is the norm of the flattening, since that is what the leaves are leaves of
    @test l2norm(ps) ≈ l2norm(flatten(ps)[1])
end

# Depth is irrelevant: the fold reaches a leaf wherever it is, so the same leaves grouped differently
# give the same number. `foldparameters` threads its accumulator through the branches, so this holds by
# construction rather than by the two groupings happening to agree.
@testset "the grouping of the leaves does not change the norm" begin
    flat   = NetworkParameters((a = [3.0, 0.0], b = [0.0, 4.0], c = [12.0]))
    nested = NetworkParameters((L1 = (a = [3.0, 0.0],), L2 = (b = [0.0, 4.0], c = [12.0])))
    deep   = NetworkParameters((G = (L1 = (a = [3.0, 0.0],), L2 = (b = [0.0, 4.0],)), H = (c = [12.0],)))

    @test l2norm(flat) == l2norm(nested) == l2norm(deep) ≈ 13.0
end

# `L2norm` folds `abs2(l2norm(leaf))` and deliberately *not* `L2norm(leaf)`, which is what leaves a leaf
# its own notion of norm. A downstream package defines `l2norm` over a structured leaf's **free**
# parameters, where the generic `L2norm(::AbstractArray)` would read a dense interface and count a
# block twice. This is the property `GeometricOptimizers` depends on, expressed here without it: a leaf
# type whose `l2norm` disagrees with its dense reading has to be the one that is used.
struct HalfCountingLeaf <: AbstractArray{Float64, 1}
    data::Vector{Float64}
end
Base.size(x::HalfCountingLeaf) = size(x.data)
Base.getindex(x::HalfCountingLeaf, i::Int) = x.data[i]
GeometricBase.Utils.l2norm(x::HalfCountingLeaf) = l2norm(x.data) / 2

@testset "the leaf's own l2norm is what is folded" begin
    ps = NetworkParameters((L1 = (w = HalfCountingLeaf([6.0, 8.0]),),))

    # the leaf's method says 5, not the dense reading's 10
    @test l2norm(ps.L1.w) ≈ 5.0
    @test l2norm(ps) ≈ 5.0
    # and the dense reading, which the fold must not have taken
    @test l2norm(ps) ≉ l2norm(collect(ps.L1.w))
end

@testset "an empty set and a one-block set" begin
    # `false` is the strong zero, so a one-block set stays in the block's own type rather than being
    # promoted by the accumulator
    f32 = NetworkParameters((L1 = (w = Float32[3.0, 4.0],),))
    @test l2norm(f32) isa Float32
    @test l2norm(f32) ≈ 5.0f0

    @test L2norm(NetworkParameters(NamedTuple())) == false
end
