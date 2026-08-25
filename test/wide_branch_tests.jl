# A branch with many children.
#
# Every walk across the children of one branch used to be an `@inline`d `Base.tail` chain, which cost
# one specialisation per child over argument types each as long as the branch — so inference on it grew
# as the cube of the width. `flatten` on a flat set took 0.17 s at 32 children, 2.2 s at 64, 17.6 s at
# 128 and was not usable at all at 369.
#
# 369 is not a synthetic number. It is the parameter set of the MNIST transformer in
# `scripts/geometric_optimizers/mnist.jl` of GMLDatasets.jl — 3·7·16 attention projections, 2·16 ResNet
# parameters and one classification weight, in one flat `NamedTuple` — which is written against this
# package alone. So this file is the width a real consumer has, not a limit chosen to be safe.
#
# What it pins is the property, not the timing: that the walks reach a wide branch at all, keep the
# shape, and stay allocation-free there. A wall-clock assertion would be a flake on a loaded CI
# machine; the timings live in the CHANGELOG beside the harness that produced them. The `@test` that
# this file *completes* is the regression test — the version before the fix does not.

using NeuralNetworkParameters
using NeuralNetworkParameters: isparametertree
using ChainRulesCore
using Test

# 369 for the reason above, and 48 as well: `Base` unrolls a tuple up to 32 fields and drops to a loop
# past it, so a case either side of that edge is worth having when a walk is rewritten.
const WIDTHS = (48, 369)

wide_set(k) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(
    Tuple(fill(Float32(i), 2, 2) for i in 1:k))

# Allocations are measured from inside a function throughout, for the reason
# `flatten_tests.jl` gives: at the top level the walk is not specialised and the figure is the
# harness's, not the code's.
_flatten_allocs(buf, ps, layout) = @allocated flatten!(buf, ps, layout)
_unflatten_allocs(dest, layout, v) = @allocated unflatten!(dest, layout, v)

@testset "flatten and unflatten reach a branch of $k children" for k in WIDTHS
    ps = wide_set(k)
    v, layout = flatten(ps)

    @test length(v) == 4k
    @test flatlength(ps) == 4k
    @test unflatten(layout, v) == ps
    # the order is the order of the keys, so the last leaf is the last four entries
    @test v[end-3:end] == fill(Float32(k), 4)
end

@testset "the in-place forms do not allocate at $k children" for k in WIDTHS
    ps = wide_set(k)
    v, layout = flatten(ps)
    buf = similar(v)
    dest = unflatten(layout, zero(v))

    _flatten_allocs(buf, ps, layout)          # warm up the call and the measurement
    _unflatten_allocs(dest, layout, v)
    @test _flatten_allocs(buf, ps, layout) == 0
    @test _unflatten_allocs(dest, layout, v) == 0
end

@testset "the walks reach a branch of $k children" for k in WIDTHS
    ps = wide_set(k)

    doubled = mapparameters(x -> 2x, ps)
    @test keys(doubled) == keys(ps)
    @test doubled.p1 == 2 * ps.p1
    @test doubled[Symbol("p", k)] == 2 * ps[Symbol("p", k)]

    # two sets in lockstep, which is the arity that used to `Base.tail` both tuples at once
    summed = mapparameters(+, ps, ps)
    @test summed.p1 == 2 * ps.p1

    @test foldparameters((acc, x) -> acc + sum(x), 0.0f0, ps) ==
          sum(sum, values(ps))

    dest = mapparameters(zero, ps)
    mapparameters!(copyto!, dest, ps)
    @test dest == ps

    visited = Ref(0)
    foreachparameters(_ -> (visited[] += 1), ps)
    @test visited[] == k

    # the `nothing` skip, which short-circuits over a wide branch
    touched = Ref(0)
    foreachparameters((_, _) -> (touched[] += 1), ps, nothing)
    @test touched[] == 0
end

@testset "a wide branch is a parameter tree" begin
    ps = wide_set(369)
    @test isparametertree(ps)
    @test parameter_eltype(ps) == Float32
end

# The arity check the `Base.tail` chains used to get by running out of `Tuple{}` methods, and which the
# written-out bodies have to raise for themselves.
@testset "walking branches of different widths is an error" begin
    @test_throws ArgumentError mapparameters(+, wide_set(48), wide_set(47))
end

# The reverse pass over a wide branch. `_accumulate_named!` was the one walk in `src/derivatives.jl`
# that a branch of layers makes wide, and a gradient step runs it on every call — so it is worth an
# assertion of its own rather than being taken on trust from the forward direction.
@testset "the pullback of unflatten reaches a branch of $k children" for k in WIDTHS
    ps = wide_set(k)
    v, layout = flatten(ps)
    _, pullback = ChainRulesCore.rrule(unflatten, layout, v)

    # a cotangent of ones in the shape of the parameters comes back as a flat vector of ones: every
    # leaf has to be found, and found at the right range. `one.(x)` and not `one(x)`: `one` of a matrix
    # is the identity, so the latter asks for the diagonal and gets it.
    Δ = pullback(mapparameters(x -> one.(x), ps))[3]
    @test Δ == ones(Float32, 4k)

    # a branch the cotangent is silent about is a structural zero rather than an error
    partial = NamedTuple{(:p1,)}((one.(ps.p1),))
    Δp = pullback(partial)[3]
    @test Δp[1:4] == ones(Float32, 4)
    @test all(iszero, Δp[5:end])
end
