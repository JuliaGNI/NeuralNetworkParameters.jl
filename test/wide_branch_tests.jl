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

# The same width with a layer's worth of nesting under each child, which is the shape a network of
# many layers actually has. `wide_set` exercises the width alone: its children are arrays, and an
# array is already a heap object, so a walk that reaches one has nothing to keep off the heap. Under
# `nested_set` every child is a branch that a walk has to take apart in place, and that is where a
# walk which is written correctly for the width but not inlined shows up.
nested_set(k) = NamedTuple{Tuple(Symbol("L", i) for i in 1:k)}(
    Tuple((W = fill(Float32(i), 2, 2), b = fill(Float32(i), 2)) for i in 1:k))

# Allocations are measured from inside a function throughout, for the reason `flatten_tests.jl` gives:
# that is the claim that matters, an optimizer's inner loop rather than the top level of a testset.
_flatten_allocs(buf, ps, layout) = @allocated flatten!(buf, ps, layout)
_unflatten_allocs(dest, layout, v) = @allocated unflatten!(dest, layout, v)

# The `foreach` family has to be measured through a **zero-argument** closure, and that is not a
# stylistic choice. `@allocated f(a, b)` lowers to `Base.allocated(f, a, b)`, and `foreachparameters`,
# `mapparameters!` and `mapstorage!` are `Vararg` methods — so the splat inside `Base.allocated`
# allocates on its own account, which both invents bytes these walks do not spend and hides the ones
# they do. `flatten!` is not a `Vararg` method, which is why the two helpers above can be written the
# obvious way. `@allocated thunk()` takes no arguments to splat.
_thunk_allocs(thunk) = (thunk(); thunk(); @allocated thunk())

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

# The in-place walks at every arity, which is the half of D13 that `flatten!` and `unflatten!` do not
# cover. Zipping two sets used to take `values` of each branch of *each* argument, which is the same
# temporary per branch the forward walks stopped taking: `mapparameters!` cost 800 bytes a call on a
# flat 48-child set and 6 144 at 369, and three to four times that on a branch of branches. These are
# the walks an optimizer runs every iteration, where `flatten!` runs once or twice, so they are the
# ones the guarantee is worth most on.
@testset "the walks that take two sets do not allocate at $k children" for k in WIDTHS
    ps = wide_set(k)
    dest = mapparameters(zero, ps)
    counted = Ref(0)
    bump1(_) = (counted[] += 1; nothing)
    bump2(_, _) = (counted[] += 1; nothing)
    bump3(_, _, _) = (counted[] += 1; nothing)

    @test _thunk_allocs(() -> foreachparameters(bump1, dest)) == 0
    @test _thunk_allocs(() -> foreachparameters(bump2, dest, ps)) == 0
    @test _thunk_allocs(() -> mapparameters!(bump2, dest, ps)) == 0
    @test _thunk_allocs(() -> mapparameters!(bump3, dest, ps, ps)) == 0
    @test _thunk_allocs(() -> mapstorage!(bump2, dest, ps)) == 0
    # the `nothing` skip too: it used to fill the branch with a fresh tuple of `nothing`s per call
    @test _thunk_allocs(() -> foreachparameters(bump2, dest, nothing)) == 0

    # and the walk really ran, rather than being elided as the dead code a side-effect-free `f` is
    @test counted[] > 0
    mapparameters!(copyto!, dest, ps)
    @test dest == ps
end

# 48 and not 369: the width is covered above, and what nesting adds is a branch whose children are
# branches, which 48 is already well past the unrolling boundary for. A real consumer has one shape or
# the other — the 369-child set is flat, and a network with a child per layer is dozens deep, not
# hundreds — and the nested 369 would cost the suite another minute to say nothing the 48 does not.
@testset "a wide branch of branches round-trips and does not allocate at $k children" for k in (48,)
    ps = nested_set(k)
    v, layout = flatten(ps)

    @test length(v) == 6k
    @test unflatten(layout, v) == ps
    @test mapparameters(x -> 2x, ps).L1.W == 2 * ps.L1.W

    buf = similar(v)
    dest = unflatten(layout, zero(v))
    _flatten_allocs(buf, ps, layout)          # warm up the call and the measurement
    _unflatten_allocs(dest, layout, v)
    @test _flatten_allocs(buf, ps, layout) == 0
    @test _unflatten_allocs(dest, layout, v) == 0

    # the two-set walks on the shape that has a branch to take apart at every child, which is where a
    # temporary per branch per argument costs the most
    counted = Ref(0)
    bump2(_, _) = (counted[] += 1; nothing)
    @test _thunk_allocs(() -> mapparameters!(bump2, dest, ps)) == 0
    @test _thunk_allocs(() -> mapstorage!(bump2, dest, ps)) == 0
    @test _thunk_allocs(() -> foreachparameters(bump2, dest, nothing)) == 0
    @test counted[] > 0
end

# The same width in the shape a consumer holds it. Every other set in this file is a bare `NamedTuple`,
# and the two are not the same measurement: a bare branch reaches `parameterlayout` through one
# `@generated` body, a wrapped one goes through `_layout(::NetworkParameters, ::Int)` first. On Julia
# 1.11 those cost 1.37 s and 13.44 s on the same 369 leaves. The *properties* below hold on both and are
# what this file pins; the cost is `scripts/wide_branch_cost.jl`'s business, and it sweeps both shapes at
# every width for exactly this reason.
#
# 48 and not 369, for the reason the nesting testset gives: nothing here depends on the width once the
# width is past the unrolling boundary, and a wrapped 369 would cost the suite the better part of half a
# minute on the compat floor to say what the 48 already says.
wrapped_set(k) = NetworkParameters(wide_set(k))

@testset "a wide NetworkParameters round-trips and does not allocate at $k children" for k in (48,)
    ps = wrapped_set(k)
    v, layout = flatten(ps)

    @test length(v) == 4k
    @test flatlength(ps) == 4k
    @test unflatten(layout, v) isa NetworkParameters
    @test unflatten(layout, v) == ps
    @test mapparameters(x -> 2x, ps).p1 == 2 * ps.p1

    buf = similar(v)
    dest = unflatten(layout, zero(v))
    _flatten_allocs(buf, ps, layout)          # warm up the call and the measurement
    _unflatten_allocs(dest, layout, v)
    @test _flatten_allocs(buf, ps, layout) == 0
    @test _unflatten_allocs(dest, layout, v) == 0

    # the two-set walks too: the wrapper is one more level for them to index in place
    counted = Ref(0)
    bump2(_, _) = (counted[] += 1; nothing)
    @test _thunk_allocs(() -> mapparameters!(bump2, dest, ps)) == 0
    @test _thunk_allocs(() -> mapstorage!(bump2, dest, ps)) == 0
    @test counted[] > 0
end

@testset "a wide branch is a ParameterSet and a parameter tree" begin
    ps = wide_set(369)
    @test ps isa ParameterSet
    @test isparametertree(ps)
    @test NetworkParameters(ps) isa ParameterSet
    @test parameter_eltype(ps) == Float32
end

# Two checks, and a pair of wide `NamedTuple`s only ever reaches the first of them: branches of
# different widths are keyed differently, so `_check_keys` rejects them before any walk starts. The
# arity check the `Base.tail` chains used to get by running out of `Tuple{}` methods, and which the
# written-out bodies raise for themselves, is reachable only through a *positional* branch — a
# multi-block leaf whose blocks do not line up.
@testset "walking mismatched branches is an error" begin
    @test_throws "different keys" mapparameters(+, wide_set(48), wide_set(47))
    @test_throws "same number of children" mapparameters(+, ([1.0], [2.0]), ([1.0],))
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
