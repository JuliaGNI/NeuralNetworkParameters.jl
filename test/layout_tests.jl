using NeuralNetworkParameters
using NeuralNetworkParameters: LeafLayout, NestedLayout, WrappedLayout, ParametersLayout
using Test

include("wrapper_types.jl")

ps = sample_parameters()
layout = parameterlayout(ps)

@testset "shape and extent" begin
    @test layout isa ParametersLayout
    @test length(layout) == 3 + 3 + 5
    @test parameterrange(layout) == 1:11
    @test flatlength(ps) == 11
    @test flatlength(layout) == 11
end

@testset "leaves land in declaration order" begin
    children = layout.inner.children
    @test keys(children) == (:L1, :L2, :L3)
    @test parameterrange(children.L1) == 1:3
    @test parameterrange(children.L1.children.W) == 1:2
    @test parameterrange(children.L1.children.b) == 3:3
    @test parameterrange(children.L2) == 4:6
    @test parameterrange(children.L3) == 7:11
end

@testset "structured leaves get a WrappedLayout" begin
    @test layout.inner.children.L2.children.S isa WrappedLayout
    @test layout.inner.children.L1.children.W isa LeafLayout
    # the two-block leaf nests: a wrapper around a tuple of a wrapper and a plain array
    g = layout.inner.children.L3.children.G
    @test g isa WrappedLayout
    @test length(g) == 5
end

@testset "sizes are recorded" begin
    @test layout.inner.children.L1.children.W.size == (1, 2)
    @test layout.inner.children.L1.children.b.size == (1,)
end

@testset "a leaf layout is the shape, not the leaf" begin
    @test fieldnames(LeafLayout) == (:range, :size)
    # leaves that differ only in what they hold share one layout type
    w64 = parameterlayout(NetworkParameters((W = [1.0 2.0],))).inner.children.W
    w32 = parameterlayout(NetworkParameters((W = Float32[1.0 2.0],))).inner.children.W
    @test typeof(w64) === typeof(w32)
    @test w64 == w32
end

@testset "scalar and empty leaves" begin
    l = parameterlayout(NetworkParameters((a = 1.5, b = Float64[], c = [2.0 3.0])))
    @test length(l) == 3
    @test l.inner.children.a.size == ()
    @test parameterrange(l.inner.children.a) == 1:1
    @test length(l.inner.children.b) == 0
    @test parameterrange(l.inner.children.c) == 2:3
end

@testset "a layout is a value, so it compares and prints" begin
    @test parameterlayout(ps) == parameterlayout(sample_parameters())
    @test parameterlayout(NetworkParameters((a = [1.0],))) !=
          parameterlayout(NetworkParameters((a = [1.0, 2.0],)))
    @test occursin("1:11", sprint(show, layout))
end

@testset "bare NamedTuples and Tuples" begin
    @test length(parameterlayout((a = [1.0, 2.0], b = [3.0]))) == 3
    @test length(parameterlayout(([1.0, 2.0], [3.0]))) == 3
end
