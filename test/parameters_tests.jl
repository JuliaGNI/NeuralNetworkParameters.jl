using NeuralNetworkParameters
using Test

nt = (L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],))
ps = NetworkParameters(nt)

@testset "construction and access" begin
    @test params(ps) === nt
    @test ps.L1 === nt.L1
    @test ps[:L1] === nt.L1
    @test ps.L1.W == [1.0 2.0]
    @test keys(ps) == (:L1, :L2)
    @test values(ps) === values(nt)
    @test length(ps) == 2
    @test collect(pairs(ps)) == collect(pairs(nt))
    @test hasproperty(ps, :L1)
    @test !hasproperty(ps, :L3)
    @test propertynames(ps) == (:L1, :L2)
end

@testset "the keyed constructor" begin
    @test NetworkParameters{(:a, :b)}(([1.0], [2.0])) ==
          NetworkParameters((a = [1.0], b = [2.0]))
    # this is the form `AbstractNeuralNetworks.initialparameters` uses
    @test NetworkParameters{keys(ps)}(values(ps)) == ps
end

@testset "equality" begin
    @test ps == NetworkParameters(deepcopy(nt))
    @test isequal(ps, NetworkParameters(deepcopy(nt)))
    @test ps != NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
    # a permuted key order is a different parameter set, which is what makes the HDF5 key-order
    # attribute matter
    @test NetworkParameters((a = [1.0], b = [2.0])) !=
          NetworkParameters((b = [2.0], a = [1.0]))
end

@testset "conversion to a NamedTuple" begin
    @test NamedTuple(ps) === nt
    @test NamedTuple(ps) === params(ps)
    # the round trip both ways, since this is what a downstream serialiser uses
    @test NetworkParameters(NamedTuple(ps)) == ps
end

@testset "iteration" begin
    @test collect(ps) == collect(nt)
end

@testset "show" begin
    @test occursin("NetworkParameters", sprint(show, ps))
    @test occursin("2 entries", sprint(show, MIME"text/plain"(), ps))
end
