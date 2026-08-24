using NeuralNetworkParameters
using NeuralNetworkParameters: parameter_eltype
using ChainRulesCore
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

@testset "building a set from its keys and values" begin
    @test NetworkParameters(NamedTuple{(:a, :b)}(([1.0], [2.0]))) ==
          NetworkParameters((a = [1.0], b = [2.0]))
    # this is the form `AbstractNeuralNetworks.initialparameters` uses
    @test NetworkParameters(NamedTuple{keys(ps)}(values(ps))) == ps
    # the element type comes first, so the keys cannot go in the braces
    e = try
        NetworkParameters{(:a, :b)}(([1.0], [2.0]))
    catch err
        err
    end
    @test e isa ArgumentError
    @test occursin("NamedTuple{(:a, :b)}(values)", sprint(showerror, e))
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

@testset "the element type" begin
    @test parameter_eltype(ps) === Float64
    @test ps isa NetworkParameters{Float64}
    @test NetworkParameters((a = Float32[1, 2],)) isa NetworkParameters{Float32}

    # a promotion, not a uniformity guarantee: unlike `GeometricOptimizers`' `ArrayNamedTuple{T}`,
    # `NetworkParameters{T}` does not license assuming every leaf is a `T`
    mixed = NetworkParameters((a = Float32[1], b = [2.0]))
    @test mixed isa NetworkParameters{Float64}
    @test eltype(mixed.a) === Float32

    # nothing to promote. `SymbolicNeuralNetworks` builds empty sets, so this is exercised
    @test NetworkParameters(NamedTuple()) isa NetworkParameters{Union{}}
    # a gap where an untouched layer's entries would be, as `docs/src/walks.md` shows
    @test NetworkParameters((p = [10.0], q = nothing)) isa NetworkParameters{Float64}
    @test NetworkParameters((q = nothing,)) isa NetworkParameters{Union{}}
    # a leaf that is not this package's business at all: constructing must not throw
    @test NetworkParameters((f = sin,)) isa NetworkParameters{Any}

    # a nested set promotes through, reading the inner element type off its type
    @test NetworkParameters((i = NetworkParameters((a = Float32[1],)), b = [2.0])) isa
          NetworkParameters{Float64}

    # the element type takes no part in equality, which compares the wrapped `NamedTuple`s
    @test NetworkParameters((a = Float32[1],)) == NetworkParameters((a = [1.0],))
end

@testset "the element type binds in a signature" begin
    # this is what `GeometricOptimizers` needs of a parameter set: `T` recoverable from the type, so
    # that a set can join its `OptimizerSolution{T}` union
    f(::NetworkParameters{T}) where {T} = T
    @test f(ps) === Float64
    @test f(NetworkParameters(NamedTuple())) === Union{}

    g(::Union{AbstractVector{T}, NetworkParameters{T}}) where {T} = T
    @test g(ps) === Float64
    @test g(Float32[1]) === Float32

    # the `where {T, VT<:...}` shape the optimizer's cache and state constructors use
    h(::VT) where {T, VT <: Union{AbstractVector{T}, NetworkParameters{T}}} = T
    @test h(NetworkParameters((a = Float32[1],))) === Float32
    @test only(Base.return_types(h, Tuple{typeof(ps)})) === Type{Float64}
end

@testset "naming the element type asserts it" begin
    VT = typeof(values(nt))
    @test NetworkParameters{Float64, keys(nt), VT}(nt) == ps
    @test NetworkParameters{Float64}(nt) == ps

    # the element type is derived from the leaves, so a wrong one is an error rather than a type
    # that lies about what it holds
    @test_throws ArgumentError NetworkParameters{Float32, keys(nt), VT}(nt)
    @test_throws ArgumentError NetworkParameters{Float32}(nt)

    # the three-parameter form is the one `ChainRulesCore` calls, from `+(::P, ::Tangent{P})`
    @test ChainRulesCore.construct(typeof(ps), (params = nt,)) == ps
    @test ps + Tangent{typeof(ps)}(; params = ZeroTangent()) == ps

    @test isconcretetype(only(Base.return_types(NetworkParameters, Tuple{typeof(nt)})))
end
