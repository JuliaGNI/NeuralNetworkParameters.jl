using NeuralNetworkParameters
using Test

include("wrapper_types.jl")

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
fp = FlatParameters(ps)

@testset "the vector interface" begin
    @test fp isa AbstractVector{Float64}
    @test length(fp) == 4
    @test size(fp) == (4,)
    @test collect(fp) == [1.0, 2.0, 3.0, 4.0]
    @test fp[2] == 2.0
    fp2 = copy(fp)
    fp2[2] = 20.0
    @test fp2[2] == 20.0
    @test fp[2] == 2.0
    @test parent(fp) isa Vector{Float64}
    @test sum(fp) == 10.0
end

@testset "construction and conversion" begin
    @test flatlayout(fp) == parameterlayout(ps)
    @test NetworkParameters(fp) == ps
    @test unflatten(fp) == ps
    @test eltype(FlatParameters(Float32, ps)) === Float32
    @test_throws DimensionMismatch FlatParameters(zeros(3), parameterlayout(ps))
end

@testset "reading a layer off the flat form" begin
    @test fp.L1.b == [3.0]
    @test fp.L1.W == [1.0 2.0]
    @test fp[:L2].W == [4.0;;]
    @test_throws ArgumentError fp.L9
    # the real fields stay reachable, which is why `getproperty` is safe to overload here
    @test fp.data === parent(fp)
    @test fp.layout === flatlayout(fp)
    @test :L1 in propertynames(fp)
end

@testset "similar keeps the layout" begin
    @test flatlayout(similar(fp)) == flatlayout(fp)
    @test similar(fp) isa FlatParameters
    @test eltype(similar(fp, Float32)) === Float32
    @test flatlayout(similar(fp, Float32)) == flatlayout(fp)
    # a different length cannot keep it
    @test similar(fp, Float64, (7,)) isa Vector{Float64}
end

@testset "broadcasting" begin
    @test 2 .* fp isa FlatParameters
    @test collect(2 .* fp) == [2.0, 4.0, 6.0, 8.0]
    @test flatlayout(2 .* fp) == flatlayout(fp)
    g = similar(fp)
    g .= fp
    @test collect(g) == collect(fp)
end

@testset "in-place conversions" begin
    ps2 = NetworkParameters((L1 = (W = [0.0 0.0], b = [0.0]), L2 = (W = [0.0;;],)))
    unflatten!(ps2, fp)
    @test ps2 == ps
    ps3 = NetworkParameters((L1 = (W = [9.0 9.0], b = [9.0]), L2 = (W = [9.0;;],)))
    flatten!(fp, ps3)
    @test collect(fp) == [9.0, 9.0, 9.0, 9.0]
end

@testset "a flat form over a bare NamedTuple" begin
    bare = FlatParameters((a = [1.0], b = [2.0]))
    @test collect(bare) == [1.0, 2.0]
    @test_throws ArgumentError NetworkParameters(bare)
    @test unflatten(bare) == (a = [1.0], b = [2.0])
end

@testset "show" begin
    @test occursin("FlatParameters", sprint(show, MIME"text/plain"(), fp))
end
