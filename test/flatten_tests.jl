using NeuralNetworkParameters
using Test

include("wrapper_types.jl")

@testset "round trip" begin
    for ps in (NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],))),
        sample_parameters(),
        NetworkParameters((a = 1.5, b = Float64[], c = [2.0 3.0])),
        NetworkParameters((t = ([1.0, 2.0], [3.0]),)))
        v, layout = flatten(ps)
        @test length(v) == length(layout)
        @test unflatten(layout, v) == ps
    end
    # a bare NamedTuple stays a bare NamedTuple
    nt = (a = [1.0], b = [2.0])
    v, l = flatten(nt)
    @test unflatten(l, v) isa NamedTuple
    @test unflatten(l, v) == nt
end

@testset "leaf order is declaration order" begin
    v, _ = flatten(NetworkParameters((
        L1 = (W = [10.0 20.0], b = [30.0]), L2 = (W = [40.0],))))
    @test v == [10.0, 20.0, 30.0, 40.0]
    # column-major within a leaf
    v2, _ = flatten(NetworkParameters((W = [1.0 3.0; 2.0 4.0],)))
    @test v2 == [1.0, 2.0, 3.0, 4.0]
end

@testset "element type follows the parameters" begin
    @test eltype(first(flatten(NetworkParameters((a = Float32[1, 2],))))) === Float32
    @test eltype(first(flatten(NetworkParameters((a = [1.0],))))) === Float64
    # mixed precision promotes rather than truncating
    @test eltype(first(flatten(NetworkParameters((a = Float32[1], b = [2.0]))))) === Float64
    # and an explicit type is honoured
    @test eltype(first(flatten(Float32, NetworkParameters((a = [1.0],))))) === Float32
end

@testset "structured leaves keep their type and their metadata" begin
    ps = sample_parameters()
    v, layout = flatten(ps)
    @test v == [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    back = unflatten(layout, v)
    @test back.L2.S isa Sym
    @test back.L2.S.n == 2
    @test back.L3.G isa TwoBlock
    @test back.L3.G.A isa Sym
    @test back.L3.G.N == 3
end

@testset "unflatten is generic in the element type" begin
    ps = NetworkParameters((a = [1.0, 2.0],))
    _, layout = flatten(ps)
    @test eltype(unflatten(layout, Float32[1, 2]).a) === Float32
    @test unflatten(layout, [10, 20]).a == [10, 20]
end

@testset "leaves do not alias the flat vector" begin
    ps = NetworkParameters((a = [1.0, 2.0],))
    v, layout = flatten(ps)
    back = unflatten(layout, v)
    back.a[1] = 99.0
    @test v[1] == 1.0
end

@testset "in-place flatten! and unflatten!" begin
    ps = sample_parameters()
    v, layout = flatten(ps)
    buf = similar(v)
    @test flatten!(buf, ps, layout) === buf
    @test buf == v

    dest = unflatten(layout, zero(v))
    @test unflatten!(dest, layout, v) === dest
    @test dest.L1.W == ps.L1.W
    @test dest.L2.S.S == ps.L2.S.S
    @test dest.L3.G.A.S == ps.L3.G.A.S
    @test dest.L3.G.B == ps.L3.G.B

    # the layout can be rebuilt if it is not to hand
    @test flatten!(similar(v), ps) == v
end

@testset "the in-place forms do not allocate" begin
    ps = sample_parameters()
    v, layout = flatten(ps)
    buf = similar(v)
    dest = unflatten(layout, zero(v))
    flatten!(buf, ps, layout)              # warm up
    unflatten!(dest, layout, v)
    @test (@allocated flatten!(buf, ps, layout)) == 0
    @test (@allocated unflatten!(dest, layout, v)) == 0
end

@testset "length mismatches are caught" begin
    ps = NetworkParameters((a = [1.0, 2.0],))
    _, layout = flatten(ps)
    @test_throws DimensionMismatch flatten!(zeros(3), ps, layout)
    @test_throws DimensionMismatch unflatten!(ps, layout, zeros(3))
end

@testset "an immutable leaf cannot be written in place" begin
    ps = NetworkParameters((a = 1.5,))
    _, layout = flatten(ps)
    @test_throws ArgumentError unflatten!(ps, layout, [2.5])
end

@testset "splitting a Jacobian by parameter block" begin
    ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
    _, layout = flatten(ps)
    J = reshape(collect(1.0:9.0), 3, 3)     # 3 flat parameters, 3 outputs
    blocks = unflatten(layout, J)
    @test blocks.L1.W == J[1:2, :]
    @test blocks.L1.b == J[3:3, :]
end
