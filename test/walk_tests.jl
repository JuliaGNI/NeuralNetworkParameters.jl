using NeuralNetworkParameters
using Test

include("wrapper_types.jl")

ps = sample_parameters()

@testset "mapparameters sees whole leaves" begin
    doubled = mapparameters(x -> 2x, ps)
    @test doubled.L1.W == [20.0 40.0]
    @test doubled isa NetworkParameters
    # a whole-leaf map over a structured leaf gets the dense object, so `2x` here is an ordinary matrix
    @test doubled.L2.S == 2 .* ps.L2.S
end

@testset "mapstorage sees only the storage" begin
    halved = mapstorage(x -> x ./ 2, ps)
    @test halved.L2.S isa Sym
    @test halved.L2.S.S == [0.5, 1.0, 1.5]
    @test halved.L2.S.n == 2
    @test halved.L1.W == [5.0 10.0]
    # the multi-block leaf recurses into both blocks, the first of them structured
    @test halved.L3.G isa TwoBlock
    @test halved.L3.G.A isa Sym
    @test halved.L3.G.A.S == [2.0, 2.5, 3.0]
    @test halved.L3.G.B == [3.5 4.0]
    @test halved.L3.G.N == 3
end

@testset "zipped walks" begin
    a = NetworkParameters((L = (x = [1.0, 2.0],),))
    b = NetworkParameters((L = (x = [10.0, 20.0],),))
    @test mapparameters(+, a, b).L.x == [11.0, 22.0]
    @test mapstorage((p, q) -> p .- q, b, a).L.x == [9.0, 18.0]
    # a NamedTuple and a NetworkParameters can be zipped together
    @test mapparameters(+, a, (L = (x = [10.0, 20.0],),)).L.x == [11.0, 22.0]
end

@testset "mismatched keys are rejected" begin
    a = NetworkParameters((L1 = [1.0],))
    b = NetworkParameters((L2 = [1.0],))
    @test_throws ArgumentError mapparameters(+, a, b)
end

@testset "in-place walks" begin
    dest = NetworkParameters((L = (x = [1.0, 2.0],),))
    src = NetworkParameters((L = (x = [10.0, 20.0],),))
    @test mapparameters!((d, s) -> (d .+= s), dest, src) === dest
    @test dest.L.x == [11.0, 22.0]

    sym = NetworkParameters((S = sample_sym(),))
    mapstorage!((d, s) -> (d .+= s), sym, NetworkParameters((S = sample_sym(),)))
    @test sym.S.S == [2.0, 4.0, 6.0]
    @test sym.S isa Sym
end

@testset "a `nothing` branch is skipped" begin
    # the shape a gradient tree has when a layer was not differentiated
    dest = NetworkParameters((p = [1.0], q = [2.0]))
    mapparameters!((d, s) -> (d .+= s), dest, NetworkParameters((p = [10.0], q = nothing)))
    @test dest.p == [11.0]
    @test dest.q == [2.0]

    # a whole subtree missing, not just a leaf
    dest2 = NetworkParameters((a = (x = [1.0], y = [2.0]), b = (z = [3.0],)))
    mapparameters!((d, s) -> (d .+= s), dest2,
        NetworkParameters((a = nothing, b = (z = [30.0],))))
    @test dest2.a.x == [1.0]
    @test dest2.a.y == [2.0]
    @test dest2.b.z == [33.0]

    visited = Symbol[]
    foreachparameters((_, _) -> push!(visited, :hit), dest2,
        NetworkParameters((a = nothing, b = (z = [1.0],))))
    @test length(visited) == 1
end

@testset "foldparameters" begin
    # whole leaves, so the structured leaves count their dense length
    @test foldparameters((n, x) -> n + length(x), 0, ps) == 2 + 1 + 4 + 9
    @test foldparameters((n, x) -> n + 1, 0, ps) == 4
    @test foldparameters((s, x) -> s + sum(x), 0.0, NetworkParameters((
        a = [1.0, 2.0], b = [3.0]))) == 6.0
end

@testset "a mapped tree takes its element type from what the map returned" begin
    @test mapstorage(x -> Float32.(x), ps) isa NetworkParameters{Float32}
    @test mapparameters(x -> Float32.(x), ps) isa NetworkParameters{Float32}

    # writing into existing leaves cannot change it: `copyto!` keeps the destination's types
    dest = NetworkParameters((a = [1.0],))
    mapparameters!((d, s) -> copyto!(d, s), dest, NetworkParameters((a = Float32[2],)))
    @test dest isa NetworkParameters{Float64}
    @test dest.a == [2.0]
end
