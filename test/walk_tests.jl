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

# `mapparameters` has always raised when two sets disagree on their keys; the in-place walks used to zip
# them *by position* instead, so a gradient tree whose keys were ordered differently wrote every leaf
# into the wrong parameter and said nothing. The check is in the generator now, which is why these are
# `ArgumentError`s at specialisation time and cost the walk nothing.
@testset "the in-place walks pair children by key, not by position" begin
    dest = NetworkParameters((a = [0.0], b = [0.0]))

    @test_throws "different keys" foreachparameters(
        copyto!, dest, NetworkParameters((b = [10.0], a = [99.0])))
    @test_throws "different keys" mapparameters!(
        copyto!, dest, NetworkParameters((x = [1.0], y = [2.0])))
    @test_throws "different keys" mapstorage!(
        copyto!, dest, NetworkParameters((b = [1.0], a = [2.0])))
    @test dest.a == [0.0] && dest.b == [0.0]

    # a nested set is checked at the level the keys disagree on and not only at the top
    @test_throws "different keys" mapparameters!(copyto!,
        NetworkParameters((L = (W = [0.0], b = [0.0]),)),
        NetworkParameters((L = (b = [1.0], W = [2.0]),)))

    # the same keys in the same order still walk, whichever form each side arrives in
    d = (a = [0.0], b = [0.0])
    foreachparameters(copyto!, d, NetworkParameters((a = [1.0], b = [2.0])))
    @test d == (a = [1.0], b = [2.0])
end

# A positional branch is the blocks of one multi-block leaf, so it is short — but `_tuple_for` used to
# fill it against a `nothing` with an `ntuple` over a *runtime* length, which stops being inferable past
# ten. The generated body splices the `nothing`s in as literals now, so there is no length to be runtime
# about.
@testset "a positional branch of more than ten blocks stays inferable" begin
    blocks = ntuple(i -> [Float64(i)], 12)
    @test only(Base.return_types(foreachparameters,
        Tuple{typeof(sum), typeof(blocks), Nothing})) === Nothing

    dest = ntuple(_ -> [0.0], 12)
    foreachparameters(copyto!, dest, blocks)
    @test dest == blocks

    touched = Ref(0)
    foreachparameters((_, _) -> (touched[] += 1), dest, nothing)
    @test touched[] == 0
end

@testset "foldparameters" begin
    # whole leaves, so the structured leaves count their dense length
    @test foldparameters((n, x) -> n + length(x), 0, ps) == 2 + 1 + 4 + 9
    @test foldparameters((n, x) -> n + 1, 0, ps) == 4
    @test foldparameters((s, x) -> s + sum(x), 0.0, NetworkParameters((
        a = [1.0, 2.0], b = [3.0]))) == 6.0
end

# The zipped arities, which is what `mapparameters` has always had and the fold did not. A consumer
# wanting ∑ᵢaᵢbᵢ or ∑ᵢf(aᵢ)² over a parameter set had to write the recursion itself, and
# `GeometricOptimizers` wrote three — see issue #19.
@testset "a zipped fold pairs the leaves" begin
    a = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
    b = mapparameters(x -> 2x, a)

    # the inner product of the two, taken without flattening either
    @test foldparameters((acc, x, y) -> acc + sum(x .* y), 0.0, a, b) ==
          2 * (1 + 4 + 9 + 16)
    # and at arity three, since nothing about the walk stops at two
    @test foldparameters((acc, x, y, z) -> acc + sum(x .* y .* z), 0.0, a, b, b) ==
          4 * (1 + 8 + 27 + 64)
    # a `NamedTuple` and a `NetworkParameters` pair, as they do for the other walks
    @test foldparameters((acc, x, y) -> acc + sum(x .* y), 0.0, a, params(b)) == 2 * 30

    # the fold still visits in the order `flatten` writes, at every arity
    paired = foldparameters((v, x, y) -> append!(v, vec(x .* y)), Float64[], a, b)
    @test paired == 2 .* first(flatten(a)) .^ 2
end

@testset "foldstorage folds the storage and not the leaf" begin
    # whole leaves count the dense length above; the storage counts what `flatten` writes
    @test foldstorage((n, x) -> n + length(x), 0, ps) == flatlength(ps)
    @test foldstorage((v, x) -> append!(v, vec(x)), Float64[], ps) == first(flatten(ps))

    # a multi-block leaf pairs block by block, and its structured block by its stored numbers
    doubled = mapstorage(x -> 2x, ps)
    @test foldstorage((acc, x, y) -> acc + sum(x .* y), 0.0, ps, doubled) ==
          2 * sum(abs2, first(flatten(ps)))
end

# A hole where a leaf keeps its storage behind an interface. `mapparameters` and `foldparameters` hand
# the whole leaf and the `nothing` to the caller and are done; the storage walks have to descend, and
# used to descend by asking the hole for its `freeparameters` — which answered with the leaf protocol's
# own error, telling the caller to define `freeparameters(::Nothing)`.
@testset "a nothing leaf reaches the storage walks too" begin
    holes = NetworkParameters((L1 = (W = nothing, b = nothing), L2 = (S = nothing,),
        L3 = (G = ps.L3.G,)))

    # the storage of a `Sym` is one array, so there is exactly one thing to pair the hole with, and
    # `op` sees it: 2 + 1 numbers from `L1` and 3 from `L2`, with `L3` paired against itself
    @test foldstorage((acc, x, y) -> y === nothing ? acc + length(x) : acc, 0, ps, holes) ==
          6
    @test mapstorage((x, y) -> y === nothing ? zero(x) : x, ps, holes).L2.S.S == zeros(3)

    # a hole against a *multi-block* leaf raises instead: one `nothing` cannot stand for each block,
    # and each walk says so in its own terms rather than in the leaf protocol's
    gaps = NetworkParameters((L1 = (W = nothing, b = nothing), L2 = (S = nothing,),
        L3 = (G = nothing,)))
    @test_throws "partial sum" foldstorage((acc, x, y) -> acc, 0.0, ps, gaps)
    @test_throws "nothing to put" mapstorage((x, y) -> x, ps, gaps)
end

@testset "a fold rejects what it cannot reduce" begin
    a = NetworkParameters((L1 = [1.0], L2 = [2.0]))

    # by key and not by position, which is the guarantee `mapparameters!` gained in D18
    @test_throws "different keys" foldparameters(
        (acc, x, y) -> acc, 0.0, a, NetworkParameters((L2 = [1.0], L1 = [2.0])))
    @test_throws "same number of children" foldparameters(
        (acc, x, y) -> acc, 0.0, ([1.0], [2.0]), ([1.0],))

    # a set that is `nothing` is an error and not a skip: a fold reduces every leaf it is given, so
    # leaving one out would make the result a partial sum without saying so
    @test_throws "partial sum" foldparameters((acc, x, y) -> acc, 0.0, a, nothing)
    @test_throws "partial sum" foldstorage((acc, x, y) -> acc, 0.0, a, nothing)
    @test_throws "partial sum" foldparameters(
        (acc, x, y) -> acc, 0.0, NetworkParameters((L = (x = [1.0],),)),
        NetworkParameters((L = nothing,)))

    # a `nothing` *leaf* still reaches `op`, exactly as it reaches `f` in `mapparameters`: deciding
    # what a missing leaf contributes is the caller's to make
    @test foldparameters((acc, x, y) -> y === nothing ? acc : acc + sum(y), 0.0,
        a, NetworkParameters((L1 = [10.0], L2 = nothing))) == 10.0
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
