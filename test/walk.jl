using NeuralNetworkParameters
using Test

include("helpers/wrapper_types.jl")

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

# A downstream test matches this message by substring, so its wording and its count are pinned.
@testset "a zipped fold names the set whose keys differ ($T)" for T in (Float32, Float64)
    a = NetworkParameters((L1 = (W = T[1 2; 3 4], b = T[1, 2]),))
    b = NetworkParameters((L2 = (W = T[1 2; 3 4], b = T[1, 2]),))
    @test_throws "and, in argument 2," foldparameters((acc, x, y) -> acc + sum(x .* y), zero(T), a, b)
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

# A positional branch is the blocks of one multi-block leaf, so it is short — but a walk that filled
# it against a `nothing` with an `ntuple` over a *runtime* length would stop being inferable past ten.
# The primitives splice the `nothing`s in as literals, so there is no length to be runtime about.
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

# A `Tuple` branch against a set that is `nothing` raises as a keyed branch does: the branch of the
# set, a set that is a `Tuple`, and a leaf whose storage is a `Tuple` of blocks.
@testset "a fold of a tuple branch against a nothing set raises ($T)" for T in (Float32, Float64)
    op = (acc, x, y) -> acc
    @test_throws "partial sum" foldparameters(op, zero(T), (L = (T[1], T[2]),), (L = nothing,))
    @test_throws "partial sum" foldparameters(op, zero(T), (T[1], T[2]), nothing)
    @test_throws "partial sum" foldstorage(op, zero(T), (T[1], T[2]), nothing)
    lift = Lift(Sym(T[0.7, -0.8, 0.9], 2), T[0.1 -0.2; 0.3 0.4], 4)
    @test_throws "partial sum" foldstorage(op, zero(T),
        NetworkParameters((L = lift,)), NetworkParameters((L = nothing,)))
end

# A walk inside the function of another walk, as an optimizer runs one step per leaf: the outer walk
# runs the inner walk of the same kind on a one-leaf set of each leaf. The inner function is a
# callable struct, whose type nests one level deeper than the outer one, once around a function and
# once around a closure with captured coefficients, and the `TwoBlock` leaf takes the storage walks
# into the keyed storage of a leaf as well.
struct _InnerStep{F} <: Function
    f::F
end
(s::_InnerStep)(args::Vararg{Any, N}) where {N} = s.f(args...)

_axpy_leaf!(d, a) = (d .+= a; nothing)
# a step that captures its coefficients, as an Adam step does
_scaled_axpy(α, β, γ, δ, ε) = (d, a) -> (d .+= (α * β * γ * δ * ε) .* a; nothing)
function _dot_leaf(acc, a, b)
    for i in eachindex(a, b)
        acc += a[i] * b[i]
    end
    acc
end
_one_leaf(x) = NetworkParameters((leaf = x,))
# fixed entries of type `T`, all different, for a leaf of the given size
_entries(::Type{T}, dims...) where {T} = T.(reshape(1:prod(dims), dims) ./ 7)

function _nested_foreach!(X, Y)
    foreachparameters(X, Y) do x, y
        mapparameters!(_InnerStep(_axpy_leaf!), _one_leaf(x), _one_leaf(y))
        nothing
    end
end
function _nested_mapstorage!(X, Y)
    mapstorage!(X, Y) do x, y
        mapstorage!(_InnerStep(_axpy_leaf!), _one_leaf(x), _one_leaf(y))
        nothing
    end
end
function _nested_foreach_captured!(X, Y, c)
    foreachparameters(X, Y) do x, y
        mapparameters!(_InnerStep(_scaled_axpy(c, c, c, c, c)), _one_leaf(x), _one_leaf(y))
        nothing
    end
end
function _nested_mapstorage_captured!(X, Y, c)
    mapstorage!(X, Y) do x, y
        mapstorage!(_InnerStep(_scaled_axpy(c, c, c, c, c)), _one_leaf(x), _one_leaf(y))
        nothing
    end
end
function _nested_fold(X, Y, init)
    foldparameters(init, X, Y) do acc, x, y
        foldparameters(_InnerStep(_dot_leaf), acc, _one_leaf(x), _one_leaf(y))
    end
end
function _nested_foldstorage(X, Y, init)
    foldstorage(init, X, Y) do acc, x, y
        foldstorage(_InnerStep(_dot_leaf), acc, _one_leaf(x), _one_leaf(y))
    end
end

function _nested_allocs(walk::F, args::Vararg{Any, N}) where {F, N}
    (walk(args...); @allocated walk(args...))
end

@testset "a walk in the function of another walk allocates nothing ($T)" for T in (Float32, Float64)
    dense() = NetworkParameters((
        L1 = (W = _entries(T, 3, 2), b = _entries(T, 3)), L2 = (W = _entries(T, 1, 3),)))
    X, Y = dense(), dense()
    @test _nested_allocs(_nested_foreach!, X, Y) == 0
    @test _nested_allocs(_nested_mapstorage!, X, Y) == 0
    @test _nested_allocs(_nested_fold, X, Y, zero(T)) == 0
    @test _nested_allocs(_nested_foldstorage, X, Y, zero(T)) == 0
    @test _nested_allocs(_nested_foreach_captured!, X, Y, one(T)) == 0
    @test _nested_allocs(_nested_mapstorage_captured!, X, Y, one(T)) == 0

    structured() = NetworkParameters((
        L1 = (W = TwoBlock(Sym(_entries(T, 3), 2), _entries(T, 1, 2), 3),),
        L2 = (b = _entries(T, 2),)))
    S, R = structured(), structured()
    @test _nested_allocs(_nested_mapstorage!, S, R) == 0
    @test _nested_allocs(_nested_foldstorage, S, R, zero(T)) == 0

    # and the walks still do the work: one more step adds `Y` to `X`, and the fold is the inner product
    W = copy(X.L1.W)
    _nested_foreach!(X, Y)
    @test X.L1.W ≈ W + Y.L1.W
    inner = sum(X.L1.W .* Y.L1.W) + sum(X.L1.b .* Y.L1.b) + sum(X.L2.W .* Y.L2.W)
    @test _nested_fold(X, Y, zero(T)) ≈ inner
    A = copy(S.L1.W.B)
    _nested_mapstorage!(S, R)
    @test S.L1.W.B ≈ A + R.L1.W.B
end

# The inner walk takes four sets where the outer one takes two, as the `Adam` step of a leaf walks six
# sets inside a walk of five. A leaf method of the walk between the generated body and `f` would be
# inferred a second time from the same body, with a longer argument list, which inference shortens to
# a `Vararg` and then calls without specialising on `f`. These allocate at every leaf unless
# `foreachparameters` and `foldparameters` call `f` and `op` at a leaf directly.
function _scaled_axpy3(α, β, γ, δ, ε)
    (d, a, b, c) -> (d .+= (α * β * γ * δ * ε) .* a .* b .* c; nothing)
end
function _product_leaf(acc, a, b, c, d)
    for i in eachindex(a, b, c, d)
        acc += a[i] * b[i] * c[i] * d[i]
    end
    acc
end
function _nested_foreach_wider!(X, Y, c)
    foreachparameters(X, Y) do x, y
        mapparameters!(
            _InnerStep(_scaled_axpy3(c, c, c, c, c)), _one_leaf(x), _one_leaf(y),
            _one_leaf(y), _one_leaf(y))
        nothing
    end
end
function _nested_fold_wider(X, Y, init)
    foldparameters(init, X, Y) do acc, x, y
        foldparameters(
            _InnerStep(_product_leaf), acc, _one_leaf(x), _one_leaf(y), _one_leaf(y), _one_leaf(y))
    end
end

@testset "a walk of four sets in the function of a walk of two allocates nothing ($T)" for T in (
    Float32, Float64)
    dense() = NetworkParameters((
        L1 = (W = _entries(T, 3, 2), b = _entries(T, 3)), L2 = (W = _entries(T, 1, 3),)))
    X, Y = dense(), dense()
    @test _nested_allocs(_nested_foreach_wider!, X, Y, one(T)) == 0
    @test _nested_allocs(_nested_fold_wider, X, Y, zero(T)) == 0

    # and the walks still do the work: one more step adds `Y .^ 3` to `X`, and the fold is the sum
    # of `X .* Y .^ 3`
    W = copy(X.L1.W)
    _nested_foreach_wider!(X, Y, one(T))
    @test X.L1.W ≈ W + Y.L1.W .^ 3
    @test _nested_fold_wider(X, Y, zero(T)) ≈
          sum(X.L1.W .* Y.L1.W .^ 3) + sum(X.L1.b .* Y.L1.b .^ 3) +
          sum(X.L2.W .* Y.L2.W .^ 3)
end

# A map inside the function of another map, as an optimizer builds its cache and state per leaf: the
# outer map runs an inner map on a set made of each leaf, of one leaf and of two leaves at two depths,
# and at arity two. The inner function nests one level deeper than the outer one. The result type is
# inferred, so a caller that builds and uses the result in one function dispatches statically.
_double(x) = 2x
_two_leaves(x) = NetworkParameters((a = x, b = (c = x,)))
function _mapped_one_leaf(X)
    mapparameters(x -> mapparameters(_InnerStep(_double), _one_leaf(x)), X)
end
function _mapped_two_leaves(X)
    mapparameters(x -> mapparameters(_InnerStep(_double), _two_leaves(x)), X)
end
function _mapped_zipped(X, Y)
    mapparameters((x, y) -> mapparameters(_InnerStep(+), _one_leaf(x), _one_leaf(y)), X, Y)
end

@testset "a map in the function of another map infers its result ($T)" for T in (Float32, Float64)
    X = NetworkParameters((
        L1 = (W = _entries(T, 3, 2), b = _entries(T, 3)), L2 = (W = _entries(T, 1, 3),)))
    @test @inferred(_mapped_one_leaf(X)) isa NetworkParameters{T}
    @test @inferred(_mapped_two_leaves(X)) isa NetworkParameters{T}
    @test @inferred(_mapped_zipped(X, X)) isa NetworkParameters{T}

    # and the maps still do the work
    one_leaf = _mapped_one_leaf(X)
    @test one_leaf.L1.W.leaf == 2 * X.L1.W
    @test one_leaf.L2.W.leaf == 2 * X.L2.W
    @test one_leaf.L1.b.leaf isa Vector{T}
    two_leaves = _mapped_two_leaves(X)
    @test two_leaves.L1.b.a == 2 * X.L1.b
    @test two_leaves.L1.b.b.c == 2 * X.L1.b
    zipped = _mapped_zipped(X, X)
    @test zipped.L1.W.leaf == 2 * X.L1.W
    @test zipped.L2.W.leaf == 2 * X.L2.W
end

# `mapparameters` is one generated body per tree, as the in-place walks are: a mismatch below the top
# raises before `f` runs, a field type that is not concrete is walked into at run time, a set of
# `nothing` beside a branch raises, and a branch of more than 32 children below the top is mapped
# across at run time, inside a set that is rebuilt in the shape of the first.
@testset "the generated map keeps the contract of the map" begin
    calls = Ref(0)
    counted = (x, y) -> (calls[] += 1; x)
    a = (L1 = (W = [1.0],), L2 = (a = [1.0], b = [2.0]))
    @test_throws "different keys" mapparameters(counted, a,
        (L1 = (W = [1.0],), L2 = (b = [1.0], a = [2.0])))
    @test_throws "different keys" mapparameters(counted, a, (
        L1 = (W = [1.0],), L2 = (a = [1.0],)))
    @test_throws "same number of children" mapparameters(
        counted, (L1 = [1.0], L2 = ([1.0], [2.0])),
        (L1 = [1.0], L2 = ([1.0],)))
    @test_throws "nothing to put" mapparameters(counted, a, (
        L1 = (W = [1.0],), L2 = nothing))
    @test calls[] == 0

    loose = NamedTuple{(:L1, :L2), Tuple{Any, Vector{Float64}}}((
        (W = [1.0, 2.0], b = [3.0]), [4.0]))
    @test mapparameters(x -> 2x, loose) == (L1 = (W = [2.0, 4.0], b = [6.0]), L2 = [8.0])
    @test mapparameters(+, params(mapparameters(zero, NetworkParameters(loose))), loose) ==
          loose

    wide = NamedTuple{ntuple(i -> Symbol(:p, i), 33)}(ntuple(i -> [Float64(i)], 33))
    mapped = mapparameters(+, NetworkParameters((L = NetworkParameters(wide), M = [1.0])),
        (L = wide, M = [1.0]))
    @test mapped isa NetworkParameters
    @test mapped.L isa NetworkParameters
    @test mapped.L.p33 == [66.0]
    @test mapped.M == [2.0]
end

# The keys and widths of the whole tree are checked when the walk is generated, so a mismatch below
# the top raises before `f` runs on any leaf, and a walk that raises has changed nothing.
@testset "a mismatch below the top of the tree raises before any leaf is walked" begin
    a = (L1 = (W = [1.0],), L2 = (a = [1.0], b = [2.0]))
    calls = Ref(0)
    @test_throws ArgumentError foreachparameters((_, _) -> calls[] += 1, a,
        (L1 = (W = [1.0],), L2 = (b = [1.0], a = [2.0])))
    @test_throws ArgumentError foreachparameters((_, _) -> calls[] += 1, a,
        (L1 = (W = [1.0],), L2 = (a = [1.0],)))
    @test calls[] == 0
end

# A field type that is not concrete leaves the children of that node to the run time: the walk goes
# into the branch that the field holds, as it does into a branch whose type it knows.
@testset "a walk goes into a branch behind a field type that is not concrete" begin
    loose = NamedTuple{(:L1, :L2), Tuple{Any, Vector{Float64}}}((
        (W = [1.0, 2.0], b = [3.0]), [4.0]))
    seen = Float64[]
    foreachparameters(x -> append!(seen, x), loose)
    @test seen == [1.0, 2.0, 3.0, 4.0]
    @test foldparameters((acc, x) -> acc + sum(x), 0.0, loose) == 10.0
    @test foldparameters((acc, x, y) -> acc + sum(x) * sum(y), 0.0, loose, loose) ==
          9.0 + 9.0 + 16.0
end

# A leaf of a further set whose field type is not concrete can hold a `nothing` that only the run time
# knows, and the in-place walks skip it there as they skip a `nothing` whose type they know.
@testset "a nothing behind a field type that is not concrete is skipped" begin
    holes = NamedTuple{(:a, :b), Tuple{Any, Union{Nothing, Vector{Float64}}}}((
        [10.0], nothing))
    dest = (a = [1.0], b = [2.0])
    foreachparameters((d, s) -> (d .+= s; nothing), dest, holes)
    @test dest == (a = [11.0], b = [2.0])
    dest = NetworkParameters((a = [1.0], b = [2.0]))
    mapparameters!((d, s) -> (d .+= s; nothing), dest, NetworkParameters(holes))
    @test params(dest) == (a = [11.0], b = [2.0])
end

# The same for a further set: the walk goes into its branch at run time, beside the branch of the first
# set whose type it knows.
@testset "a walk goes into a branch of a further set behind a field type that is not concrete" begin
    loose = NamedTuple{(:L1, :L2), Tuple{Any, Vector{Float64}}}((
        (W = [1.0, 2.0], b = [3.0]), [4.0]))
    zeroed() = (L1 = (W = [0.0, 0.0], b = [0.0]), L2 = [0.0])
    added = (L1 = (W = [1.0, 2.0], b = [3.0]), L2 = [4.0])
    dest = zeroed()
    foreachparameters((d, s) -> (d .+= s; nothing), dest, loose)
    @test dest == added
    dest = zeroed()
    mapstorage!((d, s) -> (d .+= s; nothing), dest, loose)
    @test dest == added
    @test foldparameters((acc, x, y) -> acc + sum(x .* y), 0.0, added, loose) == 30.0
    @test foldstorage((acc, x, y) -> acc + sum(x .* y), 0.0, added, loose) == 30.0
end

@testset "a further set of the other kind of branch raises" begin
    keyed, positional = (a = [1.0],), ([1.0],)
    @test_throws MethodError foreachparameters((_, _) -> nothing, keyed, positional)
    @test_throws MethodError mapstorage!((_, _) -> nothing, keyed, positional)
    @test_throws MethodError foldparameters((acc, _, _) -> acc, 0.0, keyed, positional)
    @test_throws MethodError foldstorage((acc, _, _) -> acc, 0.0, keyed, positional)
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
