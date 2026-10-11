using NeuralNetworkParameters
using NeuralNetworkParameters: _unrolled_map, _unrolled_foreach, _unrolled_foldl,
                               _unrolled_mapaccum
using Test

struct Pair2{A, B}
    a::A
    b::B
end

@testset "the unrolled map of a $(typeof(xs).name.name) branch" for xs in (
    (a = 1.0, b = 2.0, c = 3.0), (1.0, 2.0, 3.0))
    @test _unrolled_map(x -> 2x, xs) === (2.0, 4.0, 6.0)
    # the further sets pair child by child
    @test _unrolled_map(+, xs, xs, xs) === (3.0, 6.0, 9.0)
    # a further set that is `nothing` hands `nothing` to every child
    @test _unrolled_map((x, y) -> y, xs, nothing) === (nothing, nothing, nothing)
end

@testset "the unrolled map of an empty branch" begin
    @test _unrolled_map(identity, ()) === ()
    @test _unrolled_map(identity, NamedTuple()) === ()
end

@testset "the unrolled map reads the fields of a struct" begin
    @test _unrolled_map(identity, Pair2([1.0], 2)) == ([1.0], 2)
end

@testset "the unrolled map is inferred at a width of $k" for k in (3, 48, 369)
    for T in (Float32, Float64)
        xs = ntuple(i -> T(i), k)
        nt = NamedTuple{ntuple(i -> Symbol(:p, i), k)}(xs)
        @test @inferred(_unrolled_map(x -> 2x, nt)) === ntuple(i -> T(2i), k)
        @test @inferred(_unrolled_map(+, xs, xs)) === ntuple(i -> T(2i), k)
    end
end

@testset "the unrolled foreach visits the children in order" begin
    seen = Any[]
    @test _unrolled_foreach((x, y) -> push!(seen, (x, y)), (a = 1, b = 2), (
        a = 10, b = 20)) ===
          nothing
    @test seen == [(1, 10), (2, 20)]

    # a further set that is `nothing` hands `nothing` to every child
    empty!(seen)
    _unrolled_foreach((x, y) -> push!(seen, y), (1, 2), nothing)
    @test seen == [nothing, nothing]

    @test _unrolled_foreach(error, ()) === nothing
end

@testset "the unrolled foldl is a left fold at a width of $k" for k in (3, 48, 369)
    for T in (Float32, Float64)
        xs = ntuple(i -> T(i), k)
        # subtraction is not associative, so only a left fold gives this value
        @test @inferred(_unrolled_foldl(-, zero(T), xs)) === -sum(xs)
        @test _unrolled_foldl(-, zero(T), xs) isa T
        @test @inferred(_unrolled_foldl((acc, x, y) -> acc + x * y, zero(T), xs, xs)) ==
              sum(abs2, xs)
    end
    @test _unrolled_foldl((acc, x) -> (acc..., x), (), (1, 2, 3)) === (1, 2, 3)
    @test _unrolled_foldl(error, :init, ()) === :init
end

# the running sum before each child as the result, and the running sum after it as the state
_running(s, x) = (s, s + x)
_running(s, x, y) = (s, s + x * y)

@testset "the unrolled map with a state carries it in order at a width of $k" for k in (3, 48, 369)
    for T in (Float32, Float64)
        xs = ntuple(i -> T(i), k)
        ys, s = @inferred(_unrolled_mapaccum(_running, zero(T), xs))
        @test ys isa NTuple{k, T}
        @test collect(ys) == cumsum(collect(xs)) .- collect(xs)
        @test s === sum(xs)
        # a further set in lockstep
        ys2, s2 = _unrolled_mapaccum(_running, zero(T), xs, xs)
        @test s2 ≈ sum(abs2, xs)
        @test ys2[end] + xs[end] * xs[end] === s2
    end
    # results of a different type at each child, and the state threaded through them
    @test _unrolled_mapaccum((s, x) -> ((x, s), s + 1), 0, (a = :a, b = "b", c = 3.0)) ===
          (((:a, 0), ("b", 1), (3.0, 2)), 3)
    @test _unrolled_mapaccum(error, :init, ()) === ((), :init)
end

@testset "the primitives check the arity and the keys" for walk in (
    (xs, rest...) -> _unrolled_map(tuple, xs, rest...),
    (xs, rest...) -> _unrolled_foreach(tuple, xs, rest...),
    (xs, rest...) -> _unrolled_foldl(tuple, nothing, xs, rest...),
    (xs, rest...) -> _unrolled_mapaccum((s, args...) -> (args, s), nothing, xs, rest...))
    @test_throws "same number of children" walk((1, 2), (1,))
    @test_throws "same number of children" walk((a = 1, b = 2), (a = 1,))
    @test_throws "different keys" walk((a = 1, b = 2), (b = 1, a = 2))
    @test_throws "different keys" walk((a = 1, b = 2), (a = 1, b = 2), (a = 1, c = 2))
end

@testset "a positional set beside a keyed one pairs by position" begin
    @test _unrolled_map(tuple, (a = 1, b = 2), (3, 4)) === ((1, 3), (2, 4))
end
