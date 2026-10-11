using NeuralNetworkParameters
using NeuralNetworkParameters: _unrolled_map, _unrolled_foreach, _unrolled_foldl,
                               _unrolled_mapaccum, _FixFirst
using Test

struct Pair2{A, B}
    a::A
    b::B
end

@testset "the unrolled map of a keyed and a positional branch ($T)" for T in (Float32, Float64)
    for xs in ((a = T(1), b = T(2), c = T(3)), (T(1), T(2), T(3)))
        @test _unrolled_map(x -> 2x, xs) === (T(2), T(4), T(6))
        # the further sets pair child by child
        @test _unrolled_map(+, xs, xs, xs) === (T(3), T(6), T(9))
        # a further set that is `nothing` hands `nothing` to every child
        @test _unrolled_map((x, y) -> y, xs, nothing) === (nothing, nothing, nothing)
    end
end

@testset "the unrolled map of an empty branch" begin
    @test _unrolled_map(identity, ()) === ()
    @test _unrolled_map(identity, NamedTuple()) === ()
end

@testset "the unrolled map reads the fields of a struct ($T)" for T in (Float32, Float64)
    ys = _unrolled_map(identity, Pair2(T[1], T(2)))
    @test ys isa Tuple{Vector{T}, T}
    @test ys == (T[1], T(2))
end

# The walks bind their own first argument with `_FixFirst` and pass it one child of each set, so the
# bound function takes any number of further arguments.
@testset "the bound first argument passes every further argument ($T)" for T in (Float32, Float64)
    g = _FixFirst(+, T(1))
    @test g(T(2)) === T(3)
    @test g(T(2), T(3)) === T(6)
    @test g(T(2), T(3), T(4)) === T(10)
    @test @inferred(_unrolled_map(g, (T(1), T(2)), (T(3), T(4)))) === (T(5), T(7))
    # a bound type is kept as its own type, so a walk over a type stays inferred
    @test _FixFirst(convert, T) isa _FixFirst{typeof(convert), Type{T}}
    @test @inferred(_unrolled_map(_FixFirst(convert, T), (1, 2.0))) === (T(1), T(2))
end

@testset "the unrolled map is inferred at a width of $k" for k in (3, 48, 369)
    for T in (Float32, Float64)
        xs = ntuple(i -> T(i), k)
        nt = NamedTuple{ntuple(i -> Symbol(:p, i), k)}(xs)
        @test @inferred(_unrolled_map(x -> 2x, nt)) === ntuple(i -> T(2i), k)
        @test @inferred(_unrolled_map(+, xs, xs)) === ntuple(i -> T(2i), k)
    end
end

@testset "the unrolled foreach visits the children in order ($T)" for T in (Float32, Float64)
    seen = Any[]
    @test _unrolled_foreach((x, y) -> push!(seen, (x, y)), (a = T(1), b = T(2)), (
        a = T(10), b = T(20))) ===
          nothing
    @test seen == [(T(1), T(10)), (T(2), T(20))]
    @test all(s -> s isa Tuple{T, T}, seen)

    # a further set that is `nothing` hands `nothing` to every child
    empty!(seen)
    _unrolled_foreach((x, y) -> push!(seen, y), (T(1), T(2)), nothing)
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

@testset "the primitives check the arity and the keys ($T)" for T in (Float32, Float64),
    walk in (
        (xs, rest...) -> _unrolled_map(tuple, xs, rest...),
        (xs, rest...) -> _unrolled_foreach(tuple, xs, rest...),
        (xs, rest...) -> _unrolled_foldl(tuple, nothing, xs, rest...),
        (xs, rest...) -> _unrolled_mapaccum((s, args...) -> (args, s), nothing, xs, rest...))

    x, y = T(1), T(2)
    @test_throws "same number of children" walk((x, y), (x,))
    @test_throws "same number of children" walk((a = x, b = y), (a = x,))
    @test_throws "different keys" walk((a = x, b = y), (b = x, a = y))
    @test_throws "different keys" walk((a = x, b = y), (a = x, b = y), (a = x, c = y))
end

@testset "a positional set beside a keyed one pairs by position" begin
    @test _unrolled_map(tuple, (a = 1, b = 2), (3, 4)) === ((1, 3), (2, 4))
end
