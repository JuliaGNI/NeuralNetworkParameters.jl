# What a walk costs when it runs inside the function of another walk, and what one walk costs on its
# own.
#
# Run with the repository as the active project, one row per process:
#
#     julia --project=. scripts/nested_walk_cost.jl nested
#     julia --project=. scripts/nested_walk_cost.jl wide
#     julia --project=. scripts/nested_walk_cost.jl <walk> <shape>
#
# `nested` prints the allocations of each in-place walk and fold when its function runs a second walk
# of the same kind on a one-leaf set, the way an optimizer that steps each leaf with its own method
# does. `wide` does the same with an inner walk of four sets inside an outer walk of two, as
# `KNOWN_ISSUES.md` K11 cites. `<walk>` is one of `foreach`, `mapstorage!`, `fold` and `foldstorage`, and `<shape>` one of
# `small` (3 leaves, depth 2), `deep` (64 leaves, depth 4) and `flat` (369 leaves, depth 1). That mode
# prints the first-call time of a cold process, which is compilation and a negligible run, the
# allocations of the second call, and the fastest of 1000 calls.
using NeuralNetworkParameters

# A function held in a callable struct, as a caller of the walks passes its own per-leaf step
struct Step{F} <: Function
    f::F
end
(s::Step)(args::Vararg{Any, N}) where {N} = s.f(args...)

axpy_leaf!(d, a) = (d .+= a; nothing)
# a step that captures its coefficients, as an Adam step captures its β₁, β₂, ε, α and t
scaled_axpy(α, β, γ, δ, ε) = (d, a) -> (d .+= (α * β * γ * δ * ε) .* a; nothing)
function dot_leaf(acc, a, b)
    for i in eachindex(a, b)
        acc += a[i] * b[i]
    end
    acc
end

one_leaf(x) = NetworkParameters((leaf = x,))

# Each outer walk runs the inner walk of the same kind on a one-leaf set of each leaf.
nested_foreach!(X, Y) = foreachparameters(X, Y) do x, y
    mapparameters!(Step(axpy_leaf!), one_leaf(x), one_leaf(y))
    nothing
end
nested_mapstorage!(X, Y) = mapstorage!(X, Y) do x, y
    mapstorage!(Step(axpy_leaf!), one_leaf(x), one_leaf(y))
    nothing
end
function nested_foreach_captured!(X, Y, c)
    foreachparameters(X, Y) do x, y
        mapparameters!(Step(scaled_axpy(c, c, c, c, c)), one_leaf(x), one_leaf(y))
        nothing
    end
end
function nested_mapstorage_captured!(X, Y, c)
    mapstorage!(X, Y) do x, y
        mapstorage!(Step(scaled_axpy(c, c, c, c, c)), one_leaf(x), one_leaf(y))
        nothing
    end
end
nested_fold(X, Y) = foldparameters(zero(eltype(X.L1.W)), X, Y) do acc, x, y
    foldparameters(Step(dot_leaf), acc, one_leaf(x), one_leaf(y))
end
nested_foldstorage(X, Y) = foldstorage(zero(eltype(X.L1.W)), X, Y) do acc, x, y
    foldstorage(Step(dot_leaf), acc, one_leaf(x), one_leaf(y))
end

# An inner walk of four sets inside an outer walk of two.
axpy3_leaf!(d, a, b, c) = (d .+= a .+ b .+ c; nothing)
dot3_leaf(acc, a, b, c, d) = dot_leaf(acc, a, b) + dot_leaf(zero(acc), c, d)
function wide_foreach!(X, Y)
    foreachparameters(X, Y) do x, y
        mapparameters!(
            Step(axpy3_leaf!), one_leaf(x), one_leaf(y), one_leaf(y), one_leaf(y))
        nothing
    end
end
function wide_mapstorage!(X, Y)
    mapstorage!(X, Y) do x, y
        mapstorage!(Step(axpy3_leaf!), one_leaf(x), one_leaf(y), one_leaf(y), one_leaf(y))
        nothing
    end
end
function wide_fold(X, Y)
    foldparameters(zero(eltype(X.L1.W)), X, Y) do acc, x, y
        foldparameters(
            Step(dot3_leaf), acc, one_leaf(x), one_leaf(y), one_leaf(x), one_leaf(y))
    end
end
function wide_foldstorage(X, Y)
    foldstorage(zero(eltype(X.L1.W)), X, Y) do acc, x, y
        foldstorage(
            Step(dot3_leaf), acc, one_leaf(x), one_leaf(y), one_leaf(x), one_leaf(y))
    end
end

function allocations(f::F, args::Vararg{Any, N}) where {F, N}
    f(args...)
    @allocated f(args...)
end

function nested()
    for T in (Float32, Float64)
        set() = NetworkParameters((
            L1 = (W = rand(T, 3, 2), b = rand(T, 3)), L2 = (W = rand(T, 1, 3),)))
        X, Y = set(), set()
        for (name, walk) in (("foreachparameters", nested_foreach!), (
            "mapstorage!", nested_mapstorage!),
            ("foldparameters", nested_fold), ("foldstorage", nested_foldstorage))
            println(rpad("$T $name in $name:", 48), allocations(walk, X, Y), " B")
        end
        for (name, walk) in (("foreachparameters", nested_foreach_captured!),
            ("mapstorage!", nested_mapstorage_captured!))
            println(rpad("$T $name in $name, captured:", 48), allocations(walk, X, Y, one(T)), " B")
        end
    end
end

function wide()
    for T in (Float32, Float64)
        set() = NetworkParameters((
            L1 = (W = rand(T, 3, 2), b = rand(T, 3)), L2 = (W = rand(T, 1, 3),)))
        X, Y = set(), set()
        for (name, walk) in (("foreachparameters", wide_foreach!), (
            "mapstorage!", wide_mapstorage!),
            ("foldparameters", wide_fold), ("foldstorage", wide_foldstorage))
            println(rpad("$T $name of 4 in $name of 2:", 60), allocations(walk, X, Y), " B")
        end
    end
end

# The sets of the cost mode: leaves of 2 × 2
function shape(name)
    leaf() = rand(2, 2)
    name == "small" &&
        return NetworkParameters((L1 = (W = leaf(), b = rand(2)), L2 = (W = leaf(),)))
    name == "flat" && return NetworkParameters(NamedTuple{ntuple(i -> Symbol(:p, i), 369)}(
        ntuple(_ -> leaf(), 369)))
    name == "deep" || throw(ArgumentError("unknown shape $name"))
    level(k, f) = NamedTuple{ntuple(i -> Symbol(:n, i), 4)}(ntuple(_ -> f(), 4))
    NetworkParameters(level(4, () -> level(4, () -> level(4, () -> (W = leaf(),)))))
end

sum_product(a, x, y) = a + sum(x) * sum(y)
const WALKS = Dict(
    "foreach" => (X, Y) -> mapparameters!(copyto!, X, Y),
    "mapstorage!" => (X, Y) -> mapstorage!(copyto!, X, Y),
    "fold" => (X, Y) -> foldparameters(sum_product, 0.0, X, Y),
    "foldstorage" => (X, Y) -> foldstorage(sum_product, 0.0, X, Y))

# Behind a barrier, so that the walk and the set are concrete. Compiling the barrier compiles the
# walk, so the first call is timed outside it.
function measure(walk::F, X, Y) where {F}
    bytes = @allocated walk(X, Y)
    fastest = minimum(@elapsed(walk(X, Y)) for _ in 1:1000)
    bytes, fastest
end

function cost(walk, name)
    X, Y = shape(name), shape(name)
    first = @elapsed WALKS[walk](X, Y)
    bytes, fastest = measure(WALKS[walk], X, Y)
    println(rpad("$walk $name:", 20), "first call ", round(first; digits = 2), " s, ",
        bytes, " B, fastest ", round(Int, fastest * 1e9), " ns")
end

ARGS == ["nested"] ? nested() : ARGS == ["wide"] ? wide() : cost(ARGS...)
