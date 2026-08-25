# What the walks cost as a function of the *width* of one branch.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/wide_branch_cost.jl              # the width sweep
#     julia --project=. scripts/wide_branch_cost.jl --cold-map    # `mapparameters` in a fresh session
#
# This is the harness behind the two tables in the 0.2.2 CHANGELOG entry, committed because a number
# is reproducible only where the code that produced it is. It reports *first-call* time, which is
# compilation plus a negligible run, and allocations on the second call from inside a function.
#
# The walk down a parameter tree is ordinary dispatch and has never been the cost. The walk across the
# children of one branch is, and it is superlinear in the width for any recursion that reaches the next
# child by retyping the tuple — which `Base.tail` does. The point of the sweep is that the exponent is
# visible: doubling the width used to multiply `flatten`'s compile time by about eight.
#
# The order of the calls matters and is why `--cold-map` exists. A `flatten` that has already compiled
# for a given branch shape leaves much less for `mapparameters` to do at the same shape, so the
# `mapparameters` column below is measured *before* `flatten`, and the figure for a session that never
# calls `flatten` at all comes from the second form.

using NeuralNetworkParameters

const T = Float32

# All leaves the same tiny matrix: this measures the shape of the branch, not the size of the arrays
# in it, and a 2×2 keeps the run time next to nothing beside the compilation.
wide_set(k::Integer) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(
    Tuple(fill(T(i), 2, 2) for i in 1:k))

# `time()` and not `@elapsed`: the point is the very first call, and `@elapsed` in a loop reports the
# second.
function first_call(f, args...)
    t = time()
    f(args...)
    round(time() - t; digits = 2)
end

# From inside a function, for the reason `test/flatten_tests.jl` gives: that is the claim that matters,
# an optimizer's inner loop rather than the top level of a script.
_flatten_allocs(buf, ps, layout) = @allocated flatten!(buf, ps, layout)
_unflatten_allocs(dest, layout, v) = @allocated unflatten!(dest, layout, v)

# 32 is where `Base` stops unrolling a tuple; 369 is the parameter set of the MNIST transformer in
# `scripts/geometric_optimizers/mnist.jl` of GMLDatasets.jl, which is the width that made the defect
# worth fixing rather than noting.
const WIDTHS = (16, 32, 48, 64, 128, 369)

function sweep()
    println(rpad("children", 10), rpad("layout", 9), rpad("map(zero)", 11),
            rpad("mapparams", 11), rpad("flatten", 9), rpad("unflatten", 11),
            "allocs flatten!/unflatten!")
    for k in WIDTHS
        ps = wide_set(k)
        t_layout = first_call(parameterlayout, ps)
        t_map = first_call(x -> map(zero, x), ps)
        t_mapp = first_call(x -> mapparameters(zero, x), ps)
        t_flat = first_call(flatten, ps)

        v, layout = flatten(ps)
        t_unflat = first_call(x -> unflatten(layout, x), v)

        buf = similar(v)
        dest = unflatten(layout, zero(v))
        _flatten_allocs(buf, ps, layout)          # warm the call and the measurement
        _unflatten_allocs(dest, layout, v)
        a_flat = _flatten_allocs(buf, ps, layout)
        a_unflat = _unflatten_allocs(dest, layout, v)

        println(rpad(k, 10), rpad(t_layout, 9), rpad(t_map, 11), rpad(t_mapp, 11),
                rpad(t_flat, 9), rpad(t_unflat, 11), a_flat, "/", a_unflat)
        flush(stdout)
    end
end

# `mapparameters` on a wide branch in a session that has compiled nothing else for that shape. This is
# the figure to quote when asking what the walk costs on its own.
function cold_map()
    println(rpad("children", 10), rpad("map(zero)", 11), "mapparameters(zero)")
    for k in WIDTHS
        ps = wide_set(k)
        println(rpad(k, 10), rpad(first_call(x -> map(zero, x), ps), 11),
                first_call(x -> mapparameters(zero, x), ps))
        flush(stdout)
    end
end

"--cold-map" in ARGS ? cold_map() : sweep()
