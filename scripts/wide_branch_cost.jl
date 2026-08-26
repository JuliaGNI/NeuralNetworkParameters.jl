# What the walks cost as a function of the *width* of one branch.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/wide_branch_cost.jl
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
# **Every width runs in a process of its own**, and that is the methodology rather than a detail.
# Compilation is shared across a session, so a `flatten` that has already compiled for one branch shape
# leaves less for `mapparameters` to do at the same shape — and a sweep that walks the widths in one
# process is measuring, from the second row on, what the first row already paid for.
#
# This script used to do exactly that, with a `--cold-map` flag to recover the honest figure on demand.
# The flag was the tell: a number you have to remember to ask for in a special way is a number the
# default run is getting wrong. `GeometricOptimizers` quoted the default one — `mapparameters` at
# "0.00 s" on a 369-entry branch, against 1.51 s measured cold — and closed an issue on it. A table is
# cold here or it is not printed.

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

header() = println(rpad("children", 10), rpad("layout", 9), rpad("map(zero)", 11),
                   rpad("mapparams", 11), rpad("flatten", 9), rpad("unflatten", 11),
                   "allocs flatten!/unflatten!")

# One width, in a process that has compiled nothing for it. Within the row the order still matters —
# `flatten` builds a layout and would absorb `parameterlayout`, and `mapparameters` would absorb the
# rest — so the rows are ordered cheapest-first and `flatten` comes after both.
function sweep_one(k)
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

# `--in-process k` is what the child processes are invoked with, and is not for interactive use.
function fan_out()
    header()
    for k in WIDTHS
        run(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
                      $(@__FILE__) --in-process $k`))
    end
end

"--in-process" in ARGS ? sweep_one(parse(Int, ARGS[findfirst(==("--in-process"), ARGS) + 1])) : fan_out()
