# What the walks cost as a function of the *width* of one branch, and of the *shape* it is held in.
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
#
# **Every width is swept in both shapes: a bare `NamedTuple` and the same set inside a
# `NetworkParameters`.** That is the same methodology applied to the argument rather than to the clock,
# and it is worth as much. `parameterlayout` reaches a bare branch through one `@generated` body and a
# wrapped one through `_layout(::NetworkParameters, ::Int)` first, and until 0.2.3 those two cost wildly
# different amounts on the same leaves: on Julia 1.11, 1.35 s bare against 13.50 s wrapped at 369
# children, and 2.77 s against 87.77 s at 768 through `scripts/leaf_layout_cost.jl`, which reaches that
# width. The *whole* path cost nearly the same either way — 19.96 s
# bare against 21.90 s wrapped at 369, summed over `parameterlayout`, `flatten` and `unflatten` — so
# what the wrapper changed was which entry point paid, not the total.
#
# That is how this sweep found what 0.2.3 removed. The gap was not a cost of the wrapper, and issue #16
# was right that it was not a cost of nesting either: it was `LeafLayout`'s `prototype` type parameter,
# which put every leaf's concrete array type into the layout type this method's caller infers through.
# The two columns now agree — 1.05 s wrapped against 1.04 s bare at 369, 3.01 s against 3.07 s at 768,
# 0.84 s against 0.85 s on a 16 × 24 set of the same 384 leaves — and the whole 369-wrapped path is
# 8.64 s. The rows past 369 and the nested one are `scripts/leaf_layout_cost.jl`.
# Both shapes are still swept, for two reasons: a consumer holds a `NetworkParameters`, so a sweep of
# bare sets alone would report a shape nobody has, and two columns that agree are how the asymmetry
# coming back would be noticed.
#
# The two shapes run in separate processes for the reason the widths do. They are different methods but
# they share the inner `_layout(::NamedTuple, ::Int)` specialisation, so a bare row run first leaves the
# wrapped row less to compile. On Julia 1.11 the wrapped 369 row now takes about 10 s on its own
# account, where before 0.2.3 it took about 25 s.

using NeuralNetworkParameters

const T = Float32

# All leaves the same tiny matrix: this measures the shape of the branch, not the size of the arrays
# in it, and a 2×2 keeps the run time next to nothing beside the compilation.
wide_set(k::Integer) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(
    Tuple(fill(T(i), 2, 2) for i in 1:k))

# The shape a consumer actually holds. The leaves are the same; what differs is the method
# `parameterlayout` enters through.
wrapped_set(k::Integer) = NetworkParameters(wide_set(k))

const SHAPES = (:bare => wide_set, :wrapped => wrapped_set)

# `time()` and not `@elapsed`: the point is the very first call, and `@elapsed` in a loop reports the
# second.
#
# **`invokelatest` and not a direct `f(args...)`, and that is the whole measurement rather than a
# detail.** Compiling `first_call` itself infers through the call in its body — so with a direct call the
# 1.6 s of inference that `parameterlayout` on a 369-child branch costs is spent while `first_call` is
# being compiled, *before* `t = time()` ever runs, and the figure printed is 0.00 s. This script read
# 0.00 s for every width and every column on Julia 1.13 until it was written this way. `invokelatest`
# makes the call opaque, so there is nothing left for the caller's own compilation to do first.
#
# That is the same lesson as the one-process-per-width fix below, from the other end: the harness has to
# be arranged so that the cost cannot have been paid somewhere the clock is not looking.
function first_call(f, args...)
    t = time()
    Base.invokelatest(f, args...)
    round(time() - t; digits = 2)
end

# From inside a function, for the reason `test/flatten_tests.jl` gives: that is the claim that matters,
# an optimizer's inner loop rather than the top level of a script.
_flatten_allocs(buf, ps, layout) = @allocated flatten!(buf, ps, layout)
_unflatten_allocs(dest, layout, v) = @allocated unflatten!(dest, layout, v)

# `mapparameters!` needs a **zero-argument** closure and not the two above. `@allocated f(a, b)` lowers
# to `Base.allocated(f, a, b)`, and `mapparameters!` is a `Vararg` method — so the splat inside
# `Base.allocated` allocates on its own account, which both invents bytes the walk does not spend and
# hides the ones it does. `flatten!` is not a `Vararg` method, which is why the two above can be written
# the obvious way. A thunk has nothing to splat.
_thunk_allocs(thunk) = (thunk(); thunk(); @allocated thunk())

# A destination-and-source walk, which is what an optimizer's elementwise primitives are, and the third
# thing this sweep is for: `mapparameters!` took a `values(...)` tuple per branch per argument until the
# walks began indexing them in place.
_touch!(d, s) = (@inbounds d[begin] = s[begin]; nothing)

# `map(zero, ·)` is the floor the other columns are read against, and it is `Base`'s function rather than
# one of this package's — so it is measured on the `NamedTuple` in both shapes, and the wrapped column
# reports the same figure as the bare one by construction.
_leaves(ps::NetworkParameters) = params(ps)
_leaves(ps) = ps

# 32 is where `Base` stops unrolling a tuple; 369 is the parameter set of the MNIST transformer in
# `scripts/geometric_optimizers/mnist.jl` of GMLDatasets.jl, which is the width that made the defect
# worth fixing rather than noting.
const WIDTHS = (16, 32, 48, 64, 128, 369)

header() = println(rpad("shape", 10), rpad("children", 10), rpad("layout", 9), rpad("map(zero)", 11),
                   rpad("mapparams", 11), rpad("flatten", 9), rpad("unflatten", 11),
                   "allocs flatten!/unflatten!/mapparameters!")

# One width in one shape, in a process that has compiled nothing for either. Within the row the order
# still matters — `flatten` builds a layout and would absorb `parameterlayout`, and `mapparameters`
# would absorb the rest — so the rows are ordered cheapest-first and `flatten` comes after both.
function sweep_one(shape::Symbol, k)
    build = last(SHAPES[findfirst(s -> first(s) === shape, SHAPES)])
    ps = build(k)
    t_layout = first_call(parameterlayout, ps)
    t_map = first_call(x -> map(zero, x), _leaves(ps))
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
    a_mapp = _thunk_allocs(() -> mapparameters!(_touch!, dest, ps))

    println(rpad(shape, 10), rpad(k, 10), rpad(t_layout, 9), rpad(t_map, 11), rpad(t_mapp, 11),
            rpad(t_flat, 9), rpad(t_unflat, 11), a_flat, "/", a_unflat, "/", a_mapp)
    flush(stdout)
end

# `--in-process shape k` is what the child processes are invoked with, and is not for interactive use.
function fan_out()
    header()
    flush(stdout)
    for (shape, _) in SHAPES, k in WIDTHS
        run(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
                      $(@__FILE__) --in-process $shape $k`))
    end
end

if "--in-process" in ARGS
    i = findfirst(==("--in-process"), ARGS)
    sweep_one(Symbol(ARGS[i + 1]), parse(Int, ARGS[i + 2]))
else
    fan_out()
end
