# What `parameterlayout` costs in each of the two shapes a caller can hold it in, at the two widths and
# the one nested set the 0.2.3 CHANGELOG entry quotes.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/leaf_layout_cost.jl
#
# This is the harness behind that entry's table, committed for the reason `wide_branch_cost.jl` is: a
# number is reproducible only where the code that produced it is. The two scripts overlap at 369 leaves
# and are kept apart because they answer different questions. That one sweeps the *width* of one branch
# across the whole walk — `flatten`, `unflatten`, `mapparameters`, and the allocation ceilings with
# them — and stops at 369 because that is the parameter set that made the 0.2.2 defect worth fixing.
# This one holds the walk fixed and varies only what a layout is *held in*, because that is where the
# cost 0.2.3 removed actually lived, and it reaches 768 leaves and a nested set to show that neither the
# width nor the nesting is what drives it.
#
# **Every row runs in a process of its own**, for the reason that script's header gives at length:
# compilation is shared across a session, so a row that runs second is measuring what the first already
# paid for. There is nothing to interleave here — every row calls the same function — which makes the
# hazard worse rather than better.
#
# The figures in the CHANGELOG were taken on Julia 1.11, which is the compat floor and was the version
# that behaved differently before this release. Pass another with `julia +1.12` and so on; 1.11, 1.12
# and 1.13 now agree to within a tenth of a second on the wrapped 369 row.

using NeuralNetworkParameters

const T = Float32

# `time()` and `invokelatest`, not `@elapsed` and a direct call, and that is the measurement rather
# than a detail: compiling this function would otherwise infer through the call in its body, so the
# cost would be spent before the clock starts and every row would read 0.00 s. `wide_branch_cost.jl`
# read exactly that on Julia 1.13 until it was written this way.
function first_call(f, args...)
    t = time()
    Base.invokelatest(f, args...)
    round(time() - t; digits = 2)
end

# All leaves the same tiny matrix: this measures the shape a layout is held in, not the size of the
# arrays in it, and a 2×2 keeps the run time next to nothing beside the compilation.
flat_set(k::Integer) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(
    Tuple(fill(T(i), 2, 2) for i in 1:k))

# The keys differ per branch, so the 16 branches are 16 distinct `NamedTuple` types rather than one
# compiled once and reused. A nested set whose branches share a type costs a fraction of this and would
# make the nesting look free for the wrong reason.
branch(j::Integer, k::Integer) = NamedTuple{Tuple(Symbol("L", j, "p", i) for i in 1:k)}(
    Tuple(fill(T(i), 2, 2) for i in 1:k))

nested_set(o::Integer, k::Integer) = NamedTuple{Tuple(Symbol("L", j) for j in 1:o)}(
    Tuple(branch(j, k) for j in 1:o))

# Nodes in the type tree of the layout, which is the quantity the entry's "2.5 times the type" is.
# `length(string(T))` cannot be used for it: Julia elides long types when printing, so a 369-child
# layout prints at about seven characters per child whether or not the leaves' array types are in it.
type_nodes(x) = x isa DataType ? 1 + sum(type_nodes(p) for p in x.parameters; init = 0) : 1

const CASES = (
    ("369 leaves, flat", () -> flat_set(369)),
    ("768 leaves, flat", () -> flat_set(768)),
    ("16 × 24 nested", () -> nested_set(16, 24))
)

function run_one(case::Int, wrapped::Bool)
    name, build = CASES[case]
    ps = wrapped ? NetworkParameters(build()) : build()
    t = first_call(parameterlayout, ps)
    println(rpad(name, 18), rpad(wrapped ? "wrapped" : "bare", 10), rpad(t, 8),
        type_nodes(typeof(parameterlayout(ps))))
    flush(stdout)
end

function fan_out()
    println(rpad("set", 18), rpad("held in", 10), rpad("layout", 8), "type nodes")
    flush(stdout)
    for case in eachindex(CASES), wrapped in (false, true)
        run(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
                      $(@__FILE__) --in-process $case $wrapped`))
    end
end

if "--in-process" in ARGS
    i = findfirst(==("--in-process"), ARGS)
    run_one(parse(Int, ARGS[i + 1]), parse(Bool, ARGS[i + 2]))
else
    fan_out()
end
