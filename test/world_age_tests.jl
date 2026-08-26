# The generated walks still work when the sources are evaluated rather than loaded from a cache.
#
# Every across-children walk is a `@generated` function, and four of them call `_children_arity` from
# the *generator* body — as does, for a walk over more than one set, the chain of `_branch_keys`,
# `_child_expr` and their two error helpers. A generator runs in the world age of its own method
# definition, so a helper defined further down the file is invisible to it and the generator raises
# `MethodError: no method matching _children_arity(…)  The applicable method may be too new`. That is
# a real defect and it was one: `_children_arity` used to sit at the foot of `src/walk.jl`, below
# `_map_zip` and `_foreach_zip`.
#
# It cannot be caught in-process. Loading a precompiled package deserialises every method in the
# module into one world age, which is exactly the condition that hides it — so the whole of the rest
# of this suite passes with the bug present. `--compiled-modules=no` is what evaluates the sources,
# and a subprocess is the only way to ask for it.
#
# Cheap enough to keep: the package's dependency is `ChainRulesCore`, so a from-source load is about a
# second. The walks exercised below are the five whose generators call `_children_arity` — `_map_zip`,
# `_foreach_zip`, `_fold_zip` and, through `flatten`, `_flatten_children!` and `_unflatten_children!`.
# `_foreach_zip` and `_fold_zip` are run at **two** arities, because the key and index helpers a
# further set needs are only reached at the second — one set has no second set to pair against, so an
# arity-one walk would leave exactly that chain of helpers untried.
#
# For the same reason the file ends with three walks that are *wrong*. `_arity_error`,
# `_child_keys_error` and `_no_folded_skip_error` are reached from a generator only when a caller has
# mismatched something, so every successful walk above leaves all three untried — and each could be
# moved below its caller with this file still passing. What a misplaced one produces is a `MethodError`
# saying the method "may be too new" in place of the `ArgumentError` the caller is owed, so the error
# *type* is the assertion and not merely that something was thrown.

using Test

const _WORLD_AGE_PROGRAM = """
using NeuralNetworkParameters

ps = (L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],))

# `_map_zip`
mapparameters(zero, ps) == (L1 = (W = [0.0 0.0], b = [0.0]), L2 = (W = [0.0;;],)) || exit(1)
mapparameters(+, ps, ps).L1.b == [6.0] || exit(2)

# `_foreach_zip` at arity one
let n = Ref(0)
    foreachparameters(_ -> n[] += 1, ps)
    n[] == 3 || exit(3)
end

# `_foreach_zip` at arity two, which is what reaches `_branch_keys` and `_child_expr` from a generator
let dest = mapparameters(zero, ps)
    mapparameters!(copyto!, dest, ps)
    dest == ps || exit(8)
end

# and the `nothing` skip, which is the other branch of `_child_expr`
let n = Ref(0)
    foreachparameters((_, _) -> n[] += 1, ps, nothing)
    n[] == 0 || exit(9)
end

# `_flatten_children!`, `_unflatten_children`, `_unflatten_children!`, `_layout`,
# `_promote_eltypes`
v, layout = flatten(ps)
v == [1.0, 2.0, 3.0, 4.0] || exit(4)
unflatten(layout, v) == ps || exit(5)
let qs = mapparameters(zero, ps)
    unflatten!(qs, layout, v)
    qs == ps || exit(6)
end

# `_fold_zip` at arity one, and at the arity that reaches `_branch_keys` and `_child_expr` from the
# fold's own generator; `foldstorage` is its other `_fold_step`
foldparameters((a, x) -> a + length(x), 0, ps) == 4 || exit(7)
foldparameters((a, x, y) -> a + sum(x) * sum(y), 0.0, ps, ps) == 34.0 || exit(10)
foldstorage((a, x, y) -> a + sum(x) * sum(y), 0.0, ps, ps) == 34.0 || exit(11)

# The three error helpers, each reached from a generator and none of them by a walk that succeeds
raises_argument_error(f) = try
    f()
    false
catch e
    e isa ArgumentError
end

# `_arity_error`: two children on the left of `L1` and one on the right
raises_argument_error(() -> foreachparameters((_, _) -> nothing, ps,
    (L1 = (W = [1.0 2.0],), L2 = (W = [4.0;;],)))) || exit(12)
# `_child_keys_error`: the same keys in the other order, which is the crossing-over it exists to stop
raises_argument_error(() -> foreachparameters((_, _) -> nothing, ps,
    (L2 = ps.L2, L1 = ps.L1))) || exit(13)
# `_no_folded_skip_error`: the fold declining to half-reduce a set it was not given
raises_argument_error(() -> foldparameters((a, x, y) -> a, 0.0, ps, nothing)) || exit(14)

exit(0)
"""

@testset "the generated walks survive a from-source load" begin
    project = dirname(@__DIR__)
    cmd = `$(Base.julia_cmd()) --startup-file=no --compiled-modules=no --project=$project
           -e $_WORLD_AGE_PROGRAM`
    @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
end
