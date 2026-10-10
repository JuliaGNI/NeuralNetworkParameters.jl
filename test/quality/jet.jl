# JET.jl optimisation analysis of the hot paths. See https://github.com/aviatesk/JET.jl.
#
# The entry points are the functions of `src/` that a test file asserts with `@allocated`:
# `flatten!` and `unflatten!` (`flatten.jl`, `integration/wide_branches.jl`), `unflatten` (`flatten.jl`), the
# walks `foreachparameters`, `mapparameters!`, `mapstorage!`, `foldparameters` and `foldstorage`,
# `parameter_eltype`, the `NetworkParameters` constructor, `flatten(ps)`, `flatten(T, ps)` and the
# `rrule` of each (`integration/wide_branches.jl`), and the pullback of the `unflatten` rule (`derivatives.jl`).
#
# Each entry point has one line per element type that a test outside `test/quality/` passes in a
# direct call to the method the `@allocated` call reaches, at the argument types of one such call,
# named on the line above it. Another container at the same element type gets no line, and neither
# does an element type that reaches the entry point only through another function. A set with no
# numeric leaf has the element type `Union{}`, and gets a line where a test passes one directly.
# A `Dual` line uses the tag `Nothing`, because the tag of a test's `ForwardDiff.gradient` holds a
# closure of that test file.

using NeuralNetworkParameters
using ChainRulesCore
using ForwardDiff
using JET
using Test

include("../helpers/wrapper_types.jl")

const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

# the leaf of `leaves.jl` that keeps its numbers behind an interface that is not an array's
struct Blocks
    data::Vector{Float64}
end
NNP.freeparameters(b::Blocks) = b.data
NNP.rebuild(::Blocks, data) = Blocks(data)

# the 48-child branches of `integration/wide_branches.jl`
function wide_set(k)
    NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(Tuple(fill(Float32(i), 2, 2)
    for i in 1:k))
end

# the functions handed to the walks, in the shape of those in `integration/wide_branches.jl` and `walk.jl`
bump2(_, _) = nothing
fold2(acc, x, y) = acc + sum(x) * sum(y)
addto!(d, s) = (d .+= s)

if JET_WORKS
    @testset "JET report_opt" begin
        ps32 = wide_set(48)
        v32, l32 = flatten(ps32)
        _, pb32 = ChainRulesCore.rrule(unflatten, l32, v32)
        ps64 = sample_parameters()
        v64, l64 = flatten(ps64)
        noleaf = NetworkParameters(NamedTuple())

        # `flatten.jl` "the in-place forms do not allocate" (`Float64`); `integration/wide_branches.jl` "the
        # in-place forms do not allocate at $k children" (`Float32`)
        @test isempty(JET.get_reports(JET.report_opt(flatten!,
            (typeof(v64), typeof(ps64), typeof(l64));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(flatten!,
            (typeof(v32), typeof(ps32), typeof(l32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(unflatten!,
            (typeof(ps64), typeof(l64), typeof(v64));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(unflatten!,
            (typeof(ps32), typeof(l32), typeof(v32));
            target_modules = (NeuralNetworkParameters,))))

        # `flatten.jl` "wrapping in a NetworkParameters costs nothing extra" (`Float64`),
        # "unflatten is generic in the element type" (`Float32`, `Int`); `derivatives.jl`
        # "unflatten carries Duals" (a `Dual` of `Float64`) and "a gradient from the `flatten`
        # rule is not converted again ($T)" at `Float32` (a `Dual` of `Float32`, under
        # `ForwardDiff.gradient`)
        wrapped = NetworkParameters((a = [1.0, 2.0], b = [3.0]))
        w, lw = flatten(wrapped)
        _, la = flatten(NetworkParameters((a = [1.0, 2.0],)))
        psr = NetworkParameters((L1 = (W = [0.1 0.2; 0.3 0.4], b = [0.5, 0.6]),
            L2 = (W = [0.7 0.8], b = [0.9])))
        vr, lr = flatten(psr)
        dual = ForwardDiff.Dual.(vr, 1.0)
        vs32, ls32 = flatten(NetworkParameters((L1 = (S = Sym(Float32[1, 2, 3], 2),),)))
        vs64, ls64 = flatten(NetworkParameters((L1 = (S = Sym([1.0, 2.0, 3.0], 2),),)))
        dual32 = ForwardDiff.Dual.(vs32, 1.0f0)  # a `Dual` of `Float32`
        dual64 = ForwardDiff.Dual.(vs64, 1.0)
        @test isempty(JET.get_reports(JET.report_opt(unflatten, (typeof(lw), typeof(w));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(
            unflatten, (typeof(la), Vector{Float32});
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(unflatten, (typeof(la), Vector{Int});
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(unflatten, (typeof(lr), typeof(dual));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(
            unflatten, (typeof(ls32), typeof(dual32));
            target_modules = (NeuralNetworkParameters,))))

        # the walks: `integration/wide_branches.jl` "the walks that take two sets do not allocate at $k
        # children" (`Float32`);
        # `walk.jl` "the in-place walks pair children by key, not by position", "in-place walks",
        # "a zipped fold pairs the leaves" and "foldstorage folds the storage and not the leaf"
        # (`Float64`)
        d = (a = [0.0], b = [0.0])
        ab = NetworkParameters((a = [1.0], b = [2.0]))
        dest = NetworkParameters((L = (x = [1.0, 2.0],),))
        sym = NetworkParameters((S = sample_sym(),))
        zipped = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
        @test isempty(JET.get_reports(JET.report_opt(foreachparameters,
            (typeof(bump2), typeof(ps32), typeof(ps32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(foreachparameters,
            (typeof(copyto!), typeof(d), typeof(ab));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(mapparameters!,
            (typeof(bump2), typeof(ps32), typeof(ps32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(mapparameters!,
            (typeof(addto!), typeof(dest), typeof(dest));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(mapstorage!,
            (typeof(bump2), typeof(ps32), typeof(ps32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(mapstorage!,
            (typeof(addto!), typeof(sym), typeof(sym));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(foldparameters,
            (typeof(fold2), Float32, typeof(ps32), typeof(ps32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(foldparameters,
            (typeof(fold2), Float64, typeof(zipped), typeof(zipped));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(foldstorage,
            (typeof(fold2), Float32, typeof(ps32), typeof(ps32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(foldstorage,
            (typeof(fold2), Float64, typeof(ps64), typeof(ps64));
            target_modules = (NeuralNetworkParameters,))))

        # `integration/wide_branches.jl` "the element type of a branch of $k children costs nothing"
        # (`Float32`); `leaves.jl` "parameter_eltype" (`Float64`, and the mixed branch that raises)
        # and "parameter_eltype does not raise where freeparameters does" (`Union{}`);
        # `derivatives.jl` "unflatten carries Duals"
        a1 = NetworkParameters((a = [1.0],))
        mixed = (a = Float32[1], b = [2.0])
        psd = unflatten(lr, dual)
        @test isempty(JET.get_reports(JET.report_opt(parameter_eltype, (typeof(ps32),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(parameter_eltype, (typeof(mixed),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(parameter_eltype, (typeof(psd),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(
            parameter_eltype, (typeof(NamedTuple()),);
            target_modules = (NeuralNetworkParameters,))))

        # `integration/wide_branches.jl` (`Float32`); `parameters.jl` "the element type" (`Float64`, `Union{}`)
        @test isempty(JET.get_reports(JET.report_opt(NetworkParameters, (typeof(ps32),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(
            NetworkParameters, (typeof(params(a1)),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(
            NetworkParameters, (typeof(NamedTuple()),);
            target_modules = (NeuralNetworkParameters,))))

        # `integration/wide_branches.jl` (`Float32`); `flatten.jl` "round trip" (`Float64`) and "the empty set
        # flattens to an empty vector" (`Union{}`); `derivatives.jl` "a gradient from the
        # `flatten` rule is not converted again ($T)", whose `f` flattens a set of `Dual`s of `T`
        # under `ForwardDiff.gradient`
        psd32 = unflatten(ls32, dual32)
        psd64 = unflatten(ls64, dual64)
        @test isempty(JET.get_reports(JET.report_opt(flatten, (typeof(psd32),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(flatten, (typeof(psd64),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(flatten, (typeof(ps32),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(flatten, (typeof(ps64),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(flatten, (typeof(noleaf),);
            target_modules = (NeuralNetworkParameters,))))

        # `integration/wide_branches.jl` (`Float32`); `flatten.jl` "element type follows the parameters" (a
        # `Float64` set into `Float32`); `leaves.jl` "a leaf that is not an array contributes
        # no element type" (`Float64`)
        blocks = NetworkParameters((L1 = (B = Blocks([1.0, 2.0]),),))
        @test isempty(JET.get_reports(JET.report_opt(
            flatten, (Type{Float32}, typeof(ps32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(flatten, (Type{Float32}, typeof(a1));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(
            flatten, (Type{Float64}, typeof(blocks));
            target_modules = (NeuralNetworkParameters,))))

        # `integration/wide_branches.jl` (`Float32`); `derivatives.jl` "`nothing` is a structural zero for the
        # `flatten` rules too" (`Float64`)
        @test isempty(JET.get_reports(JET.report_opt(ChainRulesCore.rrule,
            (typeof(flatten), typeof(ps32)); target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(ChainRulesCore.rrule,
            (typeof(flatten), typeof(psr)); target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(ChainRulesCore.rrule,
            (typeof(flatten), Type{Float32}, typeof(ps32));
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(ChainRulesCore.rrule,
            (typeof(flatten), Type{Float64}, typeof(psr));
            target_modules = (NeuralNetworkParameters,))))

        # the pullback of the `unflatten` rule: `derivatives.jl` "the pullback does not pay for the
        # depth of the tree" (`Float64`); `integration/wide_branches.jl` "the pullback of unflatten reaches a
        # branch of $k children" (`Float32`)
        L = [1.0, 2.0]
        flat = NetworkParameters((a = L, b = L, c = L, d = L))
        wf, lf = flatten(flat)
        _, pb64 = ChainRulesCore.rrule(unflatten, lf, wf)
        @test isempty(JET.get_reports(JET.report_opt(pb64, (typeof(flat),);
            target_modules = (NeuralNetworkParameters,))))
        @test isempty(JET.get_reports(JET.report_opt(pb32, (typeof(ps32),);
            target_modules = (NeuralNetworkParameters,))))
    end
else
    @test_skip "JET does not work on Julia $VERSION" # aviatesk/JET.jl#681
end
