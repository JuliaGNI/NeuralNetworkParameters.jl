using NeuralNetworkParameters
using ForwardDiff
using Zygote
using ChainRulesCore
using Test

include("wrapper_types.jl")

ps = NetworkParameters((L1 = (W = [0.1 0.2; 0.3 0.4], b = [0.5, 0.6]),
    L2 = (W = [0.7 0.8], b = [0.9])))
x = [1.0, 2.0]
model(p) = tanh.(p.L2.W * tanh.(p.L1.W * x .+ p.L1.b) .+ p.L2.b)
loss(p) = sum(model(p))

v, layout = flatten(ps)

@testset "unflatten carries Duals" begin
    d = ForwardDiff.Dual.(v, 1.0)
    psd = unflatten(layout, d)
    @test eltype(psd.L1.W) <: ForwardDiff.Dual
    @test psd isa NetworkParameters
    # and the element type follows them onto the type of the set
    @test psd isa NetworkParameters{<:ForwardDiff.Dual}
    @test NeuralNetworkParameters.parameter_eltype(psd) === eltype(psd.L1.W)
end

@testset "ForwardDiff on the flat form, read back structured" begin
    g_flat = ForwardDiff.gradient(w -> loss(unflatten(layout, w)), v)
    g = unflatten(layout, g_flat)
    @test g isa NetworkParameters
    @test size(g.L1.W) == size(ps.L1.W)
    # a finite difference on one entry, as an independent check
    h = 1e-6
    vp = copy(v)
    vp[1] += h
    vm = copy(v)
    vm[1] -= h
    @test g_flat[1] ≈ (loss(unflatten(layout, vp)) - loss(unflatten(layout, vm))) / 2h atol = 1e-7
end

@testset "the gradient of a NetworkParameters is a NetworkParameters" begin
    # supplied by the ZygoteRules extension; without it the reverse pass returns a tangent for the
    # struct instead, which nothing downstream can consume
    g = Zygote.gradient(loss, ps)[1]
    @test g isa NetworkParameters
    @test keys(g) == keys(ps)
end

@testset "flat and structured gradients agree" begin
    g_flat = ForwardDiff.gradient(w -> loss(unflatten(layout, w)), v)
    g_struct = Zygote.gradient(loss, ps)[1]
    g_from_flat = unflatten(layout, g_flat)
    @test g_from_flat.L1.W ≈ g_struct.L1.W
    @test g_from_flat.L1.b ≈ g_struct.L1.b
    @test g_from_flat.L2.W ≈ g_struct.L2.W
    @test g_from_flat.L2.b ≈ g_struct.L2.b
end

@testset "Zygote differentiates through unflatten" begin
    # this is the rrule; the answer has to match differentiating the flat form directly
    g_zygote = Zygote.gradient(w -> loss(unflatten(layout, w)), v)[1]
    g_forward = ForwardDiff.gradient(w -> loss(unflatten(layout, w)), v)
    @test g_zygote ≈ g_forward
end

@testset "Zygote differentiates through flatten" begin
    g = Zygote.gradient(p -> sum(abs2, first(flatten(p))), ps)[1]
    @test g isa NetworkParameters
    @test g.L1.W ≈ 2 .* ps.L1.W
    @test g.L2.b ≈ 2 .* ps.L2.b
end

@testset "an untouched layer gives a zero block, not an error" begin
    two = NetworkParameters((a = [1.0, 2.0], b = [3.0, 4.0]))
    _, l = flatten(two)
    # the loss ignores `b` entirely, so the reverse pass says nothing about it
    g = Zygote.gradient(w -> sum(unflatten(l, w).a), first(flatten(two)))[1]
    @test g == [1.0, 1.0, 0.0, 0.0]
end

@testset "an untouched nested branch gives a zero block" begin
    # a whole layer the loss never mentions, not just one array of it. The cotangent is a structural
    # zero at the *branch*, which is a case of its own: handling a zero leaf does not cover it.
    ps2 = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0], b = [5.0])))
    v2, l2 = flatten(ps2)
    g = Zygote.gradient(w -> sum(unflatten(l2, w).L1.W), v2)[1]
    @test g == [1.0, 1.0, 0.0, 0.0, 0.0]
end

@testset "structured leaves differentiate through the flat form" begin
    sp = NetworkParameters((S = sample_sym(),))
    sv, sl = flatten(sp)
    f(w) = sum(abs2, unflatten(sl, w).S.S)
    @test ForwardDiff.gradient(f, sv) ≈ 2 .* sv
    @test Zygote.gradient(f, sv)[1] ≈ 2 .* sv
end

@testset "a gradient tree with a hole still has an element type" begin
    # the `ZygoteRules` extension rewraps the pullback's `NamedTuple`, in which an untouched layer is
    # `nothing`. Deriving the element type in the constructor must not trip over the hole.
    two = NetworkParameters((a = [1.0, 2.0], b = [3.0, 4.0]))
    g = Zygote.gradient(p -> sum(p.a), two)[1]
    @test g isa NetworkParameters{Float64}
    @test g.b === nothing
    @test g.a == [1.0, 1.0]
end

@testset "a set the reverse pass never touched comes back as `nothing`" begin
    # The extension rewraps the pullback's `NamedTuple`, and when the reverse pass touched none of the
    # parameters there is no `NamedTuple` to rewrap: one `nothing` stands for the whole set and cannot
    # be spread over `keys(ps)`. It used to raise `MethodError: no method matching _values(::Nothing)`
    # here, where the bare `NamedTuple` the set wraps returns `nothing` quite happily — so a loss that
    # reads only the layout, or a frozen sub-network, failed on the wrapper and not on its contents.
    @test Zygote.gradient(p -> 1.0, ps) === (nothing,)
    @test Zygote.gradient(p -> Float64(length(flatten(p)[2])), ps) === (nothing,)
    @test Zygote.gradient(p -> 1.0, params(ps)) === (nothing,)      # what it has to agree with
    # a set that *was* touched still comes back wrapped, holes and all
    @test Zygote.gradient(p -> sum(p.L1.W), ps)[1] isa NetworkParameters
end

@testset "`nothing` is a structural zero for the `flatten` rules too" begin
    # `_normalized` is explicit that `nothing` means "no derivative"; the `flatten` pullback took
    # `ZeroTangent` and not `nothing`, so the two spellings disagreed on the one path Zygote does not
    # take — a caller invoking the rule directly got a `MethodError` instead of a zero.
    (_, _), pb1 = ChainRulesCore.rrule(flatten, ps)
    @test pb1(nothing) == (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent())
    (_, _), pb2 = ChainRulesCore.rrule(flatten, Float64, ps)
    @test pb2(nothing)[3] == ChainRulesCore.ZeroTangent()
end

@testset "a structural zero contributes nothing to the promotion" begin
    @test NeuralNetworkParameters.parameter_eltype(ChainRulesCore.ZeroTangent()) === Union{}
    @test NeuralNetworkParameters.parameter_eltype((a = [1.0f0], b = ChainRulesCore.ZeroTangent())) ===
          Float32
end

# The pullback walks the layout against a cotangent whose shape it cannot know, and it used to do so
# with `for k in keys(l.children)`: indexing a heterogeneous `NamedTuple` with a runtime `Symbol`
# yields the union of the branch's child layout types, so `_accumulate!` was dispatched dynamically
# once per child. The cost grew with the number of layers — on Julia 1.13, 5312 bytes for an
# eight-layer set whose gradient vector is 192 — and none of the value tests above could see it.
_pullback_allocs(pb, Δ) = @allocated pb(Δ)

@testset "the pullback does not pay for the depth of the tree" begin
    # the same four leaves, once flat and once one to a layer. The walk is over layouts known at
    # compile time, so the layering cannot show up in the bill. Before it was written this way the two
    # cost 1088 and 1664 bytes on Julia 1.13, against a 128-byte answer — the gap is the dynamic
    # dispatch a runtime-`Symbol` lookup into `l.children` forced at every child. Both now sit at the
    # answer's own cost, 128 bytes on 1.11, 1.12 and 1.13; they are still compared with each other
    # rather than with `zero`, because it is the *depth* that must not show up and not the figure.
    L = [1.0, 2.0]
    function _cost(p)
        w, l = flatten(p)
        _, pb = ChainRulesCore.rrule(unflatten, l, w)
        Δ = unflatten(l, w)
        _pullback_allocs(pb, Δ)                       # warm up the call and the measurement
        _pullback_allocs(pb, Δ)
    end
    flat = NetworkParameters((a = L, b = L, c = L, d = L))
    layered = NetworkParameters((L1 = (a = L,), L2 = (b = L,), L3 = (c = L,), L4 = (d = L,)))
    @test _cost(layered) == _cost(flat)
end

@testset "the pullback does not pay to find a structured leaf's storage" begin
    # `_storage_field` asks which of the prototype's fields `freeparameters` handed back, and used to
    # ask it with a `for` over `fieldnames(typeof(prototype))`: `getfield` at a runtime `Symbol` comes
    # back as the union of the type's field types, so the comparison was dispatched dynamically once
    # per field. Written out, the names are literals and every one of them is constant. Measured on
    # `Sym`, whose storage is one of two fields: 176 bytes a call as a loop, 144 written out.
    #
    # Bounded rather than equated, on both counts: the figure is the whole pullback's and not this
    # walk's, and `@allocated` jitters on Windows.
    sp = NetworkParameters((L1 = (W = sample_sym(), b = [1.0, 2.0]),))
    sv, sl = flatten(sp)
    _, spb = ChainRulesCore.rrule(unflatten, sl, sv)
    Δs = ChainRulesCore.Tangent{Any}(params = ChainRulesCore.Tangent{Any}(
        L1 = ChainRulesCore.Tangent{Any}(
            W = ChainRulesCore.Tangent{Any}(S = [1.0, 1.0, 1.0], n = ChainRulesCore.NoTangent()),
            b = [1.0, 1.0])))
    _pullback_allocs(spb, Δs)                          # warm up the call and the measurement
    @test _pullback_allocs(spb, Δs) ≤ 160
    # and it still reads the right numbers into the right places
    @test spb(Δs)[3] == ones(5)
end

@testset "a tuple branch reads a cotangent shorter than itself" begin
    # `_cotangent_head`/`_cotangent_tail` replaced an index into the cotangent tuple; a reverse pass
    # that filled only the leading positions still has to leave the rest as structural zeros
    tp = NetworkParameters((t = ([1.0, 2.0], [3.0], [4.0]),))
    tv, tl = flatten(tp)
    acc(Δ) = (d = zero(tv); NeuralNetworkParameters._accumulate_cotangent!(d, tl, Δ); d)
    @test acc((t = ([1.0, 1.0], [1.0], [1.0]),)) == [1.0, 1.0, 1.0, 1.0]
    @test acc((t = ([1.0, 1.0],),)) == [1.0, 1.0, 0.0, 0.0]      # runs out into zeros
    @test acc((t = (),)) == [0.0, 0.0, 0.0, 0.0]
    @test Zygote.gradient(w -> sum(unflatten(tl, w).t[2]), tv)[1] == [0.0, 0.0, 1.0, 0.0]
end

@testset "storage in more than ten blocks is matched by name and by position" begin
    # `_matching_storage` used to reach for its positional case with `ntuple` over a runtime length,
    # which Base stops inferring past ten
    storage = ntuple(i -> [Float64(i)], 12)
    backing = NamedTuple{ntuple(i -> Symbol(:b, i), 12)}(storage)
    @test NeuralNetworkParameters._matching_storage(storage, backing) === storage
    @test isconcretetype(only(Base.return_types(NeuralNetworkParameters._matching_storage,
        Tuple{typeof(storage), typeof(backing)})))
    # a backing that stops short leaves the remaining blocks as structural zeros
    short = NamedTuple{ntuple(i -> Symbol(:b, i), 3)}(ntuple(i -> [Float64(i)], 3))
    @test NeuralNetworkParameters._matching_storage(storage, short) ===
          (short.b1, short.b2, short.b3, ntuple(_ -> nothing, 9)...)
end
