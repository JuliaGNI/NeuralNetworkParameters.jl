using NeuralNetworkParameters
using ForwardDiff
using Zygote
using ZygoteRules
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
    @test NeuralNetworkParameters.parameter_eltype((
        a = [1.0f0], b = ChainRulesCore.ZeroTangent())) ===
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
    layered = NetworkParameters((
        L1 = (a = L,), L2 = (b = L,), L3 = (c = L,), L4 = (d = L,)))
    @test _cost(layered) == _cost(flat)
end

# One structured leaf beside a plain one, its cotangent given over the leaf's own fields: what the
# pullback costs and what it reads.
function _structured_leaf_pullback(leaf, Δleaf)
    sp = NetworkParameters((L1 = (W = leaf, b = [1.0, 2.0]),))
    sv, sl = flatten(sp)
    _, spb = ChainRulesCore.rrule(unflatten, sl, sv)
    Δs = ChainRulesCore.Tangent{Any}(params = ChainRulesCore.Tangent{Any}(
        L1 = ChainRulesCore.Tangent{Any}(W = Δleaf, b = [1.0, 1.0])))
    _pullback_allocs(spb, Δs)                          # warm up the call and the measurement
    _pullback_allocs(spb, Δs), spb(Δs)[3]
end

@testset "the pullback does not pay for the fields a structured leaf carries" begin
    # `_storage_component` asks which of the prototype's fields `freeparameters` handed back and reads
    # the cotangent's component for it, in one generated body. Asked with a `for` over
    # `fieldnames(typeof(prototype))` instead, `getfield` at a runtime `Symbol` comes back as the union
    # of the type's field types, so the comparison was dispatched dynamically once per field — and the
    # reverse pass paid for every field the leaf carried rather than for the one holding its numbers.
    #
    # `Sym` keeps its three numbers in the first of two fields and `Padded` the same three in the last
    # of six. They are compared with each other and not with a figure, for the reason the depth test
    # above is: it is the *field count* that must not show up, and `@allocated` jitters on Windows. As
    # a loop these cost 160 and 512 bytes on Julia 1.13; both now sit at 96.
    cost_sym, read_sym = _structured_leaf_pullback(sample_sym(),
        ChainRulesCore.Tangent{Any}(S = ones(3), n = ChainRulesCore.NoTangent()))
    cost_padded, read_padded = _structured_leaf_pullback(sample_padded(),
        ChainRulesCore.Tangent{Any}(
            a = ChainRulesCore.NoTangent(), b = ChainRulesCore.NoTangent(),
            c = ChainRulesCore.NoTangent(), d = ChainRulesCore.NoTangent(),
            e = ChainRulesCore.NoTangent(), S = ones(3)))
    @test cost_padded == cost_sym
    # and both still read the right numbers into the right places
    @test read_sym == ones(5)
    @test read_padded == ones(5)
end

@testset "a tuple branch reads a cotangent shorter than itself" begin
    # `_cotangent_head`/`_cotangent_tail` replaced an index into the cotangent tuple; a reverse pass
    # that filled only the leading positions still has to leave the rest as structural zeros
    tp = NetworkParameters((t = ([1.0, 2.0], [3.0], [4.0]),))
    tv, tl = flatten(tp)
    acc(Δ) = (
        d = zero(tv); NeuralNetworkParameters._accumulate_cotangent!(d, tl.inner, Δ); d)
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

# ---------------------------------------------------------------------------------------------------
# The loss sees a `NetworkParameters`, and the storage gradient of a structured leaf
# ---------------------------------------------------------------------------------------------------

@testset "the extension invokes Zygote's generic `pullback`" begin
    # `ext/ZygoteRulesExt.jl` calls this method by `invoke` and does not own it. A Zygote release that
    # changes its signature has to fail here, and not somewhere downstream of a wrong gradient.
    m = which(ZygoteRules.pullback, Tuple{Any, Vararg{Any}})
    @test m.module === Zygote
    @test m.sig == Tuple{typeof(ZygoteRules.pullback), Any, Vararg{Any}}
end

function sample_network(::Type{T}) where {T}
    NetworkParameters((L1 = (W = T[0.1 0.2; 0.3 0.4], b = T[0.5, 0.6]),
        L2 = (W = T[0.7 0.8], b = T[0.9])))
end

network_loss(p, x) = sum(tanh.(p.L2.W * tanh.(p.L1.W * x .+ p.L1.b) .+ p.L2.b))
typed_loss(p::NetworkParameters, x) = network_loss(p, x)

@testset "the loss receives the `NetworkParameters` itself ($T)" for T in (Float32, Float64)
    ps = sample_network(T)
    x = T[1, 2]
    v, l = flatten(ps)
    reference = unflatten(l, ForwardDiff.gradient(w -> network_loss(unflatten(l, w), x), v))
    g = Zygote.gradient(p -> typed_loss(p, x), ps)[1]
    @test g isa NetworkParameters{T}
    @test flatten(g)[1] ≈ flatten(reference)[1]
    y, pb = Zygote.pullback(p -> typed_loss(p, x), ps)
    @test y ≈ network_loss(ps, x)
    @test pb(one(T)) isa Tuple{NetworkParameters{T}}
end

@testset "reading a layer with `p.L1` infers ($T)" for T in (Float32, Float64)
    # without the extension's adjoint for `literal_getproperty`, Zygote differentiates the overloaded
    # `getproperty` with a runtime `Symbol`: the forward pass does not infer, and is slower and
    # allocates more than the same loss on the bare `NamedTuple`
    ps = sample_network(T)
    x = T[1, 2]
    f = p -> network_loss(p, x)
    _, pb = @inferred Zygote.pullback(f, ps)
    @test @inferred(pb(one(T))) isa Tuple{NetworkParameters{T}}
end

@testset "an untouched set comes back as `nothing` ($T)" for T in (Float32, Float64)
    ps = sample_network(T)
    @test Zygote.gradient(p -> one(T), ps) === (nothing,)
    @test Zygote.pullback(p -> one(T), ps)[2](one(T)) === (nothing,)
end

@testset "a layer called `params` keeps its shape ($T)" for T in (Float32, Float64)
    qs = NetworkParameters((params = (W = T[1, 2],),))
    @test Zygote.gradient(p -> sum(p.params.W), qs)[1] ==
          NetworkParameters((params = (W = T[1, 1],),))
    rs = NetworkParameters((params = (W = T[1, 2],), other = (b = T[3],)))
    g = Zygote.gradient(p -> sum(p.params.W) + 2sum(p.other.b), rs)[1]
    @test g == NetworkParameters((params = (W = T[1, 1],), other = (b = T[2],)))
end

@testset "a loss mixing `flatten(p)` with a field of `p` raises" begin
    # the `flatten` rule returns a gradient already in storage coordinates, a `NetworkParameters`, and
    # a field access returns the structural tangent `(params = …,)`; Zygote cannot add the two
    ps = sample_network(Float64)
    @test_throws MethodError Zygote.gradient(p -> sum(first(flatten(p))) + sum(p.L1.W), ps)
end

# `Frob` stands in for a structured leaf with a projection of its own and no storage gradient beyond
# the identity — a manifold element, whose storage is the whole matrix.
struct Frob{T} <: AbstractMatrix{T}
    A::Matrix{T}
end
Base.size(Y::Frob) = size(Y.A)
Base.getindex(Y::Frob, i::Int, j::Int) = Y.A[i, j]
NNP.freeparameters(Y::Frob) = Y.A
NNP.rebuild(::Frob, data) = Frob(data)
ChainRulesCore.ProjectTo(::Frob) = ChainRulesCore.ProjectTo{Frob}()
(::ChainRulesCore.ProjectTo{Frob})(dA::AbstractMatrix) = Frob(Matrix(dA))

@testset "`Zygote.gradient` projects each leaf ($T)" for T in (Float32, Float64)
    # Zygote gives a leaf read twice a dense `Matrix` cotangent, and `gradient` projects its result
    # with `ProjectTo` of the argument; for a `NetworkParameters` that is the projection of each leaf
    ps = NetworkParameters((L1 = (Y = Frob(T[1 2; 3 4]),), L2 = (b = T[1],)))
    g = Zygote.gradient(p -> sum(p.L1.Y) + sum(p.L1.Y), ps)[1]
    @test g.L1.Y isa Frob{T}
    @test g.L1.Y == Frob(fill(T(2), 2, 2))
    @test g.L2 === nothing
    # and directly, where a hole stays a hole and a structural tangent is left alone
    Δ = NetworkParameters((L1 = (Y = ones(T, 2, 2),), L2 = nothing))
    @test ChainRulesCore.ProjectTo(ps)(Δ) ==
          NetworkParameters((L1 = (Y = Frob(ones(T, 2, 2)),),
        L2 = nothing))
    Δs = NetworkParameters((L1 = (Y = (A = ones(T, 2, 2),),), L2 = (b = nothing,)))
    @test ChainRulesCore.ProjectTo(ps)(Δs) == Δs
end

# `Sym`'s projection and storage gradient, as `GeometricOptimizers` writes them for a
# `SymmetricMatrix`: `ProjectTo` is the Frobenius projection, and `storage_gradient` turns a natural
# cotangent `G` into ∂L/∂S — the lower triangle of `G + Gᵀ`, the diagonal counted once. Every call is
# counted, so a test can say that the hook ran once per reverse pass and not once per use of the leaf.
const STORAGE_GRADIENT_CALLS = Ref(0)

_lower(f, n) = [f(i, j) for i in 1:n for j in 1:i]

ChainRulesCore.ProjectTo(A::Sym) = ChainRulesCore.ProjectTo{Sym}(; n = A.n)
function (p::ChainRulesCore.ProjectTo{Sym})(dA::AbstractMatrix)
    Sym(
        _lower((i, j) -> (dA[i, j] + dA[j, i]) / 2, p.n), p.n)
end

function NNP.storage_gradient(A::Sym, G::AbstractMatrix)
    STORAGE_GRADIENT_CALLS[] += 1
    Sym(_lower((i, j) -> i == j ? G[i, i] : G[i, j] + G[j, i], A.n), A.n)
end

function sym_network(::Type{T}) where {T}
    NetworkParameters((L1 = (W = T[0.1 0.2; 0.3 0.4], b = T[0.5, 0.6]),
        L2 = (S = Sym(T[0.7, -0.8, 0.9], 2),)))
end

sym_once(p, x) = sum(tanh.(p.L2.S * tanh.(p.L1.W * x .+ p.L1.b)))
sym_twice(p, x) = sym_once(p, x) + sum(abs2, p.L2.S * x)

@testset "the storage gradient reaches the rewrapped gradient once ($T, $name)" for T in (Float32, Float64),
    (name, f) in (("one use", sym_once), ("two uses", sym_twice))

    ps = sym_network(T)
    x = T[1, -2]
    v, l = flatten(ps)
    # ForwardDiff differentiates the storage itself, so it gives ∂L/∂S with no convention to choose
    reference = ForwardDiff.gradient(w -> f(unflatten(l, w), x), v)
    STORAGE_GRADIENT_CALLS[] = 0
    g = Zygote.pullback(p -> f(p, x), ps)[2](one(T))[1]
    @test STORAGE_GRADIENT_CALLS[] == 1
    @test g isa NetworkParameters{T}
    @test g.L2.S isa Sym{T}
    @test flatten(g)[1] ≈ reference
    # `gradient` projects after the rewrap; the Frobenius projection of a symmetric matrix is itself
    STORAGE_GRADIENT_CALLS[] = 0
    @test flatten(Zygote.gradient(p -> f(p, x), ps)[1])[1] ≈ reference
    @test STORAGE_GRADIENT_CALLS[] == 1
end

@testset "the storage gradient reaches the flat gradient once ($T, $name)" for T in (Float32, Float64),
    (name, f) in (("one use", sym_once), ("two uses", sym_twice))

    ps = sym_network(T)
    x = T[1, -2]
    v, l = flatten(ps)
    reference = ForwardDiff.gradient(w -> f(unflatten(l, w), x), v)
    STORAGE_GRADIENT_CALLS[] = 0
    g = Zygote.gradient(w -> f(unflatten(l, w), x), v)[1]
    @test STORAGE_GRADIENT_CALLS[] == 1
    @test g isa Vector{T}
    @test g ≈ reference
end

@testset "a loss calling `flatten(p)` twice ($T)" for T in (Float32, Float64)
    # each `flatten` rule returns a `NetworkParameters`, and Zygote adds the two with `+`
    ps = sample_network(T)
    g = Zygote.gradient(p -> sum(first(flatten(p))) + sum(abs2, first(flatten(p))), ps)[1]
    @test g isa NetworkParameters{T}
    @test flatten(g)[1] ≈ 1 .+ 2 .* flatten(ps)[1]
    qs = sym_network(T)
    v, _ = flatten(qs)
    h = Zygote.gradient(p -> sum(first(flatten(p))) + sum(abs2, first(flatten(p))), qs)[1]
    @test h.L2.S isa Sym{T}
    @test flatten(h)[1] ≈ 1 .+ 2 .* v
end

@testset "two sets add leaf by leaf, `nothing` a zero ($T)" for T in (Float32, Float64)
    a = NetworkParameters((L1 = (W = T[1 2], b = nothing), L2 = nothing,
        L3 = (S = Sym(T[1, 2, 3], 2),)))
    b = NetworkParameters((L1 = (W = T[10 20], b = T[5]), L2 = nothing,
        L3 = (S = Sym(T[10, 20, 30], 2),)))
    c = a + b
    @test c.L1.W == T[11 22]
    @test c.L1.b == T[5]
    @test c.L2 === nothing
    @test c.L3.S isa Sym{T}
    @test c.L3.S.S == T[11, 22, 33]
end

@testset "a nested set gets the storage gradient on both paths ($T)" for T in (Float32, Float64)
    ps = NetworkParameters((L1 = (W = T[0.1 0.2; 0.3 0.4], b = T[0.5, 0.6]),
        sub = NetworkParameters((L2 = (S = Sym(T[0.7, -0.8, 0.9], 2),),))))
    x = T[1, -2]
    f(p) = sum(tanh.(p.sub.L2.S * tanh.(p.L1.W * x .+ p.L1.b)))
    v, l = flatten(ps)
    reference = ForwardDiff.gradient(w -> f(unflatten(l, w)), v)
    STORAGE_GRADIENT_CALLS[] = 0
    g = Zygote.pullback(f, ps)[2](one(T))[1]
    @test STORAGE_GRADIENT_CALLS[] == 1
    @test g.sub isa NetworkParameters{T}
    @test g.sub.L2.S isa Sym{T}
    @test flatten(g)[1] ≈ reference
    @test flatten(Zygote.gradient(f, ps)[1])[1] ≈ reference
    STORAGE_GRADIENT_CALLS[] = 0
    @test Zygote.gradient(w -> f(unflatten(l, w)), v)[1] ≈ reference
    @test STORAGE_GRADIENT_CALLS[] == 1
end

# The `flatten` rule returns a `NetworkParameters` whose leaves hold ∂L/∂S already. Neither walk may
# convert such a gradient a second time: that doubles the off-diagonal entries of a `Sym`.
@testset "a gradient from the `flatten` rule is not converted again ($T)" for T in (Float32, Float64)
    ps = NetworkParameters((L1 = (S = Sym(T[1, 2, 3], 2),),))
    v, l = flatten(ps)
    f(w) = sum(abs2, first(flatten(unflatten(l, w))))
    STORAGE_GRADIENT_CALLS[] = 0
    g = Zygote.gradient(f, v)[1]
    @test STORAGE_GRADIENT_CALLS[] == 0
    @test g isa Vector{T}
    @test g ≈ ForwardDiff.gradient(f, v)
    @test g ≈ 2 .* v
end

@testset "a nested set flattened in the loss ($T)" for T in (Float32, Float64)
    # `p.L1.S` gets a natural cotangent and is converted once; `p.sub` gets the `flatten` rule's
    ps = NetworkParameters((L1 = (S = Sym(T[0.7, -0.8, 0.9], 2),),
        sub = NetworkParameters((L2 = (S = Sym(T[1, 2, 3], 2),),))))
    x = T[1, -2]
    f(p) = sum(abs2, p.L1.S * x) + sum(abs2, first(flatten(p.sub)))
    v, l = flatten(ps)
    reference = ForwardDiff.gradient(w -> f(unflatten(l, w)), v)
    STORAGE_GRADIENT_CALLS[] = 0
    g = Zygote.pullback(f, ps)[2](one(T))[1]
    @test STORAGE_GRADIENT_CALLS[] == 1
    @test g.sub isa NetworkParameters{T}
    @test g.sub.L2.S.S ≈ T[2, 4, 6]
    @test flatten(g)[1] ≈ reference
    STORAGE_GRADIENT_CALLS[] = 0
    @test flatten(Zygote.gradient(f, ps)[1])[1] ≈ reference
    @test STORAGE_GRADIENT_CALLS[] == 1
    STORAGE_GRADIENT_CALLS[] = 0
    @test Zygote.gradient(w -> f(unflatten(l, w)), v)[1] ≈ reference
    @test STORAGE_GRADIENT_CALLS[] == 1
end

@testset "a tuple branch gets the storage gradient ($T)" for T in (Float32, Float64)
    ps = NetworkParameters((t = (T[1, 2], Sym(T[1, 2, 3], 2)), u = (b = T[3],)))
    x = T[1, -2]
    f(p) = sum(p.t[1]) + sum(abs2, p.t[2] * x)
    v, l = flatten(ps)
    # by hand: S x = (-3, -4), G = 2 (S x) xᵀ = [-6 12; -8 16], ∂L/∂S = (G₁₁, G₂₁ + G₁₂, G₂₂)
    reference = ForwardDiff.gradient(w -> f(unflatten(l, w)), v)
    @test reference ≈ T[1, 1, -6, 4, 16, 0]
    STORAGE_GRADIENT_CALLS[] = 0
    g = Zygote.pullback(f, ps)[2](one(T))[1]
    @test STORAGE_GRADIENT_CALLS[] == 1
    @test g.t[1] ≈ T[1, 1]
    @test g.t[2] isa Sym{T}
    @test g.t[2].S ≈ T[-6, 4, 16]
    @test g.u === nothing
    h = Zygote.gradient(f, ps)[1]
    @test h.t[2] isa Sym{T}
    @test h.t[2].S ≈ T[-6, 4, 16]
    # a tuple branch the loss never reads is a hole
    @test Zygote.gradient(p -> sum(p.u.b), ps)[1].t === nothing
end

# A package that writes its own pullback for a function of the layers in order, `values(ps)`, gets
# Zygote's natural cotangent of that tuple and converts it with the public walk.
@testset "map_cotangent converts the cotangent of the layers in order ($T)" for T in (Float32, Float64)
    @test Base.ispublic(NeuralNetworkParameters, :map_cotangent)
    @test !Base.isexported(NeuralNetworkParameters, :map_cotangent)
    ps = sym_network(T)
    x = T[1, -2]
    f(layers) = sum(tanh.(layers[2].S * tanh.(layers[1].W * x .+ layers[1].b)))
    v, l = flatten(ps)
    reference = ForwardDiff.gradient(w -> f(values(unflatten(l, w))), v)
    Δ = Zygote.pullback(f, values(ps))[2](one(T))[1]
    STORAGE_GRADIENT_CALLS[] = 0
    g = NeuralNetworkParameters.map_cotangent(storage_gradient, values(ps), Δ)
    @test STORAGE_GRADIENT_CALLS[] == 1
    @test g[2].S isa Sym{T}
    @test flatten(NetworkParameters(NamedTuple{keys(ps)}(g)))[1] ≈ reference
    # a layer the function never reads is a hole
    Δb = Zygote.pullback(layers -> sum(layers[1].b), values(ps))[2](one(T))[1]
    ḡ = NeuralNetworkParameters.map_cotangent(storage_gradient, values(ps), Δb)
    @test ḡ[2] === nothing
end

@testset "a tuple branch and a nested set add leaf by leaf ($T)" for T in (Float32, Float64)
    a = NetworkParameters((t = (T[1, 2], Sym(T[1, 2, 3], 2)),
        sub = NetworkParameters((L1 = (W = T[1 2], b = T[4]),))))
    b = NetworkParameters((t = (T[10, 20], Sym(T[10, 20, 30], 2)),
        sub = NetworkParameters((L1 = (W = T[10 20], b = nothing),))))
    c = a + b
    @test c.t[1] == T[11, 22]
    @test c.t[2] isa Sym{T}
    @test c.t[2].S == T[11, 22, 33]
    @test c.sub isa NetworkParameters{T}
    @test c.sub.L1.W == T[11 22]
    @test c.sub.L1.b == T[4]
    # and the same through Zygote, which adds the two gradients of `flatten(p)` with `+`
    ps = NetworkParameters((t = (T[1, 2], Sym(T[1, 2, 3], 2)),
        sub = NetworkParameters((L1 = (W = T[4 5],),))))
    v, _ = flatten(ps)
    h = Zygote.gradient(p -> sum(first(flatten(p))) + sum(abs2, first(flatten(p))), ps)[1]
    @test h.t[2] isa Sym{T}
    @test h.sub isa NetworkParameters{T}
    @test flatten(h)[1] ≈ 1 .+ 2 .* v
end

@testset "the `unflatten` rule reads each shape of a set's cotangent ($T)" for T in (Float32, Float64)
    # a `Tangent` from Zygote, the set itself from the `flatten` rule, and the structural `NamedTuple`
    ps = sample_network(T)
    v, l = flatten(ps)
    _, pb = ChainRulesCore.rrule(unflatten, l, v)
    Δ = unflatten(l, collect(T, 1:length(v)))
    expected = collect(T, 1:length(v))
    @test pb(Δ)[3] == expected
    @test pb((params = params(Δ),))[3] == expected
    @test pb(ChainRulesCore.Tangent{Any}(params = ChainRulesCore.Tangent{Any}(;
        params(Δ)...)))[3] ==
          expected
end

@testset "the storage gradient is not applied to a structural tangent" begin
    # a loss reading the storage field directly gets ∂L/∂S from Zygote already, as a tangent over the
    # leaf's fields; the default method passes it through
    ps = sym_network(Float64)
    v, l = flatten(ps)
    f(p) = sum(abs2, p.L2.S.S)
    reference = ForwardDiff.gradient(w -> f(unflatten(l, w)), v)
    STORAGE_GRADIENT_CALLS[] = 0
    @test Zygote.pullback(f, ps)[2](1.0)[1].L2.S.S ≈ reference[7:9]
    @test Zygote.gradient(w -> f(unflatten(l, w)), v)[1] ≈ reference
    @test STORAGE_GRADIENT_CALLS[] == 0
end
