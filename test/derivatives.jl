using NeuralNetworkParameters
using ForwardDiff
using Zygote
using ZygoteRules
using ChainRulesCore
using Test

include("helpers/wrapper_types.jl")

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

@testset "an untouched leaf gets a zero leaf ($T)" for T in (Float32, Float64)
    # the `ZygoteRules` extension rewraps the pullback's `NamedTuple`, in which an untouched layer is
    # `nothing`; the gradient holds a zero leaf there instead
    two = NetworkParameters((a = T[1, 2], b = T[3, 4]))
    g = Zygote.gradient(p -> sum(p.a), two)[1]
    @test g isa NetworkParameters{T}
    @test g.b == zeros(T, 2)
    @test g.b isa Vector{T}
    @test g.a == ones(T, 2)
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
    # a set that *was* touched still comes back wrapped, with a zero leaf where it was not read
    @test Zygote.gradient(p -> sum(p.L1.W), ps)[1] isa NetworkParameters
end

# A rule may hand the set back its structural tangent with a zero inside, `(params = nothing,)`. That
# is a set the reverse pass never touched too.
untouching(p) = 1.0
ZygoteRules.@adjoint function untouching(p::NetworkParameters)
    untouching(p), _ -> ((params = nothing,),)
end

# A rule that gives the set a zero, where Zygote's own pullback then gives `nothing` for the whole
# tuple of arguments.
zeroing(p) = 1.0
function ChainRulesCore.rrule(::typeof(zeroing), p::NetworkParameters)
    zeroing(p), _ -> (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent())
end

@testset "a structural tangent holding a zero comes back as `nothing`" begin
    @test Zygote.pullback(untouching, ps)[2](1.0) === (nothing,)
    @test Zygote.gradient(untouching, ps) === (nothing,)
    @test Zygote.pullback(zeroing, ps)[2](1.0) === (nothing,)
    @test Zygote.gradient(p -> untouching(p) + sum(p.L1.W), ps)[1].L1.W == ones(2, 2)
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

@testset "a structural zero contributes no element type" begin
    @test NeuralNetworkParameters.parameter_eltype(ChainRulesCore.ZeroTangent()) === Union{}
    @test NeuralNetworkParameters.parameter_eltype((
        a = [1.0f0], b = ChainRulesCore.ZeroTangent())) ===
          Float32
end

# The pullback walks a branch at keys known at compile time. A walk that indexed a heterogeneous
# `NamedTuple` with a runtime `Symbol` would get the union of the branch's child types back, and
# dispatch dynamically once per child: its cost would grow with the number of layers, and none of the
# value tests above could see it.
_pullback_allocs(pb, Δ) = @allocated pb(Δ)

@testset "the pullback does not pay for the depth of the tree" begin
    # the same four leaves, once flat and once one to a layer. The walks are at keys known at
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

@testset "an untouched position of a tuple branch is a zero block" begin
    tp = NetworkParameters((t = ([1.0, 2.0], [3.0], [4.0]),))
    tv, tl = flatten(tp)
    @test Zygote.gradient(w -> sum(unflatten(tl, w).t[2]), tv)[1] == [0.0, 0.0, 1.0, 0.0]
end

# A leaf that keeps its numbers in twelve blocks, each a field of its own, and its storage gradient
# from a tangent over those fields.
struct Twelve{T} <: AbstractVector{T}
    b1::Vector{T}
    b2::Vector{T}
    b3::Vector{T}
    b4::Vector{T}
    b5::Vector{T}
    b6::Vector{T}
    b7::Vector{T}
    b8::Vector{T}
    b9::Vector{T}
    b10::Vector{T}
    b11::Vector{T}
    b12::Vector{T}
end
Base.size(::Twelve) = (12,)
Base.getindex(x::Twelve, i::Int) = getfield(x, i)[1]
function NNP.freeparameters(x::Twelve)
    (x.b1, x.b2, x.b3, x.b4, x.b5, x.b6, x.b7, x.b8, x.b9, x.b10, x.b11, x.b12)
end
NNP.rebuild(::Twelve, data) = Twelve(data...)

@testset "storage in more than ten blocks is matched by field ($T)" for T in (Float32, Float64)
    x = Twelve(ntuple(i -> T[i], 12)...)
    # a tangent over the fields that leaves the last nine out: those blocks are zeros
    Δ = (b1 = T[10], b2 = T[20], b3 = T[30])
    g = @inferred storage_gradient(x, Δ)
    @test g isa Twelve{T}
    @test collect(g) == T[10, 20, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    full = NamedTuple{ntuple(i -> Symbol(:b, i), 12)}(ntuple(i -> T[i + 1], 12))
    @test collect(storage_gradient(x, ChainRulesCore.Tangent{typeof(x)}(; full...))) ==
          T.(2:13)
end

@testset "a tangent that leaves out a leading block is matched by field ($T)" for T in (
    Float32, Float64)
    # a `Tangent` need not hold every field, nor hold them in order: each block takes the component
    # of its own field, and a block the tangent leaves out is a zero block
    x = Lift(Sym(T[0.7, -0.8, 0.9], 2), T[0.1 -0.2; 0.3 0.4], 4)
    B = T[1 2; 3 4]
    g = storage_gradient(x, ChainRulesCore.Tangent{typeof(x)}(; B = B))
    @test g isa typeof(x)
    @test g.A.S == zeros(T, 3)
    @test g.B == B
    g = storage_gradient(x,
        ChainRulesCore.Tangent{typeof(x)}(; B = B, A = (S = T[1, 2, 3], n = nothing)))
    @test g isa typeof(x)
    @test g.A.S == T[1, 2, 3]
    @test g.B == B
end

# A leaf that keeps its numbers in two number blocks, one per field, and a leaf whose one number block
# is its first field, beside a second field of the same type.
struct NumPair{T} <: AbstractVector{T}
    x::T
    y::T
end
Base.size(::NumPair) = (2,)
Base.getindex(p::NumPair, i::Int) = i == 1 ? p.x : p.y
NNP.freeparameters(p::NumPair) = (p.x, p.y)
NNP.rebuild(::NumPair, data) = NumPair(data...)

struct NumFirst{T} <: AbstractVector{T}
    x::T
    y::T
end
Base.size(::NumFirst) = (1,)
Base.getindex(p::NumFirst, ::Int) = p.x
NNP.freeparameters(p::NumFirst) = p.x
NNP.rebuild(p::NumFirst, data) = NumFirst(data, p.y)

@testset "block i of a tuple storage is field i, whatever its value ($T)" for T in (
    Float32, Float64)
    # two equal number blocks each take the component of their own field
    p = NumPair(one(T), one(T))
    g = storage_gradient(p, ChainRulesCore.Tangent{typeof(p)}(; x = T(2), y = T(3)))
    @test g isa NumPair{T}
    @test (g.x, g.y) == (T(2), T(3))
    ps = NetworkParameters((L = p,))
    gz = Zygote.gradient(q -> q.L.x^2 + 3 * q.L.y, ps)[1]
    @test gz.L isa NumPair{T}
    @test (gz.L.x, gz.L.y) == (T(2), T(3))
    v, l = flatten(ps)
    @test Zygote.gradient(w -> (q = unflatten(l, w); q.L.x^2 + 3 * q.L.y), v)[1] == T[2, 3]
end

@testset "a storage that is two fields of the leaf raises ($T)" for T in (Float32, Float64)
    # one block of storage is one field; equal to two, it is no field in particular
    Δ = (x = T(2), y = T(3))
    @test_throws ArgumentError storage_gradient(NumFirst(one(T), one(T)), Δ)
    g = storage_gradient(NumFirst(one(T), T(5)), Δ)
    @test g isa NumFirst{T}
    @test (g.x, g.y) == (T(2), T(5))
end

# A leaf whose tuple storage holds its fields in the other order, and one whose tuple storage has
# more blocks than the leaf has fields.
struct Swap{T} <: AbstractVector{T}
    x::Vector{T}
    y::Vector{T}
end
Base.size(s::Swap) = (length(s.x) + length(s.y),)
Base.getindex(s::Swap, i::Int) = i ≤ length(s.y) ? s.y[i] : s.x[i - length(s.y)]
NNP.freeparameters(s::Swap) = (s.y, s.x)
NNP.rebuild(::Swap, data) = Swap(data[2], data[1])

struct Thrice{T} <: AbstractVector{T}
    x::Vector{T}
    y::Vector{T}
end
Base.size(s::Thrice) = (2 * length(s.x) + length(s.y),)
Base.getindex(s::Thrice, i::Int) = vcat(s.x, s.y, s.x)[i]
NNP.freeparameters(s::Thrice) = (s.x, s.y, s.x)
NNP.rebuild(::Thrice, data) = Thrice(data[1], data[2])

@testset "a tuple storage that is not the leaf's fields in order raises ($T)" for T in (
    Float32, Float64)
    Δ = (x = T[1, 1], y = T[2, 2])
    @test_throws ArgumentError storage_gradient(Swap(T[1, 2], T[3, 4]), Δ)
    @test_throws ArgumentError storage_gradient(Thrice(T[1, 2], T[3, 4]), Δ)
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
    @test g.L2.b == zeros(T, 1)
    # and directly, where a hole becomes a zero leaf and a structural tangent is left alone
    Δ = NetworkParameters((L1 = (Y = ones(T, 2, 2),), L2 = nothing))
    @test ChainRulesCore.ProjectTo(ps)(Δ) ==
          NetworkParameters((L1 = (Y = Frob(ones(T, 2, 2)),),
        L2 = (b = zeros(T, 1),)))
    Δs = NetworkParameters((L1 = (Y = (A = ones(T, 2, 2),),), L2 = (b = nothing,)))
    @test ChainRulesCore.ProjectTo(ps)(Δs) ==
          NetworkParameters((L1 = (Y = (A = ones(T, 2, 2),),), L2 = (b = zeros(T, 1),)))
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

@testset "two sets add leaf by leaf ($T)" for T in (Float32, Float64)
    a = NetworkParameters((L1 = (W = T[1 2], b = T[0]), L3 = (S = Sym(T[1, 2, 3], 2),)))
    b = NetworkParameters((L1 = (W = T[10 20], b = T[5]),
        L3 = (S = Sym(T[10, 20, 30], 2),)))
    c = a + b
    @test c isa NetworkParameters{T}
    @test c.L1.W == T[11 22]
    @test c.L1.b == T[5]
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
    @test g.u.b == zeros(T, 1)
    h = Zygote.gradient(f, ps)[1]
    @test h.t[2] isa Sym{T}
    @test h.t[2].S ≈ T[-6, 4, 16]
    # a tuple branch the loss never reads is a branch of zero leaves
    t = Zygote.gradient(p -> sum(p.u.b), ps)[1].t
    @test t[1] == zeros(T, 2)
    @test t[2] isa Sym{T}
    @test t[2].S == zeros(T, 3)
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
    # a layer the function never reads is a layer of zero leaves
    Δb = Zygote.pullback(layers -> sum(layers[1].b), values(ps))[2](one(T))[1]
    ḡ = NeuralNetworkParameters.map_cotangent(storage_gradient, values(ps), Δb)
    @test ḡ[2].S isa Sym{T}
    @test ḡ[2].S.S == zeros(T, 3)
end

# A cotangent of another tree is an error at the first branch it does not fit, not a zero or a
# non-gradient at every leaf.
@testset "map_cotangent rejects a cotangent of another shape ($T)" for T in (Float32, Float64)
    map_cotangent = NeuralNetworkParameters.map_cotangent
    ps = sym_network(T)
    x = T[1, -2]
    f(layers) = sum(tanh.(layers[2].S * tanh.(layers[1].W * x .+ layers[1].b)))
    Δ = Zygote.pullback(f, values(ps))[2](one(T))[1]
    g = Zygote.gradient(p -> f(values(p)), ps)[1]
    keyed = NamedTuple{keys(ps)}(Δ)
    # a set's gradient, or the layers in order, against the layers by name
    @test_throws ArgumentError map_cotangent(storage_gradient, params(ps), g)
    @test_throws ArgumentError map_cotangent(storage_gradient, params(ps), Δ)
    # the wrapper's structural tangent against the layers by name: a key the branch does not have
    @test_throws ArgumentError map_cotangent(storage_gradient, params(ps), (params = keyed,))
    # the layers by name against the layers in order, and too few layers
    @test_throws ArgumentError map_cotangent(storage_gradient, values(ps), keyed)
    @test_throws ArgumentError map_cotangent(storage_gradient, values(ps), (Δ[1],))
    # the layers in order as a `Tangent` read as the bare `Tuple` does, and too few of them raise
    tt = ChainRulesCore.Tangent{typeof(values(ps))}(Δ...)
    @test map_cotangent(storage_gradient, values(ps), tt)[2].S.S ==
          map_cotangent(storage_gradient, values(ps), Δ)[2].S.S
    @test_throws ArgumentError map_cotangent(storage_gradient, values(ps),
        ChainRulesCore.Tangent{typeof(values(ps))}(Δ[1]))
    # the layers by name against the set itself, which takes its structural tangent `(params = …,)`
    @test_throws ArgumentError map_cotangent(storage_gradient, ps, keyed)
    # an array where a layer is
    @test_throws ArgumentError map_cotangent(storage_gradient, values(ps), (Δ[1], Δ[2].S))
    # a named cotangent silent about a key is a hole there, as a `Tangent` with a field left out is,
    # and the hole is a zero leaf
    @test map_cotangent(storage_gradient, params(ps), (L2 = Δ[2],)).L1 ==
          (W = zeros(T, 2, 2), b = zeros(T, 2))
    t = ChainRulesCore.Tangent{typeof(params(ps))}(; L2 = Δ[2])
    @test map_cotangent(storage_gradient, params(ps), t).L2.S isa Sym{T}
    # the three shapes a set's cotangent comes in
    @test map_cotangent(storage_gradient, ps, (params = keyed,)) isa NetworkParameters{T}
    @test map_cotangent(storage_gradient, ps, g) === g
    tp = ChainRulesCore.Tangent{typeof(ps)}(; params = keyed)
    @test map_cotangent(storage_gradient, ps, tp) isa NetworkParameters{T}
    # a `Tangent` with no fields is a zero, a hole wherever it stands: a set, a branch, a leaf; and so
    # is the structural tangent of a set whose `params` is a zero. The whole argument a hole is
    # `nothing`, and a hole inside it a zero leaf.
    empty_tangent(x) = ChainRulesCore.Tangent{typeof(x)}()
    @test map_cotangent(storage_gradient, ps, empty_tangent(ps)) === nothing
    @test map_cotangent(storage_gradient, params(ps), empty_tangent(params(ps))) === nothing
    @test map_cotangent(storage_gradient, values(ps), empty_tangent(values(ps))) === nothing
    z = map_cotangent(
        storage_gradient, values(ps), (Δ[1], (S = empty_tangent(ps.L2.S),)))[2].S
    @test z isa Sym{T}
    @test z.S == zeros(T, 3)
    @test map_cotangent(storage_gradient, ps, (params = nothing,)) === nothing
    tz = ChainRulesCore.Tangent{typeof(ps)}(; params = ChainRulesCore.ZeroTangent())
    @test map_cotangent(storage_gradient, ps, tz) === nothing
end

# The reverse rule of `unflatten` reads a set's cotangent with the same rule: its three shapes, a
# `Tangent` with no fields as a zero, and anything else as an error.
@testset "the unflatten rule reads a set's cotangent by the same rule ($T)" for T in (Float32, Float64)
    ps = sym_network(T)
    v, l = flatten(ps)
    _, pb = ChainRulesCore.rrule(unflatten, l, v)
    g = unflatten(l, v)
    keyed = NamedTuple{keys(ps)}(values(g))
    @test pb(g)[3] == v
    tp = ChainRulesCore.Tangent{typeof(ps)}(; params = keyed)
    @test pb((params = keyed,))[3] == pb(tp)[3]
    @test pb(ChainRulesCore.Tangent{typeof(ps)}())[3] == zero(v)
    @test_throws ArgumentError pb(keyed)
    @test_throws ArgumentError pb(values(g))
    # a `Tangent` with no fields at a leaf is a zero block there, as `nothing` is
    empty_tangent(x) = ChainRulesCore.Tangent{typeof(x)}()
    b̄ = ps.L1.b
    @test pb((params = (L1 = (W = empty_tangent(ps.L1.W), b = b̄), L2 = nothing),))[3] ==
          pb((params = (L1 = (W = nothing, b = b̄), L2 = nothing),))[3] ==
          [zero(vec(ps.L1.W)); b̄; zero(ps.L2.S.S)]
end

# The reverse rule of `flatten` reads the cotangent of the pair `(v, layout)` by the same rule: a
# `Tangent` with no fields, for the pair or for the vector, is no derivative.
@testset "the flatten rule reads an empty `Tangent` as a zero ($T)" for T in (Float32, Float64)
    ps = sym_network(T)
    (v, l), fpb = ChainRulesCore.rrule(flatten, ps)
    empty_tangent(x) = ChainRulesCore.Tangent{typeof(x)}()
    @test fpb(empty_tangent((v, l)))[2] == ChainRulesCore.ZeroTangent()
    @test fpb((empty_tangent(v), nothing))[2] == ChainRulesCore.ZeroTangent()
    @test flatten(fpb((v, nothing))[2])[1] == v
end

@testset "a tuple branch and a nested set add leaf by leaf ($T)" for T in (Float32, Float64)
    a = NetworkParameters((t = (T[1, 2], Sym(T[1, 2, 3], 2)),
        sub = NetworkParameters((L1 = (W = T[1 2], b = T[4]),))))
    b = NetworkParameters((t = (T[10, 20], Sym(T[10, 20, 30], 2)),
        sub = NetworkParameters((L1 = (W = T[10 20], b = T[0]),))))
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

@testset "a structural tangent becomes a leaf of the leaf's type ($T)" for T in (Float32, Float64)
    # a loss reading the storage field directly gets ∂L/∂S from Zygote already, as a tangent over the
    # leaf's fields; the default method rebuilds the leaf around it, and `Sym`'s own method, which
    # converts an array cotangent, is not called
    ps = sym_network(T)
    v, l = flatten(ps)
    f(p) = sum(abs2, p.L2.S.S)
    reference = ForwardDiff.gradient(w -> f(unflatten(l, w)), v)
    STORAGE_GRADIENT_CALLS[] = 0
    g = Zygote.pullback(f, ps)[2](one(T))[1]
    @test g.L2.S isa Sym{T}
    @test g.L2.S.S ≈ reference[7:9]
    @test Zygote.gradient(w -> f(unflatten(l, w)), v)[1] ≈ reference
    @test STORAGE_GRADIENT_CALLS[] == 0
end

# `Lift`'s storage gradient converts its `Sym` block with `Sym`'s, as GeometricOptimizers' lift does
# with its `SkewSymMatrix` block: the `A` block of a natural cotangent `G` is the top-left corner of
# `G`, and `∂L/∂B` is the `B` block less the transpose of the `-Bᵀ` block.
function NNP.storage_gradient(g::Lift, G::AbstractMatrix)
    n = g.A.n
    B = G[(n + 1):end, 1:n] .- transpose(G[1:n, (n + 1):end])
    Lift(NNP.storage_gradient(g.A, G[1:n, 1:n]), Matrix(B), g.N)
end

# The loss is a polynomial of degree two in the storage, so the central difference has no truncation
# error and only the rounding of the two loss values, about `eps · |L| / h`.
function central_difference(loss, layout, v::AbstractVector{Float64})
    h = cbrt(eps(Float64))
    map(eachindex(v)) do i
        vp = copy(v)
        vm = copy(v)
        vp[i] += h
        vm[i] -= h
        (loss(unflatten(layout, vp)) - loss(unflatten(layout, vm))) / 2h
    end
end

# `Float64` against the central difference at `rtol = 1e-6`, far above its rounding and far below the
# factor of 2 a second conversion of a block puts in. `Float32` against the `Float64` central
# difference of the same numbers at `sqrt(eps(Float32))`, the rounding of a `Float32` forward pass.
tolerance(::Type{Float64}) = 1e-6
tolerance(::Type{Float32}) = sqrt(eps(Float32))

function leaf_of(kind)
    kind === :dense && return [0.1 -0.2; 0.3 0.4]
    kind === :sym && return Sym([0.7, -0.8, 0.9], 2)
    kind === :manifold && return Manifold([0.6 -0.8; 0.8 0.6])
    kind === :lift && return sample_lift()
end

# The storage a loss reads directly, by field.
storage_read(X::Matrix) = sum(abs2, X)
storage_read(X::Sym) = sum(abs2, X.S)
storage_read(X::Manifold) = sum(abs2, X.A)
storage_read(X::Lift) = sum(abs2, X.A.S) + 2sum(abs2, X.B)

# A weight that is not symmetric, so that the two places a stored number appears in the dense
# interface are weighted differently. It is a constant of the loss.
function weight(X)
    T = eltype(X)
    reshape(collect(T, 1:length(X)), size(X)) ./ T(length(X))
end

use_once(p) = sum(abs2, Matrix(p.L.X) .* Zygote.@ignore(weight(p.L.X)))
use_twice(p) = use_once(p) + sum(Matrix(p.L.X) .* Zygote.@ignore(weight(p.L.X))')
read_storage(p) = storage_read(p.L.X)

@testset "the flat and the structured gradient are one gradient ($kind, $use, $T)" for kind in (
        :dense, :sym, :manifold, :lift),
    (use, loss) in (("once", use_once), ("twice", use_twice), ("storage", read_storage)),
    T in (Float32, Float64)
    ps64 = NetworkParameters((L = (X = leaf_of(kind),),))
    v64, layout64 = flatten(ps64)
    reference = central_difference(loss, layout64, v64)
    ps = mapstorage(x -> T.(x), ps64)
    v, layout = flatten(ps)
    flat = Zygote.gradient(w -> loss(unflatten(layout, w)), v)[1]
    g = Zygote.pullback(loss, ps)[2](one(T))[1]
    @test flat isa Vector{T}
    @test g isa NetworkParameters{T}
    # a structured leaf with a storage gradient of its own, or one whose storage the loss read, has a
    # gradient of its own type
    if kind in (:sym, :lift) || (kind === :manifold && use == "storage")
        @test g.L.X isa typeof(ps.L.X)
    end
    @test flat == first(flatten(g))
    @test first(flatten(Zygote.gradient(loss, ps)[1])) == flat
    @test isapprox(flat, reference; rtol = tolerance(T))
end

@testset "a gradient has no holes ($T)" for T in (Float32, Float64)
    ps = NetworkParameters((L1 = (W = T[1 2; 3 4], b = T[5, 6]),
        L2 = (S = Sym(T[1, 2, 3], 2), G = Lift(Sym(T[4, 5, 6], 2), T[7 8; 9 10], 4)),
        sub = NetworkParameters((L3 = (W = T[11 12],),))))
    loss(p) = sum(abs2, p.L1.W)
    v, l = flatten(ps)
    expected = [T[2, 6, 4, 8]; zeros(T, length(v) - 4)]
    for g in (Zygote.pullback(loss, ps)[2](one(T))[1], Zygote.gradient(loss, ps)[1])
        @test g isa NetworkParameters{T}
        @test g.L1.b == zeros(T, 2)
        @test g.L2.S isa Sym{T}
        @test g.L2.S.S == zeros(T, 3)
        @test g.L2.G isa typeof(ps.L2.G)
        @test g.L2.G.A.S == zeros(T, 3)
        @test g.L2.G.B == zeros(T, 2, 2)
        @test g.sub isa NetworkParameters{T}
        @test g.sub.L3.W == zeros(T, 1, 2)
        @test first(flatten(g)) == expected
    end
    @test Zygote.gradient(w -> loss(unflatten(l, w)), v)[1] == expected
end

@testset "a leaf with no numbers stays `nothing` in a gradient ($T)" for T in (Float32, Float64)
    # a leaf whose `parameter_eltype` is `Union{}` has no zero leaf; every numeric leaf still gets one,
    # at a leaf hole and inside a branch hole
    ps = NetworkParameters((
        a = T[1, 2], q = nothing, f = sin, L = (q = nothing, W = T[1 2; 3 4])))
    loss(p) = sum(abs2, p.a)
    for g in (Zygote.pullback(loss, ps)[2](one(T))[1], Zygote.gradient(loss, ps)[1])
        @test g isa NetworkParameters
        @test g.a == 2 .* ps.a
        @test g.a isa Vector{T}
        @test g.q === nothing
        @test g.f === nothing
        @test g.L.q === nothing
        @test g.L.W == zeros(T, 2, 2)
        @test g.L.W isa Matrix{T}
    end
end

@testset "two gradients add with `nothing` at a leaf with no numbers ($T)" for T in (
    Float32, Float64)
    fs = NetworkParameters((a = T[1, 2], f = sin))
    g = Zygote.gradient(p -> sum(p.a), fs)[1]
    h = g + g
    @test h isa NetworkParameters
    @test h.a == 2 .* g.a
    @test h.a isa Vector{T}
    @test h.f === nothing
end

@testset "a cotangent of another precision gives a gradient of the leaf's ($T)" for T in (
    Float32, Float64)
    # the gradient of a leaf has the element type of the leaf, whatever the precision of its cotangent
    S = T === Float32 ? Float64 : Float32
    ps = NetworkParameters((a = T[1, 2], b = T[3, 4]))
    v, l = flatten(ps)
    both(p) = sum(abs2, p.a) + sum(abs2, one(S) .* p.b)
    flat = Zygote.gradient(w -> both(unflatten(l, w)), v)[1]
    @test flat isa Vector{T}
    @test flat ≈ T[2, 4, 6, 8]
    for g in (Zygote.pullback(both, ps)[2](one(T))[1], Zygote.gradient(both, ps)[1])
        @test g isa NetworkParameters{T}
        @test g.b isa Vector{T}
        @test first(flatten(g)) == flat
    end
    one_leaf(p) = sum(abs2, one(S) .* p.a)
    for g in (Zygote.pullback(one_leaf, ps)[2](one(T))[1], Zygote.gradient(one_leaf, ps)[1])
        @test g isa NetworkParameters{T}
        @test g.b == zeros(T, 2)
        @test g.b isa Vector{T}
    end
    flat = Zygote.gradient(w -> one_leaf(unflatten(l, w)), v)[1]
    @test flat isa Vector{T}
    @test flat ≈ T[2, 4, 0, 0]
end

@testset "a number leaf and a direct cotangent of another precision ($T)" for T in (
    Float32, Float64)
    # the number arm of the conversion, through Zygote, and both directions of precision through a
    # cotangent passed to `map_cotangent` and to the reverse rule of `unflatten` directly
    S = T === Float32 ? Float64 : Float32
    ps = NetworkParameters((s = one(T), a = T[1, 2], b = T[3, 4]))
    v, l = flatten(ps)
    scaled(p) = p.s * one(S) * sum(abs2, one(S) .* p.a)
    g = Zygote.gradient(scaled, ps)[1]
    @test g isa NetworkParameters{T}
    @test g.s isa T
    @test g.s == 5
    flat = Zygote.gradient(w -> scaled(unflatten(l, w)), v)[1]
    @test flat isa Vector{T}
    @test flat ≈ T[5, 2, 4, 0, 0]
    Δ = (params = (s = S(5), a = S[2, 4], b = nothing),)
    g = NeuralNetworkParameters.map_cotangent(storage_gradient, ps, Δ)
    @test g isa NetworkParameters{T}
    @test g.s isa T
    @test g.s == 5
    @test g.a isa Vector{T}
    @test g.a == T[2, 4]
    @test g.b == zeros(T, 2)
    _, pb = ChainRulesCore.rrule(unflatten, l, v)
    @test pb(Δ)[3] isa Vector{T}
    @test pb(Δ)[3] == T[5, 2, 4, 0, 0]
    # an integer cotangent is a precision too
    g = NeuralNetworkParameters.map_cotangent(storage_gradient,
        NetworkParameters((s = one(T), a = T[1, 2])), (params = (s = 3, a = [1, 2]),))
    @test g isa NetworkParameters{T}
    @test g.s isa T
    @test g.s == 3
    @test g.a isa Vector{T}
    @test g.a == T[1, 2]
end

@testset "a cotangent of another number type passes through ($T)" for T in (Float32, Float64)
    # forward over reverse with respect to a scale that is no parameter: the cotangent of the leaf is
    # a `Dual`, which is no precision to convert; d/ds of 2 s² a is 4 s a
    ps = NetworkParameters((a = T[1, 2],))
    d = ForwardDiff.derivative(s -> Zygote.gradient(p -> sum(abs2, s .* p.a), ps)[1].a, one(T))
    @test d ≈ 4 .* ps.a
end

# A leaf whose storage is a struct without the protocol, so it has no element type.
struct OpaqueBox{T}
    x::Vector{T}
end
struct Hidden{T} <: AbstractVector{T}
    o::OpaqueBox{T}
end
Base.size(::Hidden) = (2,)
Base.getindex(h::Hidden, i::Int) = h.o.x[i]
NNP.freeparameters(h::Hidden) = h.o

@testset "a leaf with no element type keeps its cotangent ($T)" for T in (Float32, Float64)
    # there is no element type to convert to, so the default `storage_gradient` returns the cotangent
    h = Hidden(OpaqueBox(T[1, 2]))
    @test NNP.parameter_eltype(h) === Union{}
    Δ = [1.0, 2.0]
    @test storage_gradient(h, Δ) === Δ
    @test storage_gradient(h, 3.0) === 3.0
end

# A function barrier with concrete argument types, which calls `f` once before it measures.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "the flat rule allocates one vector ($T)" for T in (Float32, Float64)
    # the flat gradient is the one vector the rule has to allocate: the gradient set of dense leaves
    # holds the cotangent's own arrays. The flat vector holds at least 1 MiB, so a second one is three
    # orders of magnitude above the 1 KiB that the two readings may differ by (Julia 1.11 adds 32 B).
    ps = NetworkParameters((L1 = (W = T.(reshape(1:(512 * 512), 512, 512)), b = T.(1:512)),))
    v, l = flatten(ps)
    @test sizeof(v) ≥ 2^20
    _, pb = ChainRulesCore.rrule(unflatten, l, v)
    Δ = (params = params(unflatten(l, v)),)
    @test pb(Δ)[3] == v
    @test allocations(similar, v) > 0
    @test abs(allocations(pb, Δ) - allocations(similar, v)) ≤ 1024
end
