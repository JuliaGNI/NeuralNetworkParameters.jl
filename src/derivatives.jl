# Reverse-mode rules for the two conversions.
#
# These are what make "differentiate with respect to the flat form, get the answer back in the shape of
# the network" work under Zygote. `ForwardDiff` needs no rule: `unflatten` is generic in the element type
# of its vector, so a `Dual`-valued vector simply produces `Dual`-valued parameters.
#
# At a fixed layout the two conversions are linear and mutually adjoint, so each rule is the other
# conversion.

function ChainRulesCore.rrule(::typeof(unflatten), layout::ParameterLayout, v::AbstractVector)
    ps = unflatten(layout, v)

    function unflatten_pullback(Δ)
        Δv = zero(v)
        _accumulate_cotangent!(Δv, layout, Δ)
        ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Δv
    end

    ps, unflatten_pullback
end

function ChainRulesCore.rrule(::typeof(flatten), ::Type{T}, ps) where {T}
    v, layout = flatten(T, ps)

    function flatten_pullback(Δ)
        Δv = _flat_cotangent(ChainRulesCore.unthunk(Δ))
        Δps = Δv === nothing ? ChainRulesCore.ZeroTangent() : unflatten(layout, Δv)
        ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Δps
    end

    (v, layout), flatten_pullback
end

function ChainRulesCore.rrule(::typeof(flatten), ps)
    (v, layout), pb = ChainRulesCore.rrule(flatten, parameter_eltype(ps), ps)
    flatten_pullback(Δ) = (ChainRulesCore.NoTangent(), pb(Δ)[3])
    (v, layout), flatten_pullback
end

# `flatten` returns a tuple, so its cotangent is a tangent *for the tuple*; only the first component,
# the one for the flat vector, carries anything.
_flat_cotangent(Δ::ChainRulesCore.AbstractZero) = nothing
_flat_cotangent(Δ::Tuple) = _normalized(first(Δ))
_flat_cotangent(Δ::ChainRulesCore.Tangent) = _normalized(first(ChainRulesCore.backing(Δ)))

# ---------------------------------------------------------------------------------------------------
# Reading a cotangent into the flat vector
#
# Walk the layout and add whatever the cotangent has at each position. Positions it says nothing about
# stay zero, which is how a structural zero — a layer the loss never touched — comes out as a zero block
# rather than an error.
#
# Every cotangent is put through `_normalized` before it is dispatched on, which unthunks it and turns
# any flavour of zero into `nothing`. Each layout type then has exactly *one* method, which checks for
# `nothing` itself rather than leaving it to dispatch: a method specialising on the cotangent as well as
# the layout would be ambiguous with the per-layout ones, and a missing branch would then fail where a
# missing leaf did not.
# ---------------------------------------------------------------------------------------------------

_accumulate_cotangent!(Δv, l::ParameterLayout, Δ) = _accumulate!(Δv, l, _normalized(Δ))

_normalized(Δ::ChainRulesCore.AbstractThunk) = _normalized(ChainRulesCore.unthunk(Δ))
_normalized(::ChainRulesCore.AbstractZero) = nothing
_normalized(::Nothing) = nothing
_normalized(Δ) = Δ

function _accumulate!(Δv, l::ParametersLayout, Δ)
    Δ === nothing && return Δv
    _accumulate!(Δv, l.inner, _normalized(_unwrap_parameters(Δ)))
end

function _accumulate!(Δv, l::NestedLayout, Δ)
    Δ === nothing && return Δv
    _accumulate_named!(Δv, values(l.children), keys(l.children), _cotangent_backing(Δ))
    Δv
end

function _accumulate!(Δv, l::TupleLayout, Δ)
    Δ === nothing && return Δv
    _accumulate_positional!(Δv, l.children, _cotangent_backing(Δ))
    Δv
end

function _accumulate!(Δv, l::WrappedLayout, Δ)
    Δ === nothing && return Δv
    _accumulate!(Δv, l.inner, _normalized(_wrapped_storage(l.prototype, Δ)))
end

function _accumulate!(Δv, l::LeafLayout, Δ)
    Δ === nothing && return Δv
    _add_leaf_cotangent!(Δv, l, Δ)
    Δv
end

# The branch walks, in the `Base.tail` shape the rest of the package walks a layout in. A `for` loop
# over `keys(l.children)` reads the same, but it indexes a heterogeneous `NamedTuple` with a runtime
# `Symbol`: the child layout comes back as the union of the branch's child types, so `_accumulate!` is
# dispatched dynamically once per child on every reverse pass. The keys are a compile-time constant
# and the recursion keeps them one, which is the same reason `_unflatten_children` exists.
@inline _accumulate_named!(Δv, ::Tuple{}, ::Tuple{}, Δ) = nothing
@inline function _accumulate_named!(Δv, ls::Tuple, ks::Tuple, Δ)
    _accumulate!(Δv, first(ls), _normalized(_cotangent_get(Δ, first(ks))))
    _accumulate_named!(Δv, Base.tail(ls), Base.tail(ks), Δ)
end

# The positional walk consumes the cotangent alongside the layout rather than indexing into it, so it
# needs no index to be constant. `_cotangent_head`/`_cotangent_tail` carry the two cases a position
# can be in.
@inline _accumulate_positional!(Δv, ::Tuple{}, Δ) = nothing
@inline function _accumulate_positional!(Δv, ls::Tuple, Δ)
    _accumulate!(Δv, first(ls), _normalized(_cotangent_head(Δ)))
    _accumulate_positional!(Δv, Base.tail(ls), _cotangent_tail(Δ))
end

_add_leaf_cotangent!(Δv, l::LeafLayout, Δ::Number) = (Δv[first(l.range)] += Δ; nothing)

function _add_leaf_cotangent!(Δv, l::LeafLayout, Δ::AbstractArray)
    length(Δ) == length(l.range) || throw(DimensionMismatch(string(
        "cotangent of length ", length(Δ), " for a leaf of length ", length(l.range))))
    o = first(l.range) - 1
    for (i, x) in enumerate(Δ)
        Δv[o + i] += x
    end
    nothing
end

function _add_leaf_cotangent!(_, ::LeafLayout, Δ)
    throw(ArgumentError(string(
        "cannot read a cotangent of type `", typeof(Δ), "` as the derivative of one leaf.\n",
        "If it belongs to a structured parameter, that type's `freeparameters` has to name the storage ",
        "the reverse pass produced a tangent for.")))
end

# A cotangent for a `NetworkParameters` arrives either as the type itself, or — when the reverse pass
# built it structurally — as a tangent whose single field is the wrapped `NamedTuple`.
_unwrap_parameters(Δ::NetworkParameters) = params(Δ)
_unwrap_parameters(Δ::NamedTuple{(:params,)}) = Δ.params
_unwrap_parameters(Δ) = _unwrap_parameters_backing(_cotangent_backing(Δ))

_unwrap_parameters_backing(nt::NamedTuple{(:params,)}) = nt.params
_unwrap_parameters_backing(nt) = nt

_cotangent_backing(Δ::ChainRulesCore.Tangent) = ChainRulesCore.backing(Δ)
_cotangent_backing(Δ::NetworkParameters) = params(Δ)
_cotangent_backing(Δ) = Δ

# Anything the cotangent is silent about becomes `nothing`, i.e. a structural zero.
_cotangent_get(nt::NamedTuple, k::Symbol) = haskey(nt, k) ? nt[k] : nothing
_cotangent_get(x, _) = x

# The same rule, read one position at a time: a `Tuple` cotangent shorter than the layout runs out
# into structural zeros, and a cotangent that is not a tuple at all stands in for every position of
# the branch — which is what a `Tangent` for a leaf, handed to a branch, means.
@inline _cotangent_head(::Tuple{}) = nothing
@inline _cotangent_tail(::Tuple{}) = ()
@inline _cotangent_head(Δ::Tuple) = first(Δ)
@inline _cotangent_tail(Δ::Tuple) = Base.tail(Δ)
@inline _cotangent_head(Δ) = Δ
@inline _cotangent_tail(Δ) = Δ

# The cotangent of a structured leaf: the same structured type if the reverse pass kept it, otherwise
# whatever tangent stood in for it, reduced to its storage.
_wrapped_storage(::P, Δ::P) where {P} = freeparameters(Δ)
_wrapped_storage(prototype, Δ) = _storage_from_backing(prototype, _cotangent_backing(Δ))

function _storage_from_backing(prototype, nt::NamedTuple)
    s = freeparameters(prototype)
    s isa Union{NamedTuple, Tuple} && return _matching_storage(s, nt)
    # Single-block storage against a tangent over the struct's fields: `(S = ..., n = ZeroTangent())`
    # for a leaf whose storage is its `S`. Which field that is follows from the prototype — the object
    # `freeparameters` returned is one of its fields — so the cotangent is narrowed to that component.
    f = _storage_field(prototype, s)
    f === nothing ? nt : _cotangent_get(nt, f)
end

_storage_from_backing(_, Δ) = Δ

function _storage_field(prototype, storage)
    for f in fieldnames(typeof(prototype))
        getfield(prototype, f) === storage && return f
    end
    nothing
end

# `freeparameters` of the prototype tells us which of the tangent's fields are the storage. Both cases
# recurse rather than closing over `nt`, for the reason `_unflatten_children` does — and the positional
# one used to be an `ntuple` over a *runtime* length, which stops being inferable past ten blocks.
_matching_storage(storage::NamedTuple, nt::NamedTuple) =
    NamedTuple{keys(storage)}(_matching_named(keys(storage), nt))
_matching_storage(storage::Tuple, nt::NamedTuple) = _matching_positional(storage, values(nt))
_matching_storage(_, nt::NamedTuple) = nt

@inline _matching_named(::Tuple{}, _) = ()
@inline _matching_named(ks::Tuple, nt) =
    (_cotangent_get(nt, first(ks)), _matching_named(Base.tail(ks), nt)...)

@inline _matching_positional(::Tuple{}, _) = ()
@inline _matching_positional(storage::Tuple, Δ) =
    (_cotangent_head(Δ), _matching_positional(Base.tail(storage), _cotangent_tail(Δ))...)
