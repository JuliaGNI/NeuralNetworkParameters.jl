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
#
# `nothing` is a structural zero here as it is everywhere else in this file — the convention
# `_normalized` states. Zygote converts one to a `ZeroTangent` before it reaches an `rrule`, so this
# method is for a caller that invokes the rule directly; without it the two spellings of "no
# derivative" disagree, and only the one Zygote does not use raises.
_flat_cotangent(::Nothing) = nothing
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
    _accumulate_named!(Δv, l.children, _cotangent_backing(Δ))
    Δv
end

function _accumulate!(Δv, l::TupleLayout, Δ)
    Δ === nothing && return Δv
    _accumulate_positional!(Δv, l.children, _cotangent_backing(Δ))
    Δv
end

function _accumulate!(Δv, l::WrappedLayout, Δ)
    Δ === nothing && return Δv
    Δs = storage_gradient(l.prototype, Δ)
    _accumulate!(Δv, l.inner, _normalized(_wrapped_storage(l.prototype, Δs)))
end

function _accumulate!(Δv, l::LeafLayout, Δ)
    Δ === nothing && return Δv
    _add_leaf_cotangent!(Δv, l, Δ)
    Δv
end

# The named branch walk, written out for the reason the head of `walk.jl` gives. A `for` loop over
# `keys(l.children)` reads the same, but it indexes a heterogeneous `NamedTuple` with a runtime
# `Symbol`: the child layout comes back as the union of the branch's child types, so `_accumulate!`
# would be dispatched dynamically once per child on every reverse pass. Splicing the keys in as
# literals keeps every one of them constant, and keeps the branch to one specialisation rather than one
# per child.
#
# This is the reverse pass of a branch that can be as wide as a network has layers, so it is the one
# walk here that had to be rewritten. The two positional walks below are not: their length is the
# number of *blocks of a single leaf* — two for a `StiefelLieAlgHorMatrix`, one for a Grassmann lift —
# so they stay the chain they read best as.
@generated function _accumulate_named!(Δv, children::NamedTuple{Keys}, Δ) where {Keys}
    calls = [:(_accumulate!(Δv, getfield(children, $i),
                 _normalized(_cotangent_get(Δ, $(QuoteNode(Keys[i]))))))
             for i in 1:length(Keys)]
    quote
        $(calls...)
        nothing
    end
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

# Broadcast over the leaf's stretch of the flat vector and not a loop over its elements, for the reason
# `flatten` copies with `copyto!`: an element of a leaf is never indexed individually, so the reverse
# pass reaches a GPU array on the same terms the forward one does.
function _add_leaf_cotangent!(Δv, l::LeafLayout, Δ::AbstractArray)
    length(Δ) == length(l.range) || throw(DimensionMismatch(string(
        "cotangent of length ", length(Δ), " for a leaf of length ", length(l.range))))
    @views Δv[l.range] .+= vec(Δ)
    nothing
end

function _add_leaf_cotangent!(_, ::LeafLayout, Δ)
    throw(ArgumentError(string(
        "cannot read a cotangent of type `", typeof(Δ), "` as the derivative of one leaf.\n",
        "If it belongs to a structured parameter, that type's `freeparameters` has to name the storage ",
        "the reverse pass produced a tangent for.")))
end

# A cotangent for a `NetworkParameters` arrives as the structural tangent of the wrapper, whose single
# field is the wrapped `NamedTuple` — a `Tangent` in a rule, a `NamedTuple` in Zygote's own reverse
# pass over a set nested in a set — or as the type itself, which is what the `flatten` rule returns.
_unwrap_parameters(Δ::NetworkParameters) = params(Δ)
_unwrap_parameters(Δ::NamedTuple{(:params,)}) = Δ.params
_unwrap_parameters(Δ::ChainRulesCore.Tangent) = ChainRulesCore.backing(Δ).params

_cotangent_backing(Δ::ChainRulesCore.Tangent) = ChainRulesCore.backing(Δ)
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
    _storage_component(prototype, s, nt)
end

_storage_from_backing(_, Δ) = Δ

# Which field of the prototype the storage is, and the component of the cotangent that belongs to it,
# decided in one generated body — written out for the reason `_accumulate_named!` above is, and this
# is the same walk's tail.
#
# A `for` over `fieldnames(typeof(prototype))` indexes the struct at a *runtime* `Symbol`, so
# `getfield` comes back as the union of the type's field types and the comparison is dispatched
# dynamically once per field: the reverse pass pays for every field the leaf carries, and not only for
# the one that holds its numbers. Handing back the field's *name* is not enough to stop that, because
# the name is then a runtime `Symbol` in `_cotangent_get` and its `haskey` is a runtime question —
# which is the point `_matching_named` below makes about keys. Splicing the whole selection keeps both
# constant on every arm.
#
# `@allocated` on the pullback with the call warmed, one process per reading, Julia 1.13.0-rc3 on an
# Apple M4 Max, for the same three numbers of storage behind one field of metadata and behind five:
#
# | | storage 1 of 2 fields | storage 6 of 6 |
# |---|---|---|
# | a `for` over the field names | 160 B | 512 B |
# | the name spliced in | 128 B | 128 B |
# | the selection spliced in | 96 B | 96 B |
#
# It is the second column that says what was wrong, not the first: what the loop cost was the number
# of fields. `test/derivative_tests.jl` asserts the two columns are equal rather than pinning either,
# since the figure is the whole pullback's and `@allocated` jitters on Windows.
@generated function _storage_component(prototype, storage, nt)
    expr = :nt
    for f in reverse(fieldnames(prototype))
        expr = :(getfield(prototype, $(QuoteNode(f))) === storage ?
                 _cotangent_get(nt, $(QuoteNode(f))) : $expr)
    end
    expr
end

# `freeparameters` of the prototype tells us which of the tangent's fields are the storage. Neither
# case closes over `nt`, for the reason `_unflatten_children` does not — and the positional one used to
# be an `ntuple` over a *runtime* length, which stops being inferable past ten blocks.
#
# Both walk the storage of *one leaf*, so neither is wide the way a branch of layers is; the named one
# is written out all the same, because splicing the keys in as literals is what keeps
# `_cotangent_get`'s `haskey` a compile-time question.
function _matching_storage(storage::NamedTuple, nt::NamedTuple)
    NamedTuple{keys(storage)}(_matching_named(storage, nt))
end
function _matching_storage(storage::Tuple, nt::NamedTuple)
    _matching_positional(storage, values(nt))
end
_matching_storage(_, nt::NamedTuple) = nt

@generated function _matching_named(storage::NamedTuple{Keys}, nt) where {Keys}
    :(($((:(_cotangent_get(nt, $(QuoteNode(k)))) for k in Keys)...),))
end

@inline _matching_positional(::Tuple{}, _) = ()
@inline _matching_positional(storage::Tuple, Δ) = (
    _cotangent_head(Δ), _matching_positional(Base.tail(storage), _cotangent_tail(Δ))...)

# ---------------------------------------------------------------------------------------------------
# A cotangent tree, leaf by leaf against the parameters
#
# `_map_cotangent(f, ps, Δ)` walks the branches of the primal `ps` and the cotangent `Δ` together and
# returns the cotangent with `f(leaf, Δleaf)` at each leaf. A hole — `nothing` or any flavour of zero,
# at a leaf or at a whole branch — comes back as `nothing` and `f` never sees it. The result has the
# shape Zygote gives a `NamedTuple`: keyed by the primal's keys, `nothing` where nothing was touched.
#
# Both callers run once per reverse pass: the `ZygoteRules` extension with `storage_gradient`, and
# `ProjectTo(::NetworkParameters)` with each leaf's projection. So the named walk is written out for
# the reason `_accumulate_named!` is.
# ---------------------------------------------------------------------------------------------------

function _map_cotangent(f::F, x::NamedTuple, Δ) where {F}
    Δ = _normalized(Δ)
    Δ === nothing && return nothing
    _map_cotangent_named(f, x, _cotangent_backing(Δ))
end

function _map_cotangent(f::F, x::Tuple, Δ) where {F}
    Δ = _normalized(Δ)
    Δ === nothing && return nothing
    _map_cotangent_positional(f, x, _cotangent_backing(Δ))
end

# A set inside a set: its gradient is a set too. Zygote hands its cotangent over as the structural
# `(params = …,)`, and a gradient already rewrapped arrives as the type itself.
function _map_cotangent(f::F, x::NetworkParameters, Δ) where {F}
    Δ = _normalized(Δ)
    Δ === nothing && return nothing
    NetworkParameters(_map_cotangent(f, params(x), _unwrap_parameters(Δ)))
end

function _map_cotangent(f::F, x, Δ) where {F}
    Δ = _normalized(Δ)
    Δ === nothing && return nothing
    f(x, Δ)
end

@generated function _map_cotangent_named(f, x::NamedTuple{Keys}, Δ) where {Keys}
    children = [:(_map_cotangent(f, getfield(x, $i), _cotangent_get(Δ, $(QuoteNode(Keys[i])))))
                for i in 1:length(Keys)]
    :(NamedTuple{Keys}(($(children...),)))
end

@inline _map_cotangent_positional(f, ::Tuple{}, Δ) = ()
@inline _map_cotangent_positional(f, xs::Tuple, Δ) = (
    _map_cotangent(f, first(xs), _cotangent_head(Δ)),
    _map_cotangent_positional(f, Base.tail(xs), _cotangent_tail(Δ))...)

# `Zygote.gradient` projects its result with `ProjectTo` of the argument. For a `NetworkParameters` that
# is the projection of each leaf, as it is for the `NamedTuple` inside: a leaf read twice gets a dense
# `Matrix` cotangent, and its own projection gives back the leaf's type. Only an array or a number is
# projected. A structural tangent over a leaf's fields is a tangent already, and `ProjectTo` of an
# array does not accept one in the shape Zygote gives it, a `NamedTuple`.
function ChainRulesCore.ProjectTo(ps::NetworkParameters)
    ChainRulesCore.ProjectTo{NetworkParameters}(; params = params(ps))
end

function (project::ChainRulesCore.ProjectTo{NetworkParameters})(Δ::NetworkParameters)
    NetworkParameters(_map_cotangent(_project_leaf, project.params, params(Δ)))
end

_project_leaf(x, Δ::Union{AbstractArray, Number}) = ChainRulesCore.ProjectTo(x)(Δ)
_project_leaf(_, Δ) = Δ

# Two gradients of one set add leaf by leaf, with `nothing` a zero. Zygote adds the cotangents of the
# uses of an argument with `+`, and a loss that calls `flatten` twice has two `NetworkParameters` from
# the `flatten` rule to add. A leaf adds at the level of its storage, as `mapstorage` does, so a
# structured leaf stays one rather than becoming the sum of two dense interfaces.
function Base.:+(a::NetworkParameters, b::NetworkParameters)
    NetworkParameters(_add_cotangents(
        params(a), params(b)))
end

function _add_cotangents(a::NamedTuple{Keys}, b::NamedTuple{Keys}) where {Keys}
    map(_add_cotangents, a, b)
end
_add_cotangents(a::Tuple, b::Tuple) = map(_add_cotangents, a, b)
_add_cotangents(a::NetworkParameters, b::NetworkParameters) = a + b
_add_cotangents(a, b) = _add_leaves(a, b)

_add_leaves(::Nothing, ::Nothing) = nothing
_add_leaves(::Nothing, b) = b
_add_leaves(a, ::Nothing) = a
_add_leaves(a, b) = mapstorage(+, a, b)
