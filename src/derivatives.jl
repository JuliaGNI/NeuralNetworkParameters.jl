# Reverse-mode rules for the two conversions.
#
# These are what make "differentiate with respect to the flat form, get the answer back in the shape of
# the network" work under Zygote. `ForwardDiff` needs no rule: `unflatten` is generic in the element type
# of its vector, so a `Dual`-valued vector simply produces `Dual`-valued parameters.
#
# At a fixed layout the two conversions are linear and mutually adjoint, so each rule is the other
# conversion. The reverse rule of `unflatten` converts the cotangent of the set it returned to the
# gradient of that set with `map_cotangent`, as the `ZygoteRules` extension does, and flattens it: the
# flat and the structured gradient are one gradient.

function ChainRulesCore.rrule(::typeof(unflatten), layout::ParameterLayout, v::AbstractVector)
    ps = unflatten(layout, v)

    # The layout is built again from `ps`, which costs no allocation, rather than kept: the reverse
    # pass keeps this closure, and a layout holds more than the set it describes.
    function unflatten_pullback(Δ)
        g = map_cotangent(storage_gradient, ps, Δ)
        Δv = g === nothing ? zero(v) : flatten!(similar(v), g, parameterlayout(ps))
        ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Δv
    end

    ps, unflatten_pullback
end

function ChainRulesCore.rrule(::typeof(flatten), ::Type{T}, ps) where {T}
    v, layout = flatten(T, ps)

    function flatten_pullback(Δ)
        Δv = _flat_cotangent(_normalized(Δ))
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
# The cotangent is put through `_normalized` first, as everywhere else in this file, so every flavour
# of zero for the pair — `nothing`, a `ZeroTangent`, a `Tangent` with no fields — is `nothing` here.
_flat_cotangent(::Nothing) = nothing
_flat_cotangent(Δ::Tuple) = _normalized(first(Δ))
_flat_cotangent(Δ::ChainRulesCore.Tangent) = _normalized(first(ChainRulesCore.backing(Δ)))

# ---------------------------------------------------------------------------------------------------
# Reading a cotangent
#
# Every cotangent is put through `_normalized` before it is dispatched on, which unthunks it and turns
# any flavour of zero into `nothing`.
# ---------------------------------------------------------------------------------------------------

# A `Tangent` with no fields is a zero too: `Tangent{P}()` is backed by `()` for a `Tuple` primal and
# by an empty `NamedTuple` for any other.
const _EmptyTangent = ChainRulesCore.Tangent{
    <:Any, <:Union{Tuple{}, NamedTuple{(), Tuple{}}}}

_normalized(Δ::ChainRulesCore.AbstractThunk) = _normalized(ChainRulesCore.unthunk(Δ))
_normalized(::ChainRulesCore.AbstractZero) = nothing
_normalized(::Nothing) = nothing
_normalized(::_EmptyTangent) = nothing
_normalized(Δ) = Δ

# A cotangent for a `NetworkParameters` arrives as the structural tangent of the wrapper, whose single
# field is the wrapped `NamedTuple` — a `Tangent` in a rule, a `NamedTuple` from Zygote's own reverse
# pass, for the set a loss differentiates and for a set nested in it — or as the type itself, which is
# what the `flatten` rule returns.
# Anything else is the cotangent of another tree, which both walks reject. A `Tangent` with no fields
# is a zero, and `_normalized` has made it `nothing` before it gets here.
_unwrap_parameters(Δ::NetworkParameters) = params(Δ)
_unwrap_parameters(Δ::NamedTuple{(:params,)}) = Δ.params
function _unwrap_parameters(Δ::ChainRulesCore.Tangent{<:Any, <:NamedTuple{(:params,)}})
    ChainRulesCore.backing(Δ).params
end
function _unwrap_parameters(Δ)
    throw(_shape_error("that is a `NetworkParameters`, whose cotangent is `(params = …,)`",
        _type_of(Δ)))
end

_cotangent_backing(Δ::ChainRulesCore.Tangent) = ChainRulesCore.backing(Δ)
_cotangent_backing(Δ) = Δ

# Anything the cotangent is silent about becomes `nothing`, i.e. a structural zero.
_cotangent_get(nt::NamedTuple, k::Symbol) = haskey(nt, k) ? nt[k] : nothing

# ---------------------------------------------------------------------------------------------------
# The storage gradient from a structural tangent
#
# A loss that reads the storage field of a structured leaf gets a tangent over the leaf's fields from
# Zygote, `(S = …, n = nothing)`, which holds ∂L/∂S already. The default `storage_gradient` for it
# rebuilds the leaf around the component of the tangent that belongs to its storage, converting each
# block in turn, so that a structured block inside a structured leaf is converted once, by its own
# method, and a block the loss did not read is a zero block.
# ---------------------------------------------------------------------------------------------------

function storage_gradient(leaf, Δ::Union{NamedTuple, ChainRulesCore.Tangent})
    s = freeparameters(leaf)
    rebuild(leaf, _map_child(storage_gradient, s, _storage_cotangent(leaf, s, _cotangent_backing(Δ))))
end

# Storage in several blocks is matched to the tangent by key, for a `NamedTuple` of blocks, and by the
# field each block is, for a `Tuple` of blocks. A block the tangent leaves out is a structural zero.
function _storage_cotangent(_, storage::NamedTuple, nt::NamedTuple)
    NamedTuple{keys(storage)}(_matching_named(storage, nt))
end
function _storage_cotangent(leaf, storage::Tuple, nt::NamedTuple)
    map(block -> _storage_component(leaf, block, nt), storage)
end

# Storage in one block is one of the leaf's fields, `(S = …, n = …)` for a leaf whose storage is its `S`.
_storage_cotangent(leaf, storage, nt::NamedTuple) = _storage_component(leaf, storage, nt)

# Which field of the prototype the storage is, and the component of the cotangent that belongs to it,
# decided in one generated body: the field names and the keys are literals on every arm, so neither the
# `getfield` nor the `haskey` is a runtime question.
@generated function _storage_component(prototype, storage, nt)
    expr = :(_no_storage_field_error(prototype, nt))
    for f in reverse(fieldnames(prototype))
        expr = :(getfield(prototype, $(QuoteNode(f))) === storage ?
                 _cotangent_get(nt, $(QuoteNode(f))) : $expr)
    end
    expr
end

@noinline function _no_storage_field_error(prototype, nt)
    throw(ArgumentError(string(
        "cannot read a cotangent of type `", typeof(nt), "` as the derivative of a leaf of type `",
        typeof(prototype), "`: its `freeparameters` is none of its fields.")))
end

@generated function _matching_named(storage::NamedTuple{Keys}, nt) where {Keys}
    :(($((:(_cotangent_get(nt, $(QuoteNode(k)))) for k in Keys)...),))
end

# ---------------------------------------------------------------------------------------------------
# A cotangent tree, leaf by leaf against the parameters
#
# The callers in this package run once per reverse pass: the reverse rule of `unflatten` and the
# `ZygoteRules` extension with `storage_gradient`, and `ProjectTo(::NetworkParameters)` with each
# leaf's projection. The named walk is written out for the reason the head of `walk.jl` gives.
# ---------------------------------------------------------------------------------------------------

"""
    map_cotangent(f, ps, Δ)

Walk the branches of the primal `ps` and of its cotangent `Δ` together, and return the cotangent
with `f(leaf, Δleaf)` at each leaf. The result has the shape of the primal: keyed by the primal's
keys for a `NamedTuple`, positional for a `Tuple`, and a `NetworkParameters` for a
`NetworkParameters`.

A hole is `nothing`, any flavour of zero, or a `Tangent` with no fields. Where `Δ` as a whole is a
hole, the result is `nothing`. A hole inside `Δ`, at a leaf or a whole branch, comes back as a zero
leaf of each leaf's own type, `mapstorage(zero, leaf)`, and `f` never sees it. A leaf with no numbers,
whose [`parameter_eltype`](@ref) is `Union{}` — `nothing`, or a function — has no zero and stays
`nothing`. So the result is complete wherever `Δ` touched the set at all.

`Δ` is the cotangent of `ps` itself, in the shape Zygote gives it, and a cotangent of another shape
raises an `ArgumentError` at the first branch it does not fit. At a `NamedTuple` branch the cotangent
is a `NamedTuple`, or a `Tangent` over one, whose keys are keys of the branch; a key it leaves out is
a hole, as a field left out of a `Tangent` is. At a `Tuple` branch it is a `Tuple` of the same
length, or a `Tangent` over one. At a `NetworkParameters` it is the set's structural tangent
`(params = …,)`, as a `NamedTuple` or a `Tangent`, or a `NetworkParameters`. So `(params = …,)`
passed against `params(ps)` raises, and so does a `NetworkParameters` gradient passed against
`params(ps)`.

With `f = `[`storage_gradient`](@ref), this converts the natural cotangent Zygote gives a parameter
set to its gradient with respect to the storage of each leaf. A package that writes its own
`ZygoteRules.pullback` for a function of a parameter set calls it on the cotangent of the set:

```julia
y, pb = ZygoteRules.pullback(g, x, values(ps))
function g_pullback(ȳ)
    x̄, p̄ = pb(ȳ)
    p̄ = map_cotangent(storage_gradient, values(ps), p̄)
    x̄, p̄ === nothing ? nothing : NetworkParameters(NamedTuple{keys(ps)}(p̄))
end
```

A `NetworkParameters` inside `Δ` holds the storage gradient at every leaf already — it is what the
reverse rule of [`flatten`](@ref) returns — so `map_cotangent(storage_gradient, …)` returns it as it
is.
"""
function map_cotangent(f::F, x, Δ) where {F}
    Δ = _normalized_for(x, Δ)
    Δ === nothing ? nothing : _map_cotangent(f, x, Δ)
end

# A child of a branch: a hole there is a zero leaf, or a branch of zero leaves.
function _map_child(f::F, x, Δ) where {F}
    Δ = _normalized_for(x, Δ)
    Δ === nothing ? mapparameters(_zero_leaf, x) : _map_cotangent(f, x, Δ)
end

# A leaf with no numbers, one whose `parameter_eltype` is `Union{}`, has no zero and stays `nothing`.
_zero_leaf(x) = parameter_eltype(x) === Union{} ? nothing : mapstorage(zero, x)

# Each cotangent is normalised once, because unthunking a thunk computes it again. The structural
# tangent of a set whose `params` is a zero is a hole too, and otherwise comes back as
# `(params = …,)` with its `params` normalised.
_normalized_for(_, Δ) = _normalized(Δ)
function _normalized_for(::NetworkParameters, Δ)
    Δ = _normalized(Δ)
    (Δ === nothing || Δ isa NetworkParameters) && return Δ
    p = _normalized(_unwrap_parameters(Δ))
    p === nothing ? nothing : (params = p,)
end

function _map_cotangent(f::F, x::NamedTuple, Δ) where {F}
    _map_cotangent_named(f, x, _checked_branch(x, _cotangent_backing(Δ)))
end

function _map_cotangent(f::F, x::Tuple, Δ) where {F}
    _map_cotangent_positional(f, x, _checked_branch(x, _cotangent_backing(Δ)))
end

# A set, the one a loss differentiates or one inside it: its gradient is a set too. Zygote hands its
# cotangent over as the structural `(params = …,)`, and a gradient already rewrapped arrives as the
# type itself. That one comes from the `flatten` rule, or from `+` of two such, and holds the storage
# gradient at every leaf already, so `storage_gradient` leaves it as it is.
function _map_cotangent(f::F, x::NetworkParameters, Δ) where {F}
    NetworkParameters(_map_cotangent(f, params(x), _unwrap_parameters(Δ)))
end
_map_cotangent(::typeof(storage_gradient), ::NetworkParameters, Δ::NetworkParameters) = Δ

_map_cotangent(f::F, x, Δ) where {F} = f(x, Δ)

# The shape is checked first, so that a cotangent of another tree raises rather than giving a zero or
# a non-gradient at every leaf.
function _checked_branch(::NamedTuple{K}, Δ::NamedTuple{KΔ}) where {K, KΔ}
    all(in(K), KΔ) || throw(_shape_error("keyed by $K", "keyed by $KΔ"))
    Δ
end
function _checked_branch(x::Tuple, Δ::Tuple)
    length(Δ) == length(x) || throw(_shape_error(_positions(x), _positions(Δ)))
    Δ
end
function _checked_branch(::NamedTuple{K}, Δ) where {K}
    throw(_shape_error("keyed by $K", _type_of(Δ)))
end
_checked_branch(x::Tuple, Δ) = throw(_shape_error(_positions(x), _type_of(Δ)))

function _shape_error(branch, cotangent)
    ArgumentError(string(
        "cannot read a cotangent ", cotangent, " as the cotangent of a branch ",
        branch, ". The cotangent has the shape of the primal, as Zygote gives it."))
end
_type_of(Δ) = "of type `$(typeof(Δ))`"
_positions(x::Tuple) = length(x) == 1 ? "with 1 position" : "with $(length(x)) positions"

@generated function _map_cotangent_named(f, x::NamedTuple{Keys}, Δ) where {Keys}
    children = [:(_map_child(f, getfield(x, $i), _cotangent_get(Δ, $(QuoteNode(Keys[i])))))
                for i in eachindex(Keys)]
    :(NamedTuple{Keys}(($(children...),)))
end

# The cotangent of a `Tuple` branch has its length, which `_checked_branch` has asked.
@inline _map_cotangent_positional(f, ::Tuple{}, ::Tuple{}) = ()
@inline _map_cotangent_positional(f, xs::Tuple, Δ::Tuple) = (
    _map_child(f, first(xs), first(Δ)),
    _map_cotangent_positional(f, Base.tail(xs), Base.tail(Δ))...)

# `Zygote.gradient` projects its result with `ProjectTo` of the argument. For a `NetworkParameters` that
# is the projection of each leaf, as it is for the `NamedTuple` inside: a leaf read twice gets a dense
# `Matrix` cotangent, and its own projection gives back the leaf's type. Only an array or a number is
# projected. A structural tangent over a leaf's fields is a tangent already, and `ProjectTo` of an
# array does not accept one in the shape Zygote gives it, a `NamedTuple`.
function ChainRulesCore.ProjectTo(ps::NetworkParameters)
    ChainRulesCore.ProjectTo{NetworkParameters}(; params = params(ps))
end

function (project::ChainRulesCore.ProjectTo{NetworkParameters})(Δ::NetworkParameters)
    NetworkParameters(map_cotangent(_project_leaf, project.params, params(Δ)))
end

_project_leaf(x, Δ::Union{AbstractArray, Number}) = ChainRulesCore.ProjectTo(x)(Δ)
_project_leaf(_, Δ) = Δ

# Two gradients of one set add leaf by leaf. Zygote adds the cotangents of the uses of an argument with
# `+`, and a loss that calls `flatten` twice has two `NetworkParameters` from the `flatten` rule to add.
# A leaf adds at the level of its storage, so a structured leaf stays one rather than becoming the sum
# of two dense interfaces.
Base.:+(a::NetworkParameters, b::NetworkParameters) = mapstorage(+, a, b)
