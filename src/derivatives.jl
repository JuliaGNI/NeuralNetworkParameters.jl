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
    # pass keeps this closure, and a layout holds more than the set it describes. The flat gradient
    # has the element type of the gradient, which is not `v`'s for an integer `v` or a `Dual`
    # cotangent.
    function unflatten_pullback(Δ)
        g = map_cotangent(storage_gradient, ps, Δ)
        Δv = g === nothing ? _zero_block(v, _gradient_eltype(eltype(v))) :
             flatten!(similar(v, parameter_eltype(g)), g, parameterlayout(ps))
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

# The components of the cotangent `nt` at the keys of the first argument, as a `NamedTuple` with those
# keys. A key the cotangent is silent about is `nothing`, i.e. a structural zero.
function _cotangents_at(::NamedTuple{Keys}, nt::NamedTuple) where {Keys}
    holes = NamedTuple{Keys}(ntuple(_ -> nothing, Val(length(Keys))))
    NamedTuple{Keys}(merge(holes, nt))
end

# The fields of a struct, as a `NamedTuple`.
_fields(x) = NamedTuple{fieldnames(typeof(x))}(_unrolled_map(identity, x))

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
    Δs = _storage_cotangent(leaf, s, _cotangent_backing(Δ))
    rebuild(leaf, _fill_holes(_map_child(storage_gradient, s, Δs)))
end

# Storage in several blocks is matched to the tangent by key, for a `NamedTuple` of blocks, and by
# field name, for a `Tuple` of blocks. A block the tangent leaves out is a structural zero.
_storage_cotangent(_, storage::NamedTuple, nt::NamedTuple) = _cotangents_at(storage, nt)

# Block i of a `Tuple` storage is the leaf's i-th field, and takes the component of that field's name.
# The match is by name and not by value, because a number block is `===` to every equal number.
function _storage_cotangent(leaf, storage::Tuple{Vararg{Any, N}}, nt::NamedTuple) where {N}
    fields = _fields(leaf)
    N ≤ length(fields) || _no_storage_field_error(leaf, nt)
    all(ntuple(i -> fields[i] === storage[i], Val(N))) || _no_storage_field_error(leaf, nt)
    Δ = _cotangents_at(fields, nt)
    ntuple(i -> Δ[i], Val(N))
end

# Storage in one block is the one field of the leaf that it is `===` to, `(S = …, n = …)` for a leaf
# whose storage is its `S`. A storage that is `===` to two fields raises, as one that is none of them
# does.
function _storage_cotangent(leaf, storage, nt::NamedTuple)
    fields = _fields(leaf)
    match = _unrolled_foldl((m, f, Δf) -> _match_field(m, f, Δf, storage),
        _NoField(), fields, _cotangents_at(fields, nt))
    match isa _Field ? match.Δ : _no_storage_field_error(leaf, nt)
end

# The search for the field that the storage is. A field of another type than the storage cannot be
# `===` to it, so dispatch leaves it out and the result is inferred.
struct _NoField end
struct _TwoFields end
struct _Field{D}
    Δ::D
end
_match_field(m, f::T, Δf, storage::T) where {T} = f === storage ? _matched(m, Δf) : m
_match_field(m, _, _, _) = m
_matched(::_NoField, Δf) = _Field(Δf)
_matched(_, _) = _TwoFields()

@noinline function _no_storage_field_error(prototype, nt)
    throw(ArgumentError(string(
        "cannot read a cotangent of type `", typeof(nt), "` as the derivative of a leaf of type `",
        typeof(prototype), "`: its `freeparameters` is not exactly one of its fields, nor a `Tuple` ",
        "whose i-th block is its i-th field.")))
end

# ---------------------------------------------------------------------------------------------------
# A cotangent tree, leaf by leaf against the parameters
#
# The callers in this package run once per reverse pass: the reverse rule of `unflatten` and the
# `ZygoteRules` extension with `storage_gradient`, and `ProjectTo(::NetworkParameters)` with each
# leaf's projection.
# ---------------------------------------------------------------------------------------------------

"""
    map_cotangent(f, ps, Δ)

Walk the branches of the primal `ps` and of its cotangent `Δ` together, and return the cotangent
with `f(leaf, Δleaf)` at each leaf. The result has the shape of the primal: keyed by the primal's
keys for a `NamedTuple`, positional for a `Tuple`, and a `NetworkParameters` for a
`NetworkParameters`.

A hole is `nothing`, any flavour of zero, or a `Tangent` with no fields. Where `Δ` as a whole is a
hole, the result is `nothing`. A hole inside `Δ`, at a leaf or a whole branch, comes back as a zero
leaf of each leaf's own type, and `f` never sees it. Its element type is that of the leaves `f`
returned, so a gradient of `ForwardDiff.Dual`s has `Dual` zeros; where `f` returned no numbers, it is
the leaf's own, or `float` of it for an `Integer` leaf. A leaf with no numbers, whose
[`parameter_eltype`](@ref) is `Union{}` — `nothing`, or a function — has no zero and stays `nothing`.
So the result is complete wherever `Δ` touched the set at all.

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
    Δ === nothing ? nothing : _fill_holes(_map_cotangent(f, x, Δ))
end

# A child of a branch: a hole there is a `_Hole` at each leaf, or a branch of them. `_fill_holes`
# makes each a zero leaf at the end of the walk, when the element type of the gradient is known.
function _map_child(f::F, x, Δ) where {F}
    Δ = _normalized_for(x, Δ)
    Δ === nothing ? mapparameters(_hole, x) : _map_cotangent(f, x, Δ)
end

# A leaf with no numbers, one whose `parameter_eltype` is `Union{}`, has no zero and stays `nothing`.
_hole(x) = parameter_eltype(x) === Union{} ? nothing : _Hole(x)

# The place of a zero leaf for `leaf`. It has no element type, as a leaf without numbers has none, so
# the gradient's element type is that of the leaves `f` returned.
struct _Hole{X}
    leaf::X
end

# A hole becomes a zero leaf in the element type `S` of the leaves around it, or, where they hold no
# numbers, in the leaf's own gradient element type.
_fill_holes(g) = _filled(parameter_eltype(g), g)

function _filled(::Type{S}, g::NetworkParameters) where {S}
    NetworkParameters(_filled(S, params(g)))
end
function _filled(::Type{S}, g::Union{NamedTuple, Tuple}) where {S}
    _rewrap_children(g, _unrolled_map(Base.Fix1(_filled, S), g))
end
_filled(::Type, y) = y
function _filled(::Type{S}, h::_Hole) where {S}
    T = S === Union{} ? _gradient_eltype(parameter_eltype(h.leaf)) : S
    mapstorage(Base.Fix2(_zero_block, T), h.leaf)
end

_zero_block(b::AbstractArray, T) = fill!(similar(b, T), zero(T))
_zero_block(::Number, T) = zero(T)

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

function _map_cotangent(f::F, x::NamedTuple{Keys}, Δ) where {F, Keys}
    Δx = _cotangents_at(x, _checked_branch(x, _cotangent_backing(Δ)))
    NamedTuple{Keys}(_unrolled_map(_FixFirst(_map_child, f), x, Δx))
end

# The cotangent of a `Tuple` branch has its length, which `_checked_branch` has asked.
function _map_cotangent(f::F, x::Tuple, Δ) where {F}
    _unrolled_map(_FixFirst(_map_child, f), x, _checked_branch(x, _cotangent_backing(Δ)))
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
Base.:+(a::NetworkParameters, b::NetworkParameters) = mapstorage(_add_storage, a, b)

# A leaf with no numbers is `nothing` in both gradients, and their sum is `nothing`.
_add_storage(a, b) = a + b
mapstorage(::typeof(_add_storage), ::Nothing, ::Nothing) = nothing

# A loss that calls `flatten` and reads the set too has a gradient from the `flatten` rule and the
# structural tangent `(params = …,)` of the read to add. The tangent is converted to a gradient first,
# with the gradient's leaves in place of the primal ones, which `storage_gradient` permits.
function Base.:+(a::NetworkParameters, b::NamedTuple{(:params,)})
    g = map_cotangent(storage_gradient, a, b)
    g === nothing ? a : a + g
end
Base.:+(b::NamedTuple{(:params,)}, a::NetworkParameters) = a + b
