@doc raw"""
    NetworkParameters(params::NamedTuple)
    NetworkParameters{T}(params::NamedTuple)
    NetworkParameters{T, Keys, ValueTypes}(params::NamedTuple)

The parameters of a neural network: a `NamedTuple` whose entries follow the architecture, wrapped in
a type of its own.

The wrapper exists so that the parameters of a network are a type somebody *owns*. A bare
`NamedTuple` belongs to `Base`, so a package that wants to give the parameter set its own behaviour —
saving it to file, flattening it, stepping an optimizer over it — has to write methods on a signature
in which it owns nothing. Every such method is type piracy, and two packages doing it can silently
disagree.

Entries are reached with `getproperty` or `getindex`; the underlying `NamedTuple` is [`params`](@ref).

# Examples

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
ps.L1.W

# output

1×2 Matrix{Float64}:
 1.0  2.0
```

Both forms of access agree, and `keys` reports the layers:

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
(ps[:L2] === ps.L2, keys(ps))

# output

(true, (:L1, :L2))
```

# Element type

The first type parameter is the element type the leaves promote to, so that `T` *binds* in a method
signature:

```julia
f(ps::NetworkParameters{T}) where {T} = ...
g(x::Union{AbstractVector{T}, NetworkParameters{T}}) where {T} = ...
```

That is the reason it is a type parameter and not only a function. `GeometricOptimizers` takes the
element type from the *type* of the solution it is handed, and a parameter set could not join its
`OptimizerSolution{T}` union while it carried no such parameter.

It is derived by [`parameter_eltype`](@ref) at construction and never chosen; naming it, as
`NetworkParameters{T}(params)`, asserts it and raises if the leaves say otherwise. Note that it is a
*promotion*, not a guarantee of uniformity — a mixed set reports the type its leaves promote to while
each leaf keeps its own:

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = Float32[1 2], b = [3.0]),))
(ps isa NetworkParameters{Float64}, eltype(ps.L1.W))

# output

(true, Float32)
```

A set with nothing to promote — an empty one, or a gradient tree that is all gaps — reports `Union{}`.

# Implementation

`getproperty` is overloaded to reach into the wrapped `NamedTuple`, so the field itself has to be
read with `getfield` — which is what [`params`](@ref) does.

There is deliberately no `Base.eltype`: this type forwards the `NamedTuple` interface, for which
`eltype` means the type of what iteration yields — a layer's `NamedTuple` — rather than the numeric
element type. The numeric one is [`parameter_eltype`](@ref).
"""
struct NetworkParameters{T, Keys, ValueTypes}
    params::NamedTuple{Keys, ValueTypes}

    # `T` is derived rather than chosen, which makes this an inner constructor and suppresses the ones
    # Julia would otherwise write. `parameter_eltype` lives in `leaves.jl`, included after this file;
    # nothing calls a constructor at load time, so the binding exists before any caller reaches it.
    function NetworkParameters(params::NamedTuple{Keys, ValueTypes}) where {Keys, ValueTypes}
        T = parameter_eltype(params)
        new{T, Keys, ValueTypes}(params)
    end
end

# A caller that names `T` is asserting it, so these check rather than trust: a
# `NetworkParameters{T, Keys, ValueTypes}` whose leaves promote to something else has no inhabitants.
# The three-parameter form is not optional — `ChainRulesCore.construct` calls it from
# `+(::P, ::Tangent{P})`, which is how a parameter set and a cotangent add.
function NetworkParameters{T, Keys, ValueTypes}(nt) where {T, Keys, ValueTypes}
    ps = NetworkParameters(convert(NamedTuple{Keys, ValueTypes}, nt))
    ps isa NetworkParameters{T} || _element_type_error(ps, T)
    ps
end

function NetworkParameters{T}(nt) where {T}
    T isa Type || _keys_first_error(T)
    ps = NetworkParameters(nt)
    ps isa NetworkParameters{T} || _element_type_error(ps, T)
    ps
end

@noinline function _element_type_error(ps, T)
    throw(ArgumentError(string("the leaves of these parameters promote to ", parameter_eltype(ps),
        ", not to ", T, ". The element type of a `NetworkParameters` is derived from its leaves ",
        "rather than chosen, so naming it asserts it.")))
end

# The element type comes first, so `NetworkParameters{(:a, :b)}(values)` lands here with a tuple of
# symbols where a type belongs. Left to dispatch it is a `MethodError` about a conversion nobody
# asked for.
@noinline function _keys_first_error(Keys)
    throw(ArgumentError(string("`NetworkParameters{", Keys, "}(values)` names the element type, ",
        "which comes first. To build a set from its keys and values write ",
        "`NetworkParameters(NamedTuple{", Keys, "}(values))`.")))
end

"""
    params(p::NetworkParameters)

The `NamedTuple` wrapped by `p`.

Also the accessor a `NeuralNetwork` uses for its own parameters in `AbstractNeuralNetworks`, hence
the short name.
"""
params(p::NetworkParameters) = getfield(p, :params)

"""
    ParameterSet

Either of the two forms a whole set of parameters arrives in: a [`NetworkParameters`](@ref), or the
bare `NamedTuple` it wraps.

This is the type to dispatch on when a method takes *the parameters* and does not care which of the
two it was handed — a loss, a `changebackend`, a walk, an optimizer entry point. Every package in
this ecosystem was spelling the union out inline, and `SymbolicNeuralNetworks` had named it
`EquationSet` for the equation sets that share the shape; one name means a reader meets the same
type everywhere.

!!! note "Not the same question as `isparametertree`"
    [`isparametertree`](@ref) is true for a `Tuple` as well, because a `Tuple` *is* a branch the
    walks recurse into — it is what [`freeparameters`](@ref) returns for a multi-block leaf such as a
    horizontal lift, and why [`TupleLayout`](@ref) exists. It is never a set of parameters handed in
    whole, which is always keyed. So `x isa ParameterSet` and `isparametertree(x)` disagree on
    exactly the `Tuple`s, and each is right for its own question.

Note also what this does *not* say. It puts no bound on the element type and none on the depth: a
`ParameterSet` may nest to any depth and its leaves need not share a type. A caller that needs "flat,
and every leaf an `AbstractArray{T}`" wants a narrower type of its own —
`GeometricOptimizers.ParameterContainer{T}` is one — and cannot get it from here.

# Examples

```jldoctest
using NeuralNetworkParameters

nt = (L1 = (W = [1.0 2.0], b = [3.0]),)
(nt isa ParameterSet, NetworkParameters(nt) isa ParameterSet, nt.L1 isa ParameterSet)

# output

(true, true, true)
```

A leaf is not one, and neither is the `Tuple` a multi-block leaf is made of:

```jldoctest
using NeuralNetworkParameters

([1.0, 2.0] isa ParameterSet, ([1.0], [2.0]) isa ParameterSet)

# output

(false, false)
```
"""
const ParameterSet = Union{NetworkParameters, NamedTuple}

# The `<:Any` is the element type, which comes first so that `NetworkParameters{T}` binds `T` in a
# method signature — what `GeometricOptimizers` needs of a parameter set to take it as a solution.
# These three are the only methods here that name a type parameter at all.
Base.hasproperty(::NetworkParameters{<:Any, Keys}, s::Symbol) where {Keys} = s ∈ Keys
Base.getproperty(p::NetworkParameters{<:Any, Keys}, s::Symbol) where {Keys} = params(p)[s]
Base.propertynames(::NetworkParameters{<:Any, Keys}) where {Keys} = Keys

Base.getindex(p::NetworkParameters, args...) = getindex(params(p), args...)
Base.keys(p::NetworkParameters) = keys(params(p))
Base.values(p::NetworkParameters) = values(params(p))
Base.length(p::NetworkParameters) = length(params(p))
Base.iterate(p::NetworkParameters, args...) = iterate(params(p), args...)
Base.pairs(p::NetworkParameters) = pairs(params(p))

# Deliberately no `Base.eltype`. This type forwards the `NamedTuple` interface above, and for those
# methods `eltype` means the type of what iteration yields — a layer's `NamedTuple` — not the numeric
# element type. Defining it as the latter would make `eltype(ps)` and `eltype(collect(ps))` disagree.
# The numeric one is `parameter_eltype(ps)`, and in a signature it is `NetworkParameters{T}`.

# The conversion belongs to this package, since it owns the type. Defining it downstream would be
# piracy on both counts — `Base`'s constructor and this package's type — which is what
# GeometricMachineLearning [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207)
# had to do to write a nested parameter set to HDF5.
Base.NamedTuple(p::NetworkParameters) = params(p)

Base.isequal(p1::NetworkParameters, p2::NetworkParameters) = isequal(params(p1), params(p2))
Base.:(==)(p1::NetworkParameters, p2::NetworkParameters) = (params(p1) == params(p2))

# `hash` follows `isequal` through to the wrapped `NamedTuple`, or the two would disagree: the default
# `hash` takes in the type, and the element type is part of that, so a `Float32` set and a `Float64`
# set holding the same numbers — `isequal`, by the method above — would hash apart and behave as two
# different `Dict` keys.
Base.hash(p::NetworkParameters, h::UInt) = hash(params(p), h)

function Base.show(io::IO, ::MIME"text/plain", p::NetworkParameters)
    print(io, "NetworkParameters with ", length(p), length(p) == 1 ? " entry:" :
                                                    " entries:")
    for (k, v) in pairs(p)
        print(io, "\n  ", k, " => ")
        _show_entry(io, v)
    end
end

Base.show(io::IO, p::NetworkParameters) = print(io, "NetworkParameters(", params(p), ")")

function _show_entry(io::IO, v::NamedTuple)
    print(io, "(", join(("$k = $(_entry_summary(x))" for (k, x) in pairs(v)), ", "), ")")
end
_show_entry(io::IO, v) = print(io, _entry_summary(v))

_entry_summary(x::AbstractArray) = string(join(size(x), "×"), " ", nameof(typeof(x)))
_entry_summary(x) = summary(x)
