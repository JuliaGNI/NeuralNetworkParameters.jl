@doc raw"""
    NetworkParameters(params::NamedTuple)
    NetworkParameters{Keys}(values)

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

# Implementation

`getproperty` is overloaded to reach into the wrapped `NamedTuple`, so the field itself has to be
read with `getfield` — which is what [`params`](@ref) does.
"""
struct NetworkParameters{Keys, ValueTypes}
    params::NamedTuple{Keys, ValueTypes}
end

NetworkParameters{Keys}(values) where {Keys} = NetworkParameters(NamedTuple{Keys}(values))

"""
    params(p::NetworkParameters)

The `NamedTuple` wrapped by `p`.

Also the accessor a `NeuralNetwork` uses for its own parameters in `AbstractNeuralNetworks`, hence
the short name.
"""
params(p::NetworkParameters) = getfield(p, :params)

Base.hasproperty(::NetworkParameters{Keys}, s::Symbol) where {Keys} = s ∈ Keys
Base.getproperty(p::NetworkParameters{Keys}, s::Symbol) where {Keys} = params(p)[s]
Base.propertynames(::NetworkParameters{Keys}) where {Keys} = Keys

Base.getindex(p::NetworkParameters, args...) = getindex(params(p), args...)
Base.keys(p::NetworkParameters) = keys(params(p))
Base.values(p::NetworkParameters) = values(params(p))
Base.length(p::NetworkParameters) = length(params(p))
Base.iterate(p::NetworkParameters, args...) = iterate(params(p), args...)
Base.pairs(p::NetworkParameters) = pairs(params(p))

# The conversion belongs to this package, since it owns the type. Defining it downstream would be
# piracy on both counts — `Base`'s constructor and this package's type — which is what
# GeometricMachineLearning [#207](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/207)
# had to do to write a nested parameter set to HDF5.
Base.NamedTuple(p::NetworkParameters) = params(p)

Base.isequal(p1::NetworkParameters, p2::NetworkParameters) = isequal(params(p1), params(p2))
Base.:(==)(p1::NetworkParameters, p2::NetworkParameters) = (params(p1) == params(p2))

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
