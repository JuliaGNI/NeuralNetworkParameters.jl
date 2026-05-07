module NeuralNetworkParameters

using AbstractNeuralNetworks

struct NetworkParameters{Keys, ValueTypes}
    params::NamedTuple{Keys, ValueTypes}
end

@inline Base.hasproperty(p::NetworkParameters, s::Symbol) = hasproperty(p.params, s)
@inline Base.getproperty(p::NetworkParameters, s::Symbol) = getproperty(p.params, s)
@inline Base.getindex(p::NetworkParameters, args...) = getindex(p.params, args...)






end
