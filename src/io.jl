# Storage generics. The methods live in `ext/HDF5Ext.jl` so that HDF5 — and the binary chain behind it
# — is only loaded by users who ask for it, matching what `AbstractNeuralNetworks` already does.

"""
    h5save(h5, x, path)

Write `x` into the HDF5 store `h5` at `path`. Implemented in the `HDF5Ext` extension; load `HDF5` to
get the methods.
"""
function h5save end

"""
    h5load(h5)

Read back what [`h5save`](@ref) wrote. Implemented in the `HDF5Ext` extension.
"""
function h5load end

"""
    save(h5, ps)
    save(filename, ps)

Write a parameter set to an open HDF5 store or to a file. Implemented in the `HDF5Ext` extension.
"""
function save end

"""
    load(NetworkParameters, h5)
    load(NetworkParameters, h5, prototype)

Read a parameter set back. Implemented in the `HDF5Ext` extension.

The three-argument form rebuilds the leaves against `prototype`, a parameter set of the right shape,
and is the form that needs no registration — [`rebuild`](@ref) has something to work from. The
two-argument form has to reconstruct structured leaves from what the file says about them, which is
what [`register_parameter_type!`](@ref) is for.
"""
function load end

const PARAMETER_TYPE_REGISTRY = Dict{String, Any}()

"""
    register_parameter_type!(name, reconstruct)

Teach [`load`](@ref) how to rebuild a structured leaf that was stored under `name`, when there is no
prototype to rebuild against.

`reconstruct(storage, metadata)` receives what [`freeparameters`](@ref) produced and the
[`parameter_metadata`](@ref) that went with it, and returns the leaf:

```julia
NeuralNetworkParameters.parameter_metadata(A::SymmetricMatrix) = (n = A.n,)
register_parameter_type!("SymmetricMatrix", (S, md) -> SymmetricMatrix(S, md.n))
```

A package that owns a parameter type registers it in its own `__init__`. Nothing here needs to know
the type exists, which is the point: with the structured matrix types living upstream of the package
that trains with them, the only way to serialise them without somebody committing type piracy is for
the serialiser to be driven by a protocol rather than by a list.

The registry is only consulted by the two-argument [`load`](@ref); the prototype form bypasses it.
"""
function register_parameter_type!(name::AbstractString, reconstruct)
    PARAMETER_TYPE_REGISTRY[String(name)] = reconstruct
    reconstruct
end

"""
    parameter_type_name(x)

The name a structured leaf is stored under, `nameof(typeof(x))` by default. Paired with
[`register_parameter_type!`](@ref).
"""
parameter_type_name(x) = string(nameof(typeof(x)))
