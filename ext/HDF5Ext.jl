module HDF5Ext

using HDF5
using NeuralNetworkParameters
import NeuralNetworkParameters: h5save, h5load, save, load, NetworkParameters, params,
                                FlatParameters, ParameterLayout, flatlayout,
                                freeparameters, rebuild, parameter_metadata,
                                parameter_type_name,
                                PARAMETER_TYPE_REGISTRY, flatten, unflatten, parameterlayout

# ---------------------------------------------------------------------------------------------------
# On-disk format
#
# Every group carries a `kind` attribute saying what to rebuild it as, and every keyed group carries
# its `keys` in order. The key order is the point: HDF5 hands back the members of a group sorted, so a
# network with ten layers reads back as `L1, L10, L2, …` unless the order is recorded. Guessing it from
# the names — sorting on a trailing integer — only works for names that happen to look like `L10`, and
# silently returns the wrong order for anything else.
#
# A structured leaf is a group tagged with `parameter_type`, holding the `freeparameters` under
# `storage` and the `parameter_metadata` under `metadata`. Nothing here knows which structured types
# exist: `load` looks the tag up in the registry that `register_parameter_type!` fills, or rebuilds
# against a prototype and skips the registry altogether.
# ---------------------------------------------------------------------------------------------------

const KIND_ATTR = "kind"
const KEYS_ATTR = "keys"
const TYPE_ATTR = "parameter_type"

function _group(h5::HDF5.H5DataStore, path::AbstractString)
    path == "/" ? h5 : (haskey(h5, path) ? h5[path] : HDF5.create_group(h5, path))
end

_set_attr!(h5, name, value) = (HDF5.attributes(h5)[name] = value; nothing)

# --- writing -----------------------------------------------------------------------------------

"""
    h5save(h5, x, path)

Write `x` — a parameter set, a branch of one, or a leaf — into `h5` at `path`.
"""
function h5save(h5::HDF5.H5DataStore, ps::NetworkParameters, path::AbstractString)
    g = _group(h5, path)
    _set_attr!(g, KIND_ATTR, "NetworkParameters")
    _save_keyed(g, params(ps))
end

function h5save(h5::HDF5.H5DataStore, nt::NamedTuple, path::AbstractString)
    g = _group(h5, path)
    _set_attr!(g, KIND_ATTR, "NamedTuple")
    _save_keyed(g, nt)
end

function h5save(h5::HDF5.H5DataStore, t::Tuple, path::AbstractString)
    g = _group(h5, path)
    _set_attr!(g, KIND_ATTR, "Tuple")
    _set_attr!(g, KEYS_ATTR, [string(i) for i in eachindex(t)])
    for (i, v) in enumerate(t)
        h5save(g, v, string(i))
    end
    nothing
end

function h5save(h5::HDF5.H5DataStore, x::AbstractArray, path::AbstractString)
    s = freeparameters(x)
    s === x ? (h5[path] = Array(x); nothing) : _save_wrapped(h5, x, s, path)
end

function h5save(h5::HDF5.H5DataStore, x::Number, path::AbstractString)
    h5[path] = x
    nothing
end

function _save_keyed(g, nt::NamedTuple)
    _set_attr!(g, KEYS_ATTR, [String(k) for k in keys(nt)])
    for (k, v) in pairs(nt)
        h5save(g, v, String(k))
    end
    nothing
end

function _save_wrapped(h5::HDF5.H5DataStore, x, storage, path::AbstractString)
    g = _group(h5, path)
    _set_attr!(g, KIND_ATTR, "wrapped")
    _set_attr!(g, TYPE_ATTR, parameter_type_name(x))
    h5save(g, storage, "storage")
    md = parameter_metadata(x)
    if !isempty(md)
        mg = _group(g, "metadata")
        _set_attr!(mg, KIND_ATTR, "NamedTuple")
        _save_keyed(mg, md)
    end
    nothing
end

# --- reading -----------------------------------------------------------------------------------

"""
    h5load(h5)
    h5load(h5, prototype)

Read back what [`h5save`](@ref) wrote.

With a `prototype` — the parameter set the file was written from, or any of the same shape — the
leaves are rebuilt with [`rebuild`](@ref) and no type needs to have been registered.
"""
h5load(ds::HDF5.Dataset) = read(ds)

function h5load(g::Union{HDF5.Group, HDF5.File})
    kind = _kind(g)
    kind == "NetworkParameters" && return NetworkParameters(_load_keyed(g))
    kind == "NamedTuple" && return _load_keyed(g)
    kind == "Tuple" && return _load_tuple(g)
    kind == "wrapped" && return _load_wrapped(g)
    _load_legacy(g)
end

h5load(ds::HDF5.Dataset, prototype) = _match_read(read(ds), prototype)

function h5load(g::Union{HDF5.Group, HDF5.File}, prototype)
    if prototype isa NetworkParameters
        return NetworkParameters(h5load(g, params(prototype)))
    elseif prototype isa NamedTuple
        return NamedTuple{keys(prototype)}(map(k -> h5load(g[String(k)], prototype[k]),
            keys(prototype)))
    elseif prototype isa Tuple
        return ntuple(i -> h5load(g[string(i)], prototype[i]), length(prototype))
    end
    # a structured leaf: read its storage and rebuild against the prototype
    s = freeparameters(prototype)
    s === prototype &&
        throw(ArgumentError(string("expected a dataset for the terminal leaf of type ",
            typeof(prototype), ", found a group")))
    rebuild(prototype, h5load(g["storage"], s))
end

_match_read(data, prototype) = data

function _kind(g)
    a = HDF5.attributes(g)
    haskey(a, KIND_ATTR) ? read(a[KIND_ATTR]) : ""
end

function _stored_keys(g)
    a = HDF5.attributes(g)
    haskey(a, KEYS_ATTR) ? String.(read(a[KEYS_ATTR])) : _legacy_sorted_keys(keys(g))
end

function _load_keyed(g)
    ks = _stored_keys(g)
    NamedTuple{Tuple(Symbol.(ks))}(Tuple(h5load(g[k]) for k in ks))
end

_load_tuple(g) = Tuple(h5load(g[k]) for k in _stored_keys(g))

function _load_wrapped(g)
    name = read(HDF5.attributes(g)[TYPE_ATTR])
    haskey(PARAMETER_TYPE_REGISTRY, name) || throw(ArgumentError(string(
        "the file contains a parameter of type `", name, "`, which is not registered.\n",
        "The package that owns `", name, "` has to call\n",
        "    NeuralNetworkParameters.register_parameter_type!(\"", name, "\", (storage, metadata) -> ...)\n",
        "or the parameters have to be loaded against a prototype: `load(NetworkParameters, h5, prototype)`.")))
    storage = h5load(g["storage"])
    md = haskey(g, "metadata") ? _load_keyed(g["metadata"]) : NamedTuple()
    PARAMETER_TYPE_REGISTRY[name](storage, md)
end

# --- files written before this package ---------------------------------------------------------
#
# `AbstractNeuralNetworks` wrote plain nested groups with no attributes, and `GeometricMachineLearning`
# tagged its structured matrices with `gml_type`. Both still load.

function _load_legacy(g)
    a = HDF5.attributes(g)
    if haskey(a, "gml_type")
        name = read(a["gml_type"])
        haskey(PARAMETER_TYPE_REGISTRY, name) || throw(ArgumentError(string(
            "this file was written by GeometricMachineLearning and contains a `", name, "`, which is ",
            "not registered. Register it with `register_parameter_type!`, or load against a prototype.")))
        fields = _load_keyed(g)
        return PARAMETER_TYPE_REGISTRY[name](fields, fields)
    end
    _load_keyed(g)
end

# HDF5 sorts group members, so `L10` precedes `L2`. For a file that recorded no key order this is the
# best that can be done: sort on the trailing integer when *every* name has one, and otherwise leave
# the order alone rather than pretend to know it.
function _legacy_sorted_keys(ks)
    names = collect(String.(ks))
    all(k -> occursin(r"^\D+\d+$", k), names) || return names
    sort(names; by = k -> (m = match(r"^(\D+)(\d+)$", k); (m[1], parse(Int, m[2]))))
end

# --- entry points ------------------------------------------------------------------------------

"""
    save(h5, ps)
    save(filename, ps)

Write the parameter set `ps` to an open HDF5 store or to a file.
"""
save(h5::HDF5.H5DataStore, ps::Union{NetworkParameters, NamedTuple}) = h5save(h5, ps, "/")
save(h5::HDF5.H5DataStore, fp::FlatParameters) = save(h5, unflatten(fp))

function save(filename::AbstractString, ps)
    HDF5.h5open(filename, "w") do h5
        save(h5, ps)
    end
    filename
end

"""
    load(NetworkParameters, h5, [prototype])
    load(FlatParameters, h5, [prototype])

Read a parameter set back from an open HDF5 store or a file.
"""
load(::Type{NetworkParameters}, h5::HDF5.H5DataStore) = _as_parameters(h5load(h5["/"]))
function load(::Type{NetworkParameters}, h5::HDF5.H5DataStore, prototype)
    _as_parameters(h5load(h5["/"], prototype))
end

function load(::Type{FlatParameters}, h5::HDF5.H5DataStore, args...)
    FlatParameters(load(NetworkParameters, h5, args...))
end

function load(T::Type, filename::AbstractString, args...)
    HDF5.h5open(filename, "r") do h5
        load(T, h5, args...)
    end
end

_as_parameters(ps::NetworkParameters) = ps
_as_parameters(nt::NamedTuple) = NetworkParameters(nt)

end
