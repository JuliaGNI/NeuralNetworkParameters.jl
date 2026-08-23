using NeuralNetworkParameters
using NeuralNetworkParameters: save, load, register_parameter_type!, PARAMETER_TYPE_REGISTRY
using HDF5
using Test

include("wrapper_types.jl")

register_parameter_type!("Sym", (S, md) -> Sym(S, md.n))
register_parameter_type!("TwoBlock", (data, md) -> TwoBlock(data.A, data.B, md.N))

tmp() = tempname() * ".h5"

@testset "round trip" begin
    ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]), L2 = (W = [4.0;;],)))
    f = tmp()
    @test save(f, ps) == f
    @test load(NetworkParameters, f) == ps
end

@testset "ten layers keep their order" begin
    # HDF5 returns group members sorted, so without the recorded key order this reads back as
    # L1, L10, L2, … and compares unequal
    ps = NetworkParameters(NamedTuple(Symbol("L$i") => (W = [Float64(i)],) for i in 1:10))
    f = tmp()
    save(f, ps)
    back = load(NetworkParameters, f)
    @test keys(back) == keys(ps)
    @test back == ps
end

@testset "element type is preserved" begin
    f = tmp()
    save(f, NetworkParameters((a = Float32[1, 2],)))
    @test eltype(load(NetworkParameters, f).a) === Float32
end

@testset "structured leaves, through the registry" begin
    ps = NetworkParameters((
        L1 = (S = sample_sym(), W = [4.0 5.0]), L2 = (G = sample_twoblock(),)))
    f = tmp()
    save(f, ps)
    back = load(NetworkParameters, f)
    @test back.L1.S isa Sym
    @test back.L1.S.S == ps.L1.S.S
    @test back.L1.S.n == 2
    @test back.L2.G isa TwoBlock
    @test back.L2.G.A isa Sym
    @test back.L2.G.A.S == ps.L2.G.A.S
    @test back.L2.G.B == ps.L2.G.B
    @test back.L2.G.N == 3
end

@testset "structured leaves, against a prototype" begin
    ps = NetworkParameters((L1 = (S = sample_sym(),),))
    f = tmp()
    save(f, ps)
    # the prototype form does not consult the registry at all
    saved = copy(PARAMETER_TYPE_REGISTRY)
    try
        empty!(PARAMETER_TYPE_REGISTRY)
        back = load(NetworkParameters, f, ps)
        @test back.L1.S isa Sym
        @test back.L1.S.S == ps.L1.S.S
        @test back.L1.S.n == 2
        # and without either, the error says what to do
        e = try
            load(NetworkParameters, f)
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("register_parameter_type!", sprint(showerror, e))
    finally
        merge!(PARAMETER_TYPE_REGISTRY, saved)
    end
end

@testset "scalar, empty and tuple leaves" begin
    ps = NetworkParameters((a = 1.5, b = Float64[], c = [2.0 3.0], d = ([1.0], [2.0, 3.0])))
    f = tmp()
    save(f, ps)
    back = load(NetworkParameters, f)
    @test back.a == 1.5
    @test back.b == Float64[]
    @test back.c == [2.0 3.0]
    @test back.d == ([1.0], [2.0, 3.0])
end

@testset "an open store can be written to and read from" begin
    ps = NetworkParameters((L1 = (W = [1.0 2.0],),))
    f = tmp()
    h5open(f, "w") do h5
        save(h5, ps)
    end
    h5open(f, "r") do h5
        @test load(NetworkParameters, h5) == ps
    end
end

@testset "FlatParameters" begin
    ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
    f = tmp()
    save(f, FlatParameters(ps))
    fp = load(FlatParameters, f)
    @test fp isa FlatParameters
    @test collect(fp) == [1.0, 2.0, 3.0]
    @test NetworkParameters(fp) == ps
end

@testset "files written before this package still load" begin
    # what AbstractNeuralNetworks wrote: plain nested groups, no attributes
    f = tmp()
    h5open(f, "w") do h5
        for (name, w) in (("L1", [1.0 2.0]), ("L2", [3.0 4.0]))
            g = HDF5.create_group(h5, name)
            g["W"] = w
        end
    end
    back = load(NetworkParameters, f)
    @test back.L1.W == [1.0 2.0]
    @test back.L2.W == [3.0 4.0]

    # and the same with the layer order that used to be lost
    f2 = tmp()
    h5open(f2, "w") do h5
        for i in 1:10
            g = HDF5.create_group(h5, "L$i")
            g["W"] = [Float64(i)]
        end
    end
    @test keys(load(NetworkParameters, f2)) == Tuple(Symbol("L$i") for i in 1:10)
end

@testset "and the ones GeometricMachineLearning tagged" begin
    # The other branch of the legacy reader: a group carrying a `gml_type` attribute, with the type's
    # fields under their own names. Such a file records no storage/metadata split, so there is nothing
    # here that could tell one from the other — the group's fields go to the reconstructor in *both*
    # positions, and normalising them is the job of the package that owns the type.
    # `GeometricOptimizers` is written against exactly that; this is what pins it.
    seen = Ref{Any}(nothing)
    register_parameter_type!("GmlSym", function (storage, metadata)
        seen[] = (storage, metadata)
        storage isa NamedTuple ? Sym(storage.S, storage.n) : Sym(storage, metadata.n)
    end)

    f = tmp()
    h5open(f, "w") do h5
        g = HDF5.create_group(h5, "L1")
        gW = HDF5.create_group(g, "W")
        HDF5.attributes(gW)["gml_type"] = "GmlSym"
        gW["S"] = [1.0, 2.0, 3.0]
        gW["n"] = 2
        g["b"] = [4.0]
    end

    back = load(NetworkParameters, f)
    @test back.L1.W isa Sym
    @test back.L1.W == sample_sym()
    @test back.L1.b == [4.0]

    # what the reconstructor was handed: the group's fields, the same object in both positions
    storage, metadata = seen[]
    @test storage isa NamedTuple
    @test metadata === storage
    # and by name only. A file in this layout records no key order, so the order these come back in is
    # whatever HDF5 gives for the group — which is why a reconstructor for it has to reach for
    # `storage.S` rather than `storage[1]`.
    @test issetequal(keys(storage), (:S, :n))

    # an unregistered tag says what to do about it rather than guessing
    f2 = tmp()
    h5open(f2, "w") do h5
        g = HDF5.create_group(h5, "L1")
        gW = HDF5.create_group(g, "W")
        HDF5.attributes(gW)["gml_type"] = "NotRegisteredAnywhere"
        gW["S"] = [1.0]
    end
    err = try
        load(NetworkParameters, f2)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("register_parameter_type!", err.msg)
end
