using NeuralNetworkParameters
using NeuralNetworkParameters: freeparameters, rebuild, parameter_eltype,
                               parameter_metadata,
                               isterminal, isparametertree
using Test

include("wrapper_types.jl")

@testset "defaults for ordinary leaves" begin
    A = [1.0 2.0]
    @test freeparameters(A) === A
    @test isterminal(A)
    @test rebuild(A, [3.0 4.0]) == [3.0 4.0]
    @test freeparameters(1.5) === 1.5
    @test rebuild(1.5, 2.5) === 2.5
    @test parameter_metadata(A) == NamedTuple()
end

struct Unregistered end

@testset "a type without the protocol says so" begin
    e = try
        freeparameters(Unregistered())
    catch err
        err
    end
    @test e isa ArgumentError
    @test occursin("freeparameters", sprint(showerror, e))
    @test occursin("Unregistered", sprint(showerror, e))
end

@testset "structured leaves" begin
    A = sample_sym()
    @test freeparameters(A) === A.S
    @test !isterminal(A)
    @test parameter_metadata(A) == (n = 2,)
    # the prototype carries `n` across, so the type comes back unchanged
    B = rebuild(A, [7.0, 8.0, 9.0])
    @test B isa Sym
    @test B.S == [7.0, 8.0, 9.0]
    @test B.n == 2
end

@testset "branches and leaves" begin
    @test isparametertree(NetworkParameters((a = [1.0],)))
    @test isparametertree((a = [1.0],))
    @test isparametertree(([1.0],))
    @test !isparametertree([1.0])
    @test !isparametertree(sample_sym())
end

@testset "parameter_eltype" begin
    @test parameter_eltype(NetworkParameters((a = Float32[1, 2],))) === Float32
    @test parameter_eltype(NetworkParameters((a = [1.0],))) === Float64
    @test parameter_eltype((a = Float32[1], b = [2.0])) === Float64
    @test parameter_eltype(sample_sym()) === Float64
    @test parameter_eltype(sample_twoblock()) === Float64
    @test parameter_eltype(Float32[1, 2]) === Float32
    @test parameter_eltype(1.0f0) === Float32
end

@testset "for a NetworkParameters the element type is read off the type" begin
    p32 = NetworkParameters((a = Float32[1],))
    @test parameter_eltype(p32) === Float32
    # `<:` rather than `===`: what is pinned is that `T` is a constant the compiler already has, and
    # 1.14-DEV spells a constant type-valued result `Core.TypeEgal{Float32}` where 1.13 spells it
    # `Type{Float32}`
    @test only(Base.return_types(parameter_eltype, Tuple{typeof(p32)})) <: Type{Float32}
end

@testset "parameter_eltype is total where freeparameters is not" begin
    # every `NetworkParameters` runs its constructor through `parameter_eltype`, including sets that
    # hold no numbers at all, so this function cannot raise where `freeparameters` does
    @test parameter_eltype(nothing) === Union{}
    @test parameter_eltype((a = [1.0], b = nothing)) === Float64
    @test parameter_eltype(NamedTuple()) === Union{}
    # a leaf this package cannot read numbers out of contributes nothing, exactly as a gap does
    @test parameter_eltype(Unregistered()) === Union{}
    @test parameter_eltype((a = Unregistered(),)) === Union{}
    @test parameter_eltype((a = [1.0], b = Unregistered())) === Float64

    # `flatten` still says exactly what the missing protocol is, from the layout walk
    e = try
        flatten(NetworkParameters((a = Unregistered(),)))
    catch err
        err
    end
    @test e isa ArgumentError
    @test occursin("freeparameters", sprint(showerror, e))
end

# A leaf that keeps its numbers behind an interface that is not an array's. `Sym` and friends in
# `wrapper_types.jl` are all `AbstractMatrix`es, as every structured parameter type in the ecosystem
# is; this one is what the third method of the protocol exists for.
struct Blocks
    data::Vector{Float64}
end

NNP.freeparameters(b::Blocks) = b.data
NNP.rebuild(::Blocks, data) = Blocks(data)

@testset "a leaf that is not an array contributes nothing to the promotion" begin
    ps = NetworkParameters((L1 = (B = Blocks([1.0, 2.0]),),))
    @test parameter_eltype(ps) === Union{}
    @test ps isa NetworkParameters{Union{}}

    # the layout does find its numbers, through `freeparameters`, so `flatten` names the missing
    # method rather than building a `Vector{Union{}}` that fails on the first copy
    e = try
        flatten(ps)
    catch err
        err
    end
    @test e isa ArgumentError
    @test occursin("parameter_eltype", sprint(showerror, e))

    # naming the element type at the call is the other way out, and flattens as it always did
    v, layout = flatten(Float64, ps)
    @test v == [1.0, 2.0]
    @test unflatten(layout, [3.0, 4.0]).L1.B.data == [3.0, 4.0]
end

NNP.parameter_eltype(b::Blocks) = parameter_eltype(freeparameters(b))

@testset "...and opts into the recursion with one more method" begin
    ps = NetworkParameters((L1 = (B = Blocks([1.0, 2.0]),),))
    @test parameter_eltype(ps) === Float64
    @test ps isa NetworkParameters{Float64}
    @test first(flatten(ps)) == [1.0, 2.0]
end
