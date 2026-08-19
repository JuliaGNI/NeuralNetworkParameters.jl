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

@testset "a type without the protocol says so" begin
    struct Unregistered end
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
