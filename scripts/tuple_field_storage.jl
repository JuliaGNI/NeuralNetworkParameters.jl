# Two leaves that keep their blocks in one `Tuple` field, and the gradient of a loss that reads that
# field. `TupBox` returns the `Tuple` as its storage and raises an `ArgumentError`, as
# `test/derivatives.jl` pins. `NTBox` returns the blocks as a `NamedTuple` and gets an all-zero
# gradient with no error: `KNOWN_ISSUES.md`, K3.
#
#     julia --project=. -e 'using TestEnv; TestEnv.activate(); include("scripts/tuple_field_storage.jl")'
using NeuralNetworkParameters, Zygote, Test
const NNP = NeuralNetworkParameters
struct TupBox{T} <: AbstractVector{T}
    b::Tuple{Vector{T}, Vector{T}}
end
Base.size(::TupBox) = (4,)
Base.getindex(s::TupBox, i::Int) = i <= 2 ? s.b[1][i] : s.b[2][i - 2]
NNP.freeparameters(s::TupBox) = s.b
NNP.rebuild(::TupBox, d) = TupBox(d)
struct NTBox{T} <: AbstractVector{T}
    b::Tuple{Vector{T}, Vector{T}}
end
Base.size(::NTBox) = (4,)
Base.getindex(s::NTBox, i::Int) = i <= 2 ? s.b[1][i] : s.b[2][i - 2]
NNP.freeparameters(s::NTBox) = (x = s.b[1], y = s.b[2])
NNP.rebuild(::NTBox, d) = NTBox((d.x, d.y))
for T in (Float32, Float64)
    ps = NetworkParameters((L = TupBox((T[1, 2], T[3, 4])),))
    r = try
        Zygote.gradient(q -> sum(q.L.b[1]) + 2sum(q.L.b[2]), ps)[1]
    catch e
        sprint(showerror, e)[1:min(end, 200)]
    end
    println("TupBox ", T, ": ", r)
    ps2 = NetworkParameters((L = NTBox((T[1, 2], T[3, 4])),))
    r2 = try
        g = Zygote.gradient(q -> sum(q.L.b[1]) + 2sum(q.L.b[2]), ps2)[1]
        first(flatten(g))
    catch e
        sprint(showerror, e)[1:min(end, 200)]
    end
    println("NTBox ", T, ": ", r2)
end
