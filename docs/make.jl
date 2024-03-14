using NeuralNetworkParameters
using Documenter

DocMeta.setdocmeta!(NeuralNetworkParameters, :DocTestSetup, :(using NeuralNetworkParameters); recursive=true)

makedocs(;
    modules=[NeuralNetworkParameters],
    authors="Michael Kraus <michael.kraus@ipp.mpg.de> and contributors",
    sitename="NeuralNetworkParameters.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaGNI.github.io/NeuralNetworkParameters.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/NeuralNetworkParameters.jl",
    devbranch="main",
)
