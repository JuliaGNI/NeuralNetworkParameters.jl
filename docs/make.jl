using NeuralNetworkParameters
using Documenter

DocMeta.setdocmeta!(NeuralNetworkParameters, :DocTestSetup, :(using NeuralNetworkParameters); recursive = true)

makedocs(;
    modules = [NeuralNetworkParameters],
    authors = "Michael Kraus <michael.kraus@ipp.mpg.de> and contributors",
    sitename = "NeuralNetworkParameters.jl",
    format = Documenter.HTML(;
        canonical = "https://JuliaGNI.github.io/NeuralNetworkParameters.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "The two representations" => "representations.md",
        "Converting between them" => "conversions.md",
        "The leaf protocol" => "protocol.md",
        "Walking a parameter set" => "walks.md",
        "Reading and writing HDF5" => "hdf5.md",
        "Library" => "library.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaGNI/NeuralNetworkParameters.jl",
    devbranch = "main"
)
