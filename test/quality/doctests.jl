using Documenter
using NeuralNetworkParameters
using Test

# Documenter evaluates a page's `@meta` block in `Main`, and this file runs in a module of its own.
@eval Main import NeuralNetworkParameters

DocMeta.setdocmeta!(
    NeuralNetworkParameters, :DocTestSetup, :(using NeuralNetworkParameters);
    recursive = true)

doctest(NeuralNetworkParameters)
