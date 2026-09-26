using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Parameters" include("parameters.jl")
    @safetestset "Leaf protocol" include("leaves.jl")
    @safetestset "Tree walks" include("walk.jl")
    @safetestset "Layout" include("layout.jl")
    @safetestset "Flatten / unflatten" include("flatten.jl")
    @safetestset "Flat parameters" include("flat_parameters.jl")
    @safetestset "Wide branches" include("wide_branches.jl")
    @safetestset "World age" include("world_age.jl")
    @safetestset "Derivatives" include("derivatives.jl")
    @safetestset "HDF5" include("io.jl")
    @safetestset "GeometricBase" include("norms.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
