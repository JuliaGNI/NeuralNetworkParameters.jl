using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "JET" include("quality/jet.jl")
    @safetestset "Parameters" include("parameters.jl")
    @safetestset "Leaf protocol" include("leaves.jl")
    @safetestset "Tree walks" include("walk.jl")
    @safetestset "Layout" include("layout.jl")
    @safetestset "Flatten / unflatten" include("flatten.jl")
    @safetestset "Flat parameters" include("flat_parameters.jl")
    @safetestset "Wide branches" include("integration/wide_branches.jl")
    @safetestset "World age" include("integration/world_age.jl")
    @safetestset "Derivatives" include("derivatives.jl")
    @safetestset "HDF5" include("io.jl")
    @safetestset "GeometricBase" include("norms.jl")
end
if "doctests" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
