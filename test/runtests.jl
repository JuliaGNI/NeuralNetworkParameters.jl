using SafeTestsets

@safetestset "Parameters                " begin
    include("parameters_tests.jl")
end
@safetestset "Leaf protocol             " begin
    include("leaves_tests.jl")
end
@safetestset "Tree walks                " begin
    include("walk_tests.jl")
end
@safetestset "Layout                    " begin
    include("layout_tests.jl")
end
@safetestset "Flatten / unflatten       " begin
    include("flatten_tests.jl")
end
@safetestset "Flat parameters           " begin
    include("flat_parameters_tests.jl")
end
@safetestset "Wide branches             " begin
    include("wide_branch_tests.jl")
end
@safetestset "Derivatives               " begin
    include("derivative_tests.jl")
end
@safetestset "HDF5                      " begin
    include("hdf5_tests.jl")
end
