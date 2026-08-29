"""
    GeometricBaseExt

`GeometricBase.L2norm` over a set of neural network parameters, from which the generic
`l2norm(x) = sqrt(L2norm(x))` follows.

The method lives here rather than in `GeometricBase` because **what it has to get right is this
package's, not that one's**: it walks the leaf protocol with [`foldparameters`](@ref) and it dispatches
on [`NetworkParameters`](@ref). A change to either breaks it, and this is the package where such a
change is made and tested. `GeometricBase` supports Julia 1.10 while this package requires 1.11, so a
test environment there could not resolve this package at all — a method a package cannot exercise is a
method it cannot keep correct.

Ownership does not decide the question, because it admits both: a method is type piracy only when the
function *and* every dispatched argument type belong elsewhere, and `NetworkParameters` is this
package's. A weak dependency costs nothing to a caller who does not load `GeometricBase`, whose own
sole dependency is `Unicode`.
"""
module GeometricBaseExt

using NeuralNetworkParameters: NetworkParameters, foldparameters

import GeometricBase.Utils: L2norm, l2norm

@doc raw"""
    L2norm(ps::NetworkParameters)

``\sum_i \mathrm{l2norm}(x_i)^2`` over the leaves of `ps`, at whatever depth they are.

`l2norm(ps)` follows from the generic `l2norm(x) = sqrt(L2norm(x))`, which is the quantity the callers
want: the blocks of a parameter set combine **in quadrature**. Summing their norms instead
overestimates the ℓ² norm by up to ``\sqrt{k}`` for ``k`` blocks, and thereby every stopping criterion
computed from it.

# Implementation

`L2norm` is the method and `l2norm` is what it calls on the leaves, which is deliberately not
symmetric. A leaf is entitled to its own notion of norm over its *free* parameters, and downstream
packages define exactly that: `GeometricOptimizers`' `l2norm(::AbstractLieAlgHorMatrix)` folds over the
lift's blocks, and its `l2norm(::VectorStorageMatrix)` over the stored vector. Both would be wrong if
this recursed through `L2norm` instead, because the generic `L2norm(::AbstractArray)` reads the dense
``n \times n`` interface and so counts a skew-symmetric block's entries twice.

`foldparameters` and not `map` + `sum`: a parameter set is a tree of layers and the quantities to
combine sit at its *leaves*, so `map` would hand `l2norm` a whole layer, for which there is no method.
The fold recurses into the branches, reaches a leaf at any depth, and allocates nothing.

`false` and not `zero(T)` as the initial value: there is no `T` in scope here, and `false` is the
strong zero that takes its type from whatever it is added to, so a one-block set adds it to that
block's value and stays a `T`.
"""
L2norm(ps::NetworkParameters) = foldparameters((acc, x) -> acc + abs2(l2norm(x)), false, ps)

end
