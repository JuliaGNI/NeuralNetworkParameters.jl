```@meta
CurrentModule = NeuralNetworkParameters
```

# Converting between them

## The layout

```@docs
ParameterLayout
parameterlayout
parameterrange
flatlength
```

A layout is an ordinary value: build it once, keep it, compare it. That is the difference from
returning a closure that undoes the flattening — a closure cannot be stored in an optimizer cache,
compared for equality, or inferred through.

## The conversions

```@docs
flatten
flatten!
unflatten
unflatten!
```

## Derivatives

The flat form exists to be differentiated. Forward mode needs nothing special, because `unflatten` is
generic in the element type of its vector:

```jldoctest deriv
using NeuralNetworkParameters, ForwardDiff

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
v, layout = flatten(ps)

f(w) = sum(abs2, unflatten(layout, w).L1.W)
ForwardDiff.gradient(f, v)

# output

3-element Vector{Float64}:
 2.0
 4.0
 0.0
```

and the result is turned back into the shape of the network by unflattening it again:

```jldoctest deriv
unflatten(layout, ForwardDiff.gradient(f, v)).L1.W

# output

1×2 Matrix{Float64}:
 2.0  4.0
```

Reverse mode works through the same two functions: there are `ChainRulesCore` rules for both, and at a
fixed layout they are linear and mutually adjoint, so each rule is the other conversion. A position the
reverse pass says nothing about — a layer the loss never touched — comes back as a zero block rather
than an error.

For a Jacobian, `unflatten` also accepts a matrix and splits its *rows* by parameter block:

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
_, layout = flatten(ps)
J = reshape(collect(1.0:9.0), 3, 3)
unflatten(layout, J).L1.b

# output

1×3 Matrix{Float64}:
 3.0  6.0  9.0
```

## Cost, and why it is a copy

Flattening copies. On a network of 1.3 million `Float64` parameters the round trip costs about
0.3 ms, against roughly 0.95 ms for a single forward pass at batch size 32 — so a full conversion is
some 10 % of one forward pass, and less of a training step.

Sharing storage instead, so that the flat vector and the leaves were views of each other, would not
work even setting cost aside: `unflatten` has to be able to produce parameters of a *different element
type* from the ones the layout was built from, which is exactly what forward-mode differentiation
needs. A `Float64` buffer cannot be shared with a `Dual`-valued view.

Where repeated conversion does matter is an inner loop — an optimizer flattening twice per inner
product. That is what [`flatten!`](@ref) and [`unflatten!`](@ref) are for: given a layout and a
buffer, both are allocation-free.

```jldoctest
using NeuralNetworkParameters

ps = NetworkParameters((L1 = (W = [1.0 2.0], b = [3.0]),))
v, layout = flatten(ps)
buffer = similar(v)
flatten!(buffer, ps, layout)        # warm up, then this allocates nothing
buffer == v

# output

true
```
