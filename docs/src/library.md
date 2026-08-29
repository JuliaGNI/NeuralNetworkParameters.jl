```@meta
CurrentModule = NeuralNetworkParameters
```

# Library

Everything the package exports is documented on the topical pages; this page collects the index, and
the internals that the other pages refer to but do not describe in place.

## Index

```@index
```

## Module

```@docs
NeuralNetworkParameters
```

## Layout internals

The five concrete layouts, one per thing a parameter set is made of. They are not exported, and are
constructed by [`parameterlayout`](@ref) rather than by hand, but a package storing a layout may want
to dispatch on them.

```@docs
LeafLayout
WrappedLayout
NestedLayout
TupleLayout
ParametersLayout
```

## Norms

`GeometricBase.L2norm` of a whole parameter set, from which that package's generic
`l2norm(x) = sqrt(L2norm(x))` follows. It is a method on a foreign generic and lives here rather than
in `GeometricBase` for the reason its docstring gives: the correctness of it is this package's leaf
protocol, which is not something `GeometricBase` can test.

```@docs
L2norm
```

## Classifying a node

```@docs
isparametertree
isterminal
```
