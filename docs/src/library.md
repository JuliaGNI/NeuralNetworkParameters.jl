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

## Classifying a node

```@docs
isparametertree
isterminal
```
