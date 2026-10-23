# The three primitives that walk across the children of one branch: a map, a left fold, and a map that
# carries a state from child to child. A foreach is the map, called for its effect.
#
# Each is a generated body that reads child `i` of the branch, and of every further set, with
# `getfield` at the literal index `i`. So every child is inferred at its own type, nothing is
# materialised on the way in, and a walk compiles to straight-line code at any width. `mapstorage`,
# the layout and the flattening are dispatch on the node type plus one call of a primitive;
# `mapparameters`, the in-place walks and the folds of `walk.jl` write the whole tree in one generated
# body, with the key and arity checks of this file. `docs/src/design.md` gives the reasons and the
# measurements.
#
# A branch is a `NamedTuple`, a `Tuple` or any struct, whose fields are its children. A further set
# that is `nothing` gives `nothing` at each child. Two named branches pair by key, and their keys must
# agree; a branch and a further set must have the same number of children. The generators check both,
# so the checks cost nothing at run time.
#
# A generator runs in the world age of its own definition, so every helper that a generator calls is
# defined above it. `test/integration/world_age.jl` loads the sources without a cache to check this.

# `f` with its first argument bound to `x`, called with every further argument: a walk binds its own
# function, or the flat vector, and a primitive passes it the child of each set. A type for `x` is
# kept as `Type{x}`, as `Base.Fix1` keeps it, so a walk over a type stays inferred.
struct _FixFirst{F, X} <: Function
    f::F
    x::X
    _FixFirst(f::F, x) where {F} = new{F, _type_of_bound(x)}(f, x)
end
@inline (g::_FixFirst)(args::Vararg{Any, N}) where {N} = g.f(g.x, args...)

_type_of_bound(x) = typeof(x)
_type_of_bound(::Type{T}) where {T} = Type{T}

# The number of children of the branch type `xs`, after a check that each further set has as many.
# `rest` holds the types of the further sets; a `Nothing` among them has no children to count.
function _children_arity(xs, rest::Tuple)
    n = fieldcount(xs)
    _check_arity(n, rest, 2)
    n
end

@inline _check_arity(::Int, ::Tuple{}, ::Int) = nothing
@inline function _check_arity(n::Int, rest::Tuple, j::Int)
    r = first(rest)
    r === Nothing || fieldcount(r) == n || _arity_error(n, j, fieldcount(r))
    _check_arity(n, Base.tail(rest), j + 1)
end

@noinline _arity_error(n::Int, j::Int, m::Int) = throw(ArgumentError(string(
    "parameter sets walked together must have the same number of children at every level; ",
    "got ", n, " and, in argument ", j, ", ", m, ".")))

# The keys of a branch type, or `nothing` for a branch that has none.
_branch_keys(::Type{<:NamedTuple{Keys}}) where {Keys} = Keys
_branch_keys(::Type) = nothing

@noinline _child_keys_error(ks, rk, j::Int) = throw(ArgumentError(string(
    "parameter sets have different keys: ", ks, " and, in argument ", j, ", ", rk)))

# A check that the further set `j`, of type `r`, has the keys `ks` of the branch beside it, where
# both have keys.
function _check_child_keys(ks, r::Type, j::Int)
    rk = _branch_keys(r)
    # `j + 1`: the message counts the parameter sets, of which the branch is the first
    ks === nothing || rk === nothing || rk === ks || _child_keys_error(ks, rk, j + 1)
    nothing
end

# Child `i` of the further set `j` of type `r`, as an expression: `nothing` for a set that is
# `nothing`, and `getfield` at a literal index otherwise, after a check of the keys.
function _child_expr(ks, r::Type, j::Int, i::Int)
    r === Nothing && return :nothing
    _check_child_keys(ks, r, j)
    :(getfield(rest[$j], $i))
end

# The arguments of the call at child `i`: the child of `xs`, then the child of each further set.
function _child_args(xs::Type, rest::Tuple, i::Int)
    ks = _branch_keys(xs)
    (:(getfield(xs, $i)), (_child_expr(ks, rest[j], j, i) for j in eachindex(rest))...)
end

# The `Tuple` of `f` over the children of `xs`, in lockstep with `rest`.
@generated function _unrolled_map(f, xs, rest...)
    n = _children_arity(xs, rest)
    :(($((Expr(:call, :f, _child_args(xs, rest, i)...) for i in 1:n)...),))
end

# `f` over the children of `xs`, in lockstep with `rest`, for its effect. Returns `nothing`. The walks
# that call it return `nothing` from `f`, so the map builds a `Tuple` of `nothing`s, which holds no
# data.
@inline function _unrolled_foreach(f, xs, rest::Vararg{Any, N}) where {N}
    _unrolled_map(f, xs, rest...)
    nothing
end

# The left fold `op(… op(op(init, x₁, r₁…), x₂, r₂…) …)` over the children of `xs`, in lockstep
# with `rest`. Left and not halved, because `op` need not be associative.
@generated function _unrolled_foldl(op, init, xs, rest...)
    n = _children_arity(xs, rest)
    expr = :init
    for i in 1:n
        expr = Expr(:call, :op, expr, _child_args(xs, rest, i)...)
    end
    expr
end

# A map that carries a state from child to child: `op(s, xᵢ, rᵢ…)` returns the result for child `i`
# and the state for the next child. Returns the `Tuple` of the results and the last state. `op` sees
# the state and not the results before it, so its argument types do not grow with the width.
@generated function _unrolled_mapaccum(op, init, xs, rest...)
    n = _children_arity(xs, rest)
    ys = [Symbol(:y, i) for i in 1:n]
    ss = [Symbol(:s, i) for i in 0:n]
    quote
        s0 = init
        $((:(($(ys[i]), $(ss[i + 1])) = $(Expr(:call, :op, ss[i], _child_args(xs, rest, i)...)))
        for i in 1:n)...)
        ($(ys...),), $(ss[n + 1])
    end
end
