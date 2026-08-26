# Changelog

Notable changes to `NeuralNetworkParameters` are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.5] — 2026-08-26

### Fixed

- **`parameter_eltype` took `values` of a branch, so it allocated on a wide one — and every walk here
  reaches it.** `flatten(ps)` is `flatten(parameter_eltype(ps), ps)`, which is the documented spelling
  for "the parameters' own element type", so the convenient form was the one that paid; and every
  `NetworkParameters` runs its constructor through the promotion, so *building* a wide set paid it too.
  The promotion now reads the branch in place with `getfield` at literal indices, as the other nine
  across-children walks have since 0.2.2. Issue [#22], found in
  `GeometricOptimizers` [#70](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/70), where
  0.2.4's zipped `foldparameters` put `parameter_eltype` on a hot path for the first time.

  `@allocated` from inside a function with the call warmed, `Float32` 2×2 leaves, Julia 1.13.0-rc3 on
  an Apple M4 Max:

  | branch | before | after |
  |---|---|---|
  | 32 children | 0 B | 0 B |
  | 33 children | 544 B | **0 B** |
  | 48 children | 800 B | **0 B** |
  | 369 children | 6 144 B | **0 B** |
  | `NetworkParameters` of the 369 | 6 144 B | **0 B** |
  | `flatten(ps)` / `flatten(T, ps)`, 369 | 12 352 B / 6 208 B | **6 208 B / 6 208 B** |
  | `rrule(flatten, ps)` / `rrule(flatten, T, ps)`, 369 | 12 352 B / 6 208 B | **6 208 B / 6 208 B** |

  The reverse pass paid it on the same terms, because `rrule(flatten, ps)` derives the element type the
  same way — so a gradient taken through the flat form of a wide bare set carried the 6 144 B as well.

  **A branch of branches paid it as well, and the shape that shows it is not the one that looks worst.**
  What decides the cost is the width of the *outer* branch, because `Base` unrolls a tuple up to 32
  fields and drops to its `Any32` fallback past it. So a 16 × 24 set was already free while a 48 × 2
  one — a child per layer, which is the shape a network of many layers has — cost 3 168 B and a
  369 × 2 one 23 840 B. All are 0 B now.

  These are D13 and D17's figures exactly, and this is their defect: `values(·)` on a branch
  materialises a temporary tuple, which is free while the branch stays in registers and is not beyond
  that. 0.2.2 removed it from the two forward walks and 0.2.3 from the three that take a second set.
  This was the tenth, and it was the one none of that work measured — `scripts/wide_branch_cost.jl` had
  no column for the element type until now, which is why the figure came from a consumer rather than
  from here.

  **The issue proposed a different fix, and it was not taken.** It read the cost as the runtime
  `promote_type` chain the `@generated` body emits and proposed settling the element type in the type
  domain instead. Measured, that chain is not the cost — `_promote_eltypes` handed a tuple *already
  built* allocates 0 B at 369 children — and it is not a compile cost either: `parameter_eltype` on a
  369-child branch takes 0.031 s cold with the fix against 0.036 s without. The shortcut would also
  have changed an answer. Settling an `AbstractArray` child with `eltype` reads the interface a leaf
  *presents*, where `parameter_eltype` follows `freeparameters` behind a `s === x` test and reports the
  storage a leaf *has*; a leaf that is an `AbstractMatrix{Float64}` over a `Vector{Float32}` flattens
  into a `Vector{Float32}` and would have flattened into a vector twice as wide. Nothing pinned that
  case, as the issue noted; `test/leaves_tests.jl` pins it now, and the docstring states it as a
  guarantee rather than leaving it to be read off the implementation.

  What the fix buys downstream is the argument the issue makes for it. With the call free,
  `GeometricOptimizers`' `_dot` is one widened signature rather than two — one taking the element type
  off a `ParameterContainer{T}` to keep the wide flat shape free and one using `parameter_eltype` for
  the nested shapes that bind no `T` — and `zero(parameter_eltype(x))` becomes usable in the folds
  behind `l2norm` and `solution_scale`, which is the order-dependence in their accumulator that they
  currently document instead.

- **`test/wide_branch_tests.jl` asserts the element type at both widths**, in all four shapes — bare,
  positional, wrapped and a branch of branches — together with the constructor that runs it, at exactly
  zero. `flatten(ps)` against `flatten(T, ps)`, and the same pair on the reverse pass, are asserted as a
  *bound* rather than an equality: `@allocated` reports the process-wide counter over its window and not
  the call's own, so two readings of two multi-kilobyte calls differ by a few bytes on a loaded machine
  — Windows CI reads 6 039 against 6 055 for that pair. The bound is `8k`, half the per-child cost of a
  promotion taken over `values`, which separates the jitter from the defect by two orders of magnitude
  at both widths. Inference is asserted too: it was never what allocated, and a fix that lost the
  constant would be a different regression.

- **`scripts/wide_branch_cost.jl` sweeps the element type**, in the allocations column. Read the bare
  column: the wrapped one is 0 by construction, since `parameter_eltype(::NetworkParameters{T})` is `T`
  off the type, but its constructor promotes over the bare branch — so the bare figure is what building
  the wrapped set costs. D22 is the same lesson one walk over.

## [0.2.4] — 2026-08-26

**The fold takes several sets in lockstep, as every other walk here already did.**
`foldparameters(op, init, ps)` walked one tree, so a consumer wanting `sum(aᵢ * bᵢ)` or
`sum(f(aᵢ)^2)` over a parameter set had to write the recursion itself. `GeometricOptimizers` wrote
three — `l2norm`, `solution_scale` and `_dot`, as `Base.tail` chains — and they cost it 26 to 71 s to
compile at 369 children on Julia 1.12 and 1.13, against 0.65 to 1.47 s on 1.11.9. None of that is this
package's code, and the figures below are not a regression being fixed: they are the argument for the
walk living here rather than being written again by every consumer. Issue [#19].

### Added

- **`foldparameters(op, init, ps, rest...)`.** The trees are walked in lockstep and `op` receives one
  leaf from each: `op(acc, a_leaf, b_leaf, …)`, at any arity. The children of a keyed branch are
  paired **by key** and the keys and the widths are checked in the generator, where they cost nothing
  — the guarantee the in-place walks gained in 0.2.2, and stricter than the `_check_keys` that
  `mapparameters` pays for at run time. Allocation-free at every width, depth and arity.

  A set that is `nothing` **raises**, where `foreachparameters` skips one: a fold reduces every leaf
  it is given, so a set left out would make the answer a partial sum without saying so. A `nothing`
  *leaf* is a different thing and still reaches `op`, so a caller who wants to decide what a missing
  leaf contributes still can — including where the leaf keeps its storage behind an interface, which
  is the **Fixed** entry below.

- **`foldstorage(op, init, ps, rest...)`**, which is the same walk over the `freeparameters` of each
  leaf, and the level `flatten` writes. The inner product of two sets with a symmetric leaf is over
  the n(n+1)/2 numbers it stores; a dense reading would count every off-diagonal entry twice, and for
  a skew-symmetric or triangular leaf there is no dense reading at all. It completes the pairs the
  rest of the file has: `mapparameters`/`mapstorage`, `mapparameters!`/`mapstorage!`, and now
  `foldparameters`/`foldstorage`.

- **Both are one `@generated` body**, `_fold_zip`, at literal indices — the treatment the other eight
  across-children walks got in 0.2.2, now including the arity that did not exist before. `kind`
  selects which of the two `_fold_step` continues with, exactly as it does for the `foreach` family.

### Fixed

- **A `nothing` in place of a leaf reached the storage walks as the leaf protocol's own error.**
  `foldstorage` and `mapstorage` have to descend into a leaf that keeps its storage behind an
  interface, and they descended by asking every further set's leaf for its `freeparameters` — so a
  hole was asked where its storage lives, and the protocol answered the only way it can:

  ```
  ArgumentError: no method of `freeparameters` for `Nothing`.
      NeuralNetworkParameters.freeparameters(::Nothing) = ...
  ```

  which is the one instruction that is not the fix. The hole survives the descent now. A `nothing`
  against a leaf whose storage is a single array — a `SymmetricMatrix`, a manifold element, a
  horizontal lift — reaches `op` or `f` paired with that array, which is what the docstrings say and
  what `mapparameters` and `foldparameters` already did with the whole leaf. A `nothing` against a
  **multi-block** leaf has nothing to pair one-to-one with and raises each walk's own error instead:
  the partial-sum error for the fold, and for `mapstorage` the one that names the three walks which
  skip a hole. `mapstorage` carried this since the storage walks were written; it is one line from
  the fold's and has the same cause, so it is fixed here.

- **`scripts/wide_branch_cost.jl` did not sweep the fold.** It timed `parameterlayout`,
  `map(zero, ·)`, `mapparameters`, `flatten`, `unflatten` and the in-place allocations, at both shapes
  and every width, and the fold at neither — which was the one walk whose downstream analogue had just
  produced a two-orders-of-magnitude cliff. `test/wide_branch_tests.jl` folds 369 children but asserts
  only the value, so what stood in for a figure was the suite's total wall clock. Three columns now:
  `fold` and `foldzip` are `foldparameters` at arity one and two, and `tailfold` is the `Base.tail`
  recursion over `values` a consumer writes without a zipped fold.

  First call, one process per row, bare `NamedTuple` of 369 leaves. The wrapped column reads the same
  to the spread of a single cold measurement — within 0.06 s on the fold columns, and 28.20 s against
  28.55 s and 37.74 s against 37.86 s on the control:

  | Julia | `fold` | `foldzip` | `tailfold` |
  |---|---|---|---|
  | 1.11.9 | 2.11 s | 0.35 s | 1.94 s |
  | 1.12.7 | 0.80 s | 1.08 s | **28.55 s** |
  | 1.13.0-rc3 | 0.47 s | 0.63 s | **37.86 s** |

  **On 1.11 the control is the cheaper of the two**, and that is worth stating plainly: 1.94 s against
  2.11 s for the written-out fold at arity one. The case for writing it out is not that it wins at the
  compat floor; it is that it costs about the same everywhere and *falls* with each version, while the
  chain multiplies by fifteen and then by twenty. `tailfold` is also measured last in its row, so
  everything it shares with the columns above it is already compiled — the gap is if anything
  understated.

  The two fold columns are the whole cost of the arity: 2.46 s together on 1.11, and one `tailfold` is
  one shape of one fold, of which `GeometricOptimizers` has three.

### Unchanged

- **Nothing in `src/` changed on account of the `@inline` question.** `GeometricOptimizers`' three
  folds carry no `@inline` at all and hit the cliff at the same width, which confirms from the other
  end what 0.2.2 recorded — that dropping the `@inline` from the walks here did not help either. What
  it qualifies is the *version*: 0.2.2's account of D12 was written on walks that were written out for
  every Julia, so it never showed that an un-`@inline`d `Base.tail` fold is 0.65 s on 1.11.9 and 26 to
  35 s on 1.12.7 and 1.13.0-rc3. `PLAN.md`'s D12 entry now says so, since it is where someone will
  look before writing the next `Base.tail` walk in this ecosystem.

- The `foldparameters` docstring carries one thing a consumer has to do, because this package cannot
  do it for them: **a function that hands `op` on has to annotate it `::F where {F}`.** Julia does not
  specialise on a function argument it never sees called, so without it `op` arrives boxed and every
  leaf costs a dynamic dispatch — 3 088 bytes a call on a 369-leaf set at arity one, 6 160 at arity
  two, against zero with it, on 1.11.9 and 1.13.0-rc3 alike. Annotating the methods *here* changes
  none of those figures, which is why they are not annotated; the boxing is the caller's. A closure
  that captures a function needs nothing, a closure being its own type — so
  `foldparameters((acc, x) -> acc + abs2(f(x)), 0, ps)` is how a function of each leaf is folded, and
  no second function argument is needed. `test/wide_branch_tests.jl` pins **both** halves of that —
  the annotated caller at zero and the otherwise identical unannotated one above zero — so the
  instruction this package gives its consumers cannot go stale behind a green suite.

### Noticed and not addressed

- **`unflatten` at 369 children costs 4.35 s on Julia 1.11.9, 13.85 s on 1.12.7 and 18.32 s on
  1.13.0-rc3**, in both shapes. It is the one column of the sweep that grows with the Julia version
  rather than shrinking, and it is not this release's doing: 0.2.3's own script, run on the same
  machine at the same width, reads 4.49 s, 13.75 s and 18.30 s. That comparison is also what says the
  three new columns do not perturb the row they were added to — `layout` reads 1.06/1.13/1.06 s
  against 1.03/1.11/1.05, and `flatten` 3.73/2.20/2.05 against 3.70/2.22/2.06. Nothing here is changed
  for it; it is recorded so that the next person to read the table does not have to rediscover it.

## [0.2.3] — 2026-08-26

**The cost of a wrapped parameter set was one unused type parameter.** 0.2.2 made a wide branch
compilable and left one shape standing that was not: `parameterlayout` on a 369-leaf set cost 1.35 s as
a bare `NamedTuple` and 13.40 s inside a `NetworkParameters`, and at 768 leaves 2.77 s against 87.77 s.
Those four are the table below, which is where the entry's figures come from unless it names another
harness or another run. The gap was read as a cost of the composition in
`_layout(::NetworkParameters, ::Int)`, and issue [#16] read it as a cost of *nesting*. It was neither,
and the note in `src/layout.jl` that argued the first of those is corrected here rather than released —
it was written after the 0.2.2 bump and this is its first release.

### Removed

- **`LeafLayout` no longer carries a `prototype`.** The field was written at construction and read
  nowhere: a `LeafLayout` is by construction the case where `freeparameters(x) === x`, so `rebuild` on
  it is the identity and is never called. The two prototypes the package does read are both on
  `WrappedLayout`, where unflattening genuinely rebuilds against one. The struct is now
  `LeafLayout{N}` over `range` and `size`, which is what every method dispatching on it already read
  and what `==` already compared.

  The layouts are not exported and are built by `parameterlayout` rather than by hand, which is why
  this is a patch. A caller who did construct one directly has to drop the third argument:
  `LeafLayout(range, size, prototype)` becomes `LeafLayout(range, size)`. Nothing in
  `AbstractNeuralNetworks`, `SymbolicNeuralNetworks`, `GeometricOptimizers` or
  `GeometricMachineLearning` does — they dispatch on the bare type and read `size`.

  A terminal leaf's layout also no longer holds a live reference to the array, so keeping one no longer
  keeps that leaf alive. `Base.summarysize` of the layout of a one-leaf set is 48 bytes whether the
  leaf is 2 × 2 or 1000 × 1000, and `test/layout_tests.jl` pins it. A *structured* leaf is unchanged
  and stays that way: its `WrappedLayout` holds the prototype `rebuild` unflattens against, so a set
  with a `SymmetricMatrix` or a manifold element in it still has those leaves reachable from its
  layout.

### Fixed

- **`parameterlayout` on a `NetworkParameters` costs what it costs on the `NamedTuple` inside it.** The
  type parameter above was the leaf's concrete array type, so it went into the layout type of every
  branch above that leaf — 1849 nodes in the type tree of a 369-leaf wrapped layout, against 742
  without it. `_layout(::NetworkParameters, ::Int)` is where a caller paid for that, because it infers
  through the child walk's whole return type to reach `ParametersLayout(inner)`, and inference on such a
  type grows faster than the type does: 2.5 times the type was 13 times the time.

  First call, Julia 1.11, one process per row, `scripts/leaf_layout_cost.jl`:

  | `parameterlayout` on | 0.2.2 | 0.2.3 | type nodes |
  |---|---|---|---|
  | 369 leaves, bare `NamedTuple` | 1.35 s | 1.04 s | 1848 → 741 |
  | 369 leaves, in a `NetworkParameters` | 13.40 s | **1.05 s** | 1849 → 742 |
  | 768 leaves, bare | 2.77 s | 3.07 s | 3843 → 1539 |
  | 768 leaves, in a `NetworkParameters` | 87.77 s | **3.01 s** | 3844 → 1540 |
  | 16 × 24 nested, bare | 1.46 s | 0.85 s | 1971 → 819 |
  | 16 × 24 nested, in a `NetworkParameters` | 14.00 s | **0.84 s** | 1972 → 820 |

  **The bare column is where to look, and it does not improve** — at 768 leaves it is a shade slower,
  2.77 s against 3.07 s, and 2.76 s against 2.99 s on a second run, which is the run-to-run spread of a
  single cold measurement rather than a regression. The type shrinks by 2.5× in that column too, and
  buys nothing there. What was expensive was never the walk; it was what a caller had to infer through
  to wrap the walk's result, and only the wrapped column has such a caller.

  Neither the width nor the nesting drives it either: 768 flat leaves now cost less than half what 369
  wrapped ones used to, and 384 leaves in 16 branches cost slightly less than 369 in one.

  `flatten` improves with it, 14.38 s to 3.61 s on the bare 369 set, and the whole path —
  `parameterlayout`, then `flatten`, then `unflatten` — goes from 21.90 s to **8.64 s** on the wrapped
  one. Those and the allocation figures are `scripts/wide_branch_cost.jl`, whose 24 rows all still read
  zero; 0.2.1's guarantees are the reason that had to be checked rather than assumed.

  Julia 1.11, 1.12 and 1.13 now agree at 1.05 s, 1.13 s and 1.03 s on the wrapped 369 set. The compat
  floor used to be the version that behaved differently, and the note in `src/layout.jl` that called
  that "a characteristic of the compat floor and not of the walk" was wrong: it was a characteristic of
  the layout type, which 1.12 and 1.13 were better at absorbing.

  Issue [#15] filed this as hygiene and said explicitly that it was not the cause of the nested-layout
  compile cost, on the evidence that alternating `Matrix`/`Vector` leaves cost 11.32 s against 13.9 s
  homogeneous. That measurement is sound and its conclusion — that leaf *diversity* is not the driver —
  still holds. It compared two sets under the old layout type, though, and never compared the layout
  type with and without the parameter, which is where the 13× was ([#18]).

## [0.2.2] — 2026-08-25

**A branch with many children is usable.** 0.2.1 finished making the walks type stable and allocation
free; this release makes them *compilable* on a branch wider than a couple of dozen children, which
they were not.

### Changed

- **Julia 1.11 is now the minimum**, up from 1.10. The walks below are `@generated` bodies, and Julia
  1.10 cannot inline one: neither `Expr(:meta, :inline)` in the generated body, nor `@inline` on the
  declaration, nor `@inline` at the call site has any effect there — the call stays out of line. The
  bodies are allocation-free on 1.10 too; what `flatten!` and `unflatten!` need from the inlining is
  that it lets the temporaries at a branch boundary be elided. Without it, on 1.10, against 0.2.1:

  | set | `flatten!` | `unflatten!` | `unflatten` |
  |---|---|---|---|
  | one with structured leaves | 0 → 192 | 0 → 608 | 576 → 1616 |
  | a two-level `NamedTuple` | 0 → 0 | 0 → 176 | 960 → 1408 |

  That is the guarantee 0.2.1 exists to provide, so on 1.10 the two cannot both be had. Julia 1.11 and
  later elide those temporaries without inlining, and 1.11, 1.12 and 1.13 were each checked at zero on
  a nested set, a set with structured leaves, a wide flat branch and a wide branch of branches.

### Fixed

- **Every walk across the children of one branch cost one specialisation per child.** The walk *down*
  the tree is ordinary dispatch and was never the problem. The walk *across* was an `@inline`d
  `Base.tail` chain, and `Base.tail` yields a new tuple type at every level — so a branch of `k`
  children produced `k` specialisations whose argument types were each `O(k)` long, and inference on
  that grew as `k³`.

  Nine walks were written this way. Eight are written out instead, as a `@generated` flat body of `k`
  statements reading `getfield(xs, 1) … getfield(xs, k)` at *literal* indices: one specialisation per
  branch shape rather than `k`, no new tuple types at all, every index inferred at a constant, and the
  same straight-line code the chain used to inline down to. They are `_flatten_children!`,
  `_unflatten_children`, `_unflatten_children!` (`src/flatten.jl`), `_foreach_zip`, `_fold_children`,
  `_anynothing` (`src/walk.jl`), `_promote_eltypes` (`src/leaves.jl`) and the two branch cases of
  `_layout` (`src/layout.jl`).

  The ninth, `_map_zip`, is written out **to 32 children and hands wider branches to `Base.map`**, and
  the asymmetry is the point rather than an exception. Writing a body out costs compilation per branch
  shape; `map` costs type stability past 32 fields, where `Base` drops to its `Any32` loop. Which of
  those is the bad trade depends entirely on whether the walk runs in an inner loop. The eight above
  do — `flatten!` and `unflatten!` once per optimizer iteration, `_unflatten_children` once per
  objective evaluation through the closure a `Gradient` is built from — and there a dynamic dispatch
  per child would be paid on every call. `_map_zip` is what `mapparameters` and `mapstorage` are, and a
  consumer calls those once per cache, once per `changebackend`, once per `map_to_cpu`; there the same
  dispatch is paid once, and the object handed back is concretely typed either way, so everything
  downstream of it specialises normally.

  What that is worth: `GeometricOptimizers` reaches six such primitives while building one
  `OptimizerCache`, and on the 369-entry flat set that took **1.57 s on its 0.5.0 to 71.16 s** with
  `_map_zip` written out at every width, against **2.41 s** with the split. See the note on `_map_zip`
  in `src/walk.jl`, which carries the measurement.

  First-call time, which is compilation plus a negligible run, on a flat `NamedTuple` of `k` leaves.
  **Every figure below is cold**: `scripts/wide_branch_cost.jl` runs each width in a process of its own,
  and the 0.2.1 column comes from running that same script against a worktree of `v0.2.1`, so the two
  are comparable rather than merely recorded. Julia 1.11.9, Apple M4 Max.

  Within one width the rows are *cumulative* — `parameterlayout` runs first and `flatten` builds a
  layout too, so each row is what it adds to the ones above it. The total is what a consumer pays to
  compile the whole path, and is the column to read:

  | children | `parameterlayout` | `mapparameters` | `flatten` | `unflatten` | **total** |
  |---|---|---|---|---|---|
  | 16 | 0.05 → 0.07 | 0.05 → 0.05 | 0.09 → 0.09 | 0.15 → 0.12 | 0.40 → **0.40** |
  | 32 | 0.12 → 0.08 | 0.13 → 0.12 | 0.25 → 0.18 | 0.46 → 0.26 | 0.97 → **0.65** |
  | 48 | 0.19 → 0.15 | 0.22 → 0.04 | 0.37 → 0.30 | 0.66 → 0.39 | 1.45 → **0.89** |
  | 64 | 0.29 → 0.21 | 0.43 → 0.04 | 0.62 → 0.43 | 1.12 → 0.53 | 2.47 → **1.22** |
  | 128 | 1.85 → 0.65 | 1.43 → 0.05 | 2.08 → 2.25 | 3.63 → 1.14 | 9.00 → **4.10** |
  | 369 | 12.32 → 1.35 | 8.56 → 0.09 | 12.82 → 14.39 | 22.32 → 4.38 | 56.03 → **20.22** |

  `map(zero, ·)` is 0.01 s at every width on both and is omitted; it is in the harness as the floor —
  what `Base` costs on the same branch, and what a consumer that wrote its own walk over `map` to get
  around this was paying instead.

  **Every set above is a bare `NamedTuple`, and the `parameterlayout` column reads low because of it.**
  A consumer holds a `NetworkParameters`, which reaches the walk through one more method, and on
  Julia 1.11 that method is what the column costs: 13.44 s at 369 children against the 1.35 s above,
  and 88.69 s at 768 against 2.73 s. It is the *entry point* and not the total — the whole path,
  `parameterlayout` then `flatten` then `unflatten`, is 21.66 s bare and 22.41 s wrapped at 369 — and
  Julia 1.12 and 1.13 do not distinguish the two shapes at all (1.78 s and 1.67 s wrapped at 369). The
  sweep runs both shapes now; see D20 in `PLAN.md` and the note above
  `_layout(::NetworkParameters, ::Int)` in `src/layout.jl`.

  The `flatten` column is the one that goes the wrong way at the two largest widths, and it is a
  re-attribution rather than a loss: `parameterlayout` no longer absorbs the shared compilation, so
  `flatten` pays for it instead. Measured alone in its own process at 369 children, `flatten` is
  13.78 s on 0.2.1 and 16.29 s here — a real 18 % — against 12.32 s and 1.36 s for `parameterlayout`.
  That trade is taken deliberately; see the leaf-dispatch entry below for why the instability it
  removes is worth more than the 18 %.

  **An earlier draft of this entry quoted different figures**, and they are worth naming because of how
  they were wrong rather than by how much. It had `flatten` at 17.57 s on 0.2.1 at 128 children and
  "not run to completion" at 369 — where cold it is 2.08 s and 12.82 s, and completes. The *conclusion*
  was right, and not one of the numbers supporting it was: the harness swept the widths in a single
  process, so every row after the first was measuring what its predecessors had already compiled. The
  `mapparameters` column suffered worst, reading as `0.00 s` what costs 0.09 s cold — and 1.51 s cold
  before the `_map_zip` split above. `GeometricOptimizers` closed an issue of its own on that `0.00 s`.

  The script forks per width now, so there is no ordering left to get wrong and no `--cold-map` flag to
  remember: a figure is cold or it is not printed. That is the general lesson and it is cheap to state —
  **a measurement you have to ask for in a special way is one the default run is getting wrong.**

  Re-measured with the corrected harness — see the entry below on `first_call` — on Julia 1.13.0-rc2,
  Apple M4 Max, one process per width, seconds, `0.2.1 → 0.2.2`:

  | children | `parameterlayout` | `mapparameters` | `flatten` | `unflatten` | **total** |
  |---|---|---|---|---|---|
  | 16 | 0.04 → 0.10 | 0.08 → 0.15 | 0.09 → 0.11 | 0.12 → 0.10 | 0.33 → **0.46** |
  | 32 | 0.08 → 0.13 | 0.22 → 0.26 | 0.22 → 0.16 | 0.35 → 0.25 | 0.87 → **0.80** |
  | 48 | 0.58 → 0.20 | 1.09 → 0.10 | 1.31 → 0.23 | 1.26 → 0.46 | 4.24 → **0.99** |
  | 64 | 1.32 → 0.22 | 2.63 → 0.10 | 3.15 → 0.32 | 2.95 → 0.72 | 10.05 → **1.36** |
  | 128 | 7.75 → 0.51 | 24.49 → 0.12 | 27.84 → 0.70 | 25.82 → 2.01 | 85.90 → **3.34** |
  | 369 | — → 1.73 | — → 0.17 | — → 2.48 | — → 19.27 | — → **24.65** |

  The 369-child row has no 0.2.1 column: `parameterlayout` alone was still compiling after 20 minutes
  in a process of its own, which is the "did not finish" of the original report, now with a lower bound
  on it. Every other row is a completed measurement of both trees.

  The shape is the same as the 1.11.9 table above and the margin is wider, because the figures the old
  harness printed were missing the inference that is most of the cost. 16 children is the one width
  where 0.2.2 is behind, by 0.13 s, which is a written-out body being compiled where a chain of four
  was not — the trade the `_map_zip` note describes, at the width where it does not pay.

  **The harness had a second way of not measuring the thing, and forking per width did not touch it.**
  `first_call` timed a direct `f(args...)`, and compiling `first_call` itself infers through the call in
  its body — so the inference the figure is *for* was spent while `first_call` was being compiled, before
  its own `t = time()` ran. On Julia 1.13 that reads **0.00 s for every width in every column**;
  `parameterlayout` on a 369-child branch, measured through an opaque call, is 1.61 s and `flatten` is
  3.95 s. The call goes through `Base.invokelatest` now, so there is nothing left for the caller's
  compilation to do first. Same lesson as the fork, from the other end: the harness has to be arranged so
  that the cost cannot have been paid somewhere the clock is not looking.

- **`_layout` on a leaf inferred as a three-way union, and a branch of 369 of them was what
  `parameterlayout` cost.** The leaf method asked `freeparameters(x) === x` in an `if`, and that is a
  runtime pointer comparison inference cannot fold — so `_layout(::Matrix{Float32}, ::Int)` came back
  as `Union{Tuple{LeafLayout{2,Matrix{Float32}},Int}, Tuple{WrappedLayout{…},Int}, Tuple{WrappedLayout{…} where …,Int}}`.
  One union per child is survivable; 369 of them is 11.7 s of inference.

  Terminality is decided by **dispatch** now — `_layout_storage(x::T, ::T, offset)` against the general
  method — and the branch construction is fused into the same `@generated` body that lays the children
  out, rather than returning a `k`-element tuple for a caller to infer through. `parameterlayout` on
  that set is **1.36 s**, and 25.2 s → 15.35 s on a 16 × 24 nested container.

  Three other explanations were measured and ruled out first, which is why they are recorded: a
  `@noinline` barrier on the child walk (11.78 s, no change), supplying `NestedLayout`'s type
  parameters rather than letting them be solved (no change), and computing the child lengths first so
  the children could be laid out *independently* — **worse**, 15.4 s, because it adds two more `k`-long
  bodies. The serial offset dependency was never the cost: an "every child at the same offset" control
  runs at 1.17 s against the real walk's 1.27 s.

  One semantic edge changes with it. A leaf whose `freeparameters` returns a *distinct object of its
  own type* is now treated as terminal where the `===` would have wrapped it. Nothing is lost: such a
  type never worked, since `_layout` would descend into the storage, ask it for its own
  `freeparameters`, and recurse without end unless it eventually changed type. [`isterminal`](@ref)
  still asks the identity question and is still right to — it is a predicate a caller evaluates on a
  value it holds, not a dispatch anything has to infer through.

- **`flatten!` and `unflatten!` allocated on a branch of more than about forty children**, against the
  guarantee in their own docstrings. Two temporary tuples were materialised per branch — `values(ps)`
  and `values(l.children)` — which is free while a branch is narrow enough to keep in registers and is
  not beyond that. Measured on the same flat sets on Julia 1.13, bytes per call from inside a
  function:

  | children | 16 | 32 | 48 | 64 | 128 | 369 |
  |---|---|---|---|---|---|---|
  | 0.2.1 | 0 | 0 | 70 288 | 160 928 | 770 624 | 6 608 576 |
  | 0.2.2 | 0 | 0 | **0** | **0** | **0** | **0** |

  Re-measured cold with the rest, which moved the 0.2.1 row: an earlier draft had 81 488 at 48 children
  and 187 808 at 64, and "not reached" beyond, where the per-width harness reaches every one of them.
  Same shape, same cliff, different numbers.

  The walks index the branch in place with `getfield` now and take no `values` at all, so nothing is
  materialised on the way in. This was found while fixing the entry above and is a separate defect:
  narrowing the branch made it disappear, which is why the docstrings' claim had held up in testing.

- **`mapparameters!`, `mapstorage!` and `foreachparameters` had the same defect, on the arm that takes a
  second set** — and they are the walks that run every optimizer iteration, where `flatten!` runs once
  or twice. The entry above removed the temporary from the branch being walked; each *further* argument
  was still turned into a `values(...)` tuple per branch by `_values_for`, and a skipped set into a
  fresh `map(_ -> nothing, keys)` tuple on top of that. Bytes per call of `mapparameters!(f, dest, src)`
  on Julia 1.13:

  | children | 16 | 32 | 48 | 64 | 128 | 369 |
  |---|---|---|---|---|---|---|
  | 0.2.1 | 0 | 0 | 23 488 | 54 848 | 265 408 | — |
  | first written here | 0 | 0 | 800 | 1 088 | 2 176 | 6 144 |
  | **now** | 0 | 0 | **0** | **0** | **0** | **0** |

  The middle row is the point rather than the top one: removing the temporary from the branch *being*
  walked took 23 488 bytes at 48 children down to 800, which looks like the defect closing and is the
  cliff simply moved. A branch of branches, where every child is a branch to take apart rather than an
  array, cost 3 168 bytes at 48 and 23 840 at 369 on the middle row.

  Same cliff at the same width, for the same reason. `_foreach_zip` reads every one of its arguments
  with `getfield` at a literal index now; `_values_for` and `_tuple_for` have no callers and are gone.
  Zero at 4, 16, 32, 48, 64, 128 and 369 children, flat and nested, at arity one, two and three, and on
  the `nothing`-skip path.

  Not caught by the wide-branch tests as first written, which did call `mapparameters!` at 369 children
  and asserted only its result. Measuring it needs a **zero-argument closure**: `@allocated f(a, b)`
  lowers to `Base.allocated(f, a, b)`, and for a `Vararg` method — which all three of these are — the
  splat inside `Base.allocated` allocates on its own account, inventing bytes the walk does not spend
  and hiding the ones it does. `test/wide_branch_tests.jl` says so beside the harness.

- **The in-place walks paired the children of two branches by *position*.** `_values_for(x, ks)` ignored
  its `ks`, so `foreachparameters(copyto!, (a = …, b = …), (b = …, a = …))` wrote each leaf from the
  wrong key and said nothing, where `mapparameters` has always raised. Two same-shaped sets whose keys
  are ordered differently — a gradient tree built by hand, or merged from parts — silently crossed over.

  The generated body checks the keys of a named argument against the branch's own, in the *generator*,
  so the guard `mapparameters` pays `_check_keys` for at run time is free here. **This is a behaviour
  change**: positional pairing of differently-keyed sets now raises `ArgumentError`. A `Tuple` branch
  stays positional, because the blocks of a multi-block leaf have no keys to agree on.

- **`mapparameters` and `mapstorage` re-selected every child they were just handed.** Past 32 children
  `_map_zip` hands back a `NamedTuple` already keyed with `keys(ps)`, and both then ran it through
  `NamedTuple{keys(ps)}(...)` again — a second `k`-wide generated selection and a copy, 6 272 bytes of
  the 57 120 a call costs at 369 children. `_rewrap_children` keys the written-out arm's `Tuple` and
  leaves the `map` arm's `NamedTuple` alone.

- **The reverse pass indexed leaf cotangents one element at a time.** `_add_leaf_cotangent!` accumulated
  with `enumerate(Δ)` and a scalar `Δv[o + i] += x`, while `flatten`'s docstring makes a point of the
  forward path never indexing an element, "so it runs at memory bandwidth and works unchanged for GPU
  arrays". It broadcasts over the leaf's stretch of the flat vector now, at the same 848 and 6 208 bytes
  at 48 and 369 children, and reaches a device array on the same terms the forward path does.

- **The arity check allocated on `mapparameters`' own path.** `_children_arity` took an untyped `rest...`
  and walked it with `enumerate`; every caller but one reaches it from a generator, where that is free,
  and `_map_zip` reaches it at run time, where it cost 48 bytes a call. It takes the further sets as one
  tuple now and checks them with a chain over the *arity* — one or two, never a branch width.

- `_tuple_for(::Nothing, n::Int)` filled a positional branch with `ntuple(_ -> nothing, n)`, which
  infers as `Tuple{Vararg{Nothing}}` past ten blocks — the same instability the comment in
  `src/derivatives.jl` records removing from `_matching_positional`. The generated body splices the
  `nothing`s in as literals, so there is no length left to be runtime about.

- **The reverse pass had the same two defects**, on the walk that turns a cotangent in the shape of the
  parameters back into a flat gradient. `_accumulate_named!` was a `Base.tail` chain over
  `values(l.children)` and `keys(l.children)`; it is written out now, with the branch's keys spliced in
  as *literal* `Symbol`s — which is what keeps `_cotangent_get`'s `haskey` a compile-time question, the
  property the chain had been relying on constant propagation for. `_matching_named` goes with it.

  Bytes per pullback call, on the same flat sets and again on Julia 1.13:

  | children | 16 | 32 | 48 | 64 | 128 | 369 |
  |---|---|---|---|---|---|---|
  | 0.2.1 | 320 | 576 | 70 592 | 161 504 | not reached | not reached |
  | 0.2.2 | 320 | 576 | **848** | **1 120** | **2 112** | **6 208** |

  Identical up to 32 children and 144× better at 64, which is the signature of the same cliff: below it
  the temporary tuples stay in registers, above it they do not.

  The two *positional* walks — `_accumulate_positional!` and `_matching_positional` — are deliberately
  left as chains, and the comment beside them now says why: their length is the number of blocks of a
  single leaf, two for a `StiefelLieAlgHorMatrix` and one for a Grassmann lift, so they are never wide
  the way a branch of layers is.

### Added

- **`ParameterSet`**, exported: `Union{NetworkParameters, NamedTuple}`, the type to dispatch on when a
  method takes *the parameters* and does not care which of the two forms it was handed.

  Every package in this ecosystem was spelling that union out inline — `AbstractNeuralNetworks` at
  eight sites, `GeometricMachineLearning` at fifteen, `GeometricOptimizers` at sixteen — and
  `SymbolicNeuralNetworks` had named it `EquationSet` for the equation sets that share the shape. One
  name means a reader meets the same type in all of them.

  Its docstring states the two things it is not. It is **not** `isparametertree`, which is also true of
  a `Tuple`, because a `Tuple` is a branch the walks recurse into — what `freeparameters` returns for a
  multi-block leaf — and never a set of parameters handed in whole. And it puts no bound on the element
  type or the depth, so it is **not** `GeometricOptimizers.ParameterContainer{T}`, which additionally
  requires the `NamedTuple` half to be flat and element-type-homogeneous.

- `test/wide_branch_tests.jl`, which walks a 369-child branch through `flatten`/`unflatten`,
  `mapparameters` at both arities, `mapparameters!`, `foreachparameters` with and without the `nothing`
  skip, `foldparameters` and `parameter_eltype`, and pins **every** in-place form at zero allocations
  there — `flatten!` and `unflatten!`, and `mapparameters!`, `mapstorage!` and `foreachparameters` at
  arity one, two and three and on the `nothing`-skip path. It also covers 48, either side of the 32
  fields `Base` unrolls a tuple up to.

  A 48-wide branch whose children are themselves branches is covered as well, and separately, because
  the two shapes fail differently: a flat branch's children are arrays, which are heap objects
  already, so a walk that reaches one has nothing to keep off the heap, while a branch of branches has
  to be taken apart in place at every level. The width is a consumer's; the nesting is a network's.

  It asserts the properties and not the timings — a wall-clock bound would be a flake on a loaded
  machine — so the regression test is that the file *completes*, which before this release it did not.
  It costs the suite about 45 s on Julia 1.13 and about 1 m 45 s on 1.11, all of it compilation at the
  wide width, and that is the honest price of covering the width a consumer actually has.

- `scripts/wide_branch_cost.jl`, the harness behind both tables above.

[GeometricOptimizers#68]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/68

## [0.2.1] — 2026-08-25

### Fixed

- **The out-of-place `unflatten` walked its children with `map` over a closure.** It was the one walk
  in the package written as `map(c -> unflatten(c, v), values(l.children))` rather than as the
  `Base.tail` recursion `src/walk.jl` states the house rule to be, and which `flatten!`, `unflatten!`
  and `mapparameters` all follow. Both are equally inferable — nothing here was ever type unstable —
  but `map` leaves the closure over the flat vector to be elided, and not every version elides it
  ([#13]).

  Bytes per call, measured from inside a function on the three-leaf set of the `flatten` docstring, and
  per leaf on a flat set of four identical ones:

  | | 1.10 | 1.11 | 1.12 | 1.13 |
  |---|---|---|---|---|
  | three-leaf set, before | 720 | 240 | 240 | 240 |
  | three-leaf set, after | **400** | **112** | 240 | 240 |
  | per leaf, before | 176 | 96 | 96 | 96 |
  | per leaf, after | 176 | **48** | 96 | 96 |

  Julia 1.11 is where it tells: the recursion halves what every leaf costs, from two heap allocations
  to one. On 1.10 the saving depends on the shape of the set rather than on its size — the closure is
  elided for some trees and not others — and 1.12 and 1.13 elide it throughout, so there the change is
  neutral. It is neutral or better everywhere; no set measured costs more than it did.

  The same rewrite takes the walk off `Base`'s `Any32` fallback, which `map` drops to past 32 children
  and which returns a tuple with no concrete type — so a parameter set with more than 32 layers, or a
  layer with more than 32 entries, no longer unflattens through a type-unstable path. This holds for
  the `AbstractMatrix` overload set as well, which splits a Jacobian by parameter block.

  `SymbolicNeuralNetworks` splits the result of a generated function into the shape of a parameter set
  through this function, once per call, which is how the cost came to be noticed
  ([SymbolicNeuralNetworks#55](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/55)).
- **The `unflatten` rrule dispatched dynamically once per child, on every reverse pass.** Its branch
  walks read `for k in keys(l.children)` and indexed `l.children[k]`: a runtime `Symbol` into a
  heterogeneous `NamedTuple` gives back the union of the branch's child layout types, so nothing about
  the recursion was known at compile time. The cost climbed with the number of layers and with the
  depth of the tree, which is exactly what a gradient step pays most often. The walks are now the same
  `Base.tail` recursion as everything else ([#13]).

  Four leaves, one to a layer, against a gradient vector that is itself 128 bytes:

  | | 1.10 | 1.11 | 1.12 | 1.13 |
  |---|---|---|---|---|
  | before | 1664 | 1664 | 1664 | 1664 |
  | after | **128** | **128** | **128** | **128** |

  The pullback now allocates its answer and, on three of the four versions, nothing else. Grouping the
  same leaves into layers no longer costs anything at all, where it used to multiply the bill by five.
- `_matching_storage`, which narrows a cotangent to the storage of a multi-block structured leaf, built
  its positional case with `ntuple` over a *runtime* length. `Base` stops inferring that past ten, so a
  leaf whose `freeparameters` are more than ten blocks fell to a `Tuple` with no concrete type. Both of
  its cases are recursions now ([#13]).
- `parameterrange`, `length(::ParameterLayout)`, `_reshape_leaf` and the `parameter_eltype` family are
  marked `@inline` alongside, for the same reason the in-place walks are. None of these moved a
  measurement; they are consistency with the rule the walks follow, not a claimed effect ([#13]).

## [0.2.0] — 2026-08-24

### Changed

- **`NetworkParameters` carries the element type of its leaves as its first type parameter**, so that
  `T` binds in a method signature: `f(ps::NetworkParameters{T}) where {T}`, or a `Union` with
  `AbstractVector{T}`. `GeometricOptimizers` takes the element type from the *type* of the solution it
  is handed — eleven sites spell it `where {T, VT<:OptimizerSolution{T}}`, across every cache and
  state constructor, both `Optimizer` constructors and the `BFGSState` `update!` methods — and a
  parameter set could not join that union while it carried no such parameter ([#12]).

  The element type is derived by `parameter_eltype` at construction rather than chosen. Naming it, as
  `NetworkParameters{T}(nt)`, asserts it and raises if the leaves say otherwise, so a set whose type
  disagrees with what it holds has no inhabitants. Deriving it costs nothing: it folds to a constant
  and construction allocates as it did.

  It is a *promotion*, not a guarantee of uniformity — unlike `GeometricOptimizers`'
  `ArrayNamedTuple{T}`, `NetworkParameters{T}` does not license assuming every leaf is a `T` — and a
  set with nothing to promote reports `Union{}`.
- **The element type comes first, so the keys no longer fit in the braces.** Build a set from its keys
  and values with `NetworkParameters(NamedTuple{keys}(vals))`, which is what the removed
  `NetworkParameters{Keys}(values)` did anyway. The old spelling raises an `ArgumentError` saying so
  rather than a `MethodError` about a conversion nobody asked for ([#12]).
- **`parameter_eltype` is total, where `freeparameters` is not.** Every parameter set now runs its
  constructor through this function, including sets that hold no numbers at all: a gradient tree with
  `nothing` where an untouched layer's entries would be, or `SymbolicNeuralNetworks` wrapping
  generated functions in one. A leaf this package cannot read numbers out of contributes nothing to
  the promotion, exactly as an empty set does, and both report `Union{}` — where a leaf with no
  protocol used to raise. A set that cannot be flattened is still a set with an element type; the
  protocol error comes from `parameterlayout`, which is where it decides something ([#12]).

  The recursion follows `freeparameters` for an `AbstractArray` leaf, which every structured leaf in
  the ecosystem is; asking an arbitrary type where its storage is would mean raising, which this
  function must not do. A leaf that keeps its numbers behind another interface opts in with one method
  more, `parameter_eltype(x::MyLeaf) = parameter_eltype(freeparameters(x))`, and `flatten` says so if
  it is missing rather than guessing.
- `parameter_eltype(ps::NetworkParameters)` reads the type parameter instead of walking the leaves, so
  `flatten(ps)` and the `flatten` rrule are cheaper than they were.

### Added

- `NetworkParameters{T, Keys, ValueTypes}(nt)` and `NetworkParameters{T}(nt)` are written out
  ([#12]), since the derived element type makes the one that computes it an inner constructor and
  suppresses the defaults. The three-parameter form is not optional: `ChainRulesCore.construct`
  calls it from `+(::P, ::Tangent{P})`, which is how a parameter set and a cotangent add.
- `flatten(ps)` raises an `ArgumentError` naming `parameter_eltype` when the leaves promote to
  `Union{}` and the layout nevertheless found numbers to copy — a leaf keeping its storage behind an
  interface that is not an array's, with no `parameter_eltype` method of its own. It used to be a
  `Vector{Union{}}` failing on the first copy. An empty set still flattens to a `Vector{Union{}}`
  ([#12]).
- `Base.hash(::NetworkParameters)`, forwarding to the wrapped `NamedTuple` as `isequal` does. The
  default takes the type in, and the element type is part of that, so a `Float32` set and a
  `Float64` set holding the same numbers — `isequal`, and `==` — used to hash apart and behave as
  two different `Dict` keys ([#12]).

### Unchanged

- **The on-disk HDF5 format.** The element type is derived on read, so nothing new is written and a
  file from 0.1 loads with the element type its leaves imply.
- `Base.eltype` is still deliberately undefined on `NetworkParameters`. The type forwards the
  `NamedTuple` interface, for which `eltype` means the type of what iteration yields — a layer's
  `NamedTuple` — rather than the numeric element type.
- Equality, which compares the wrapped `NamedTuple`s: a `Float32` set and a `Float64` set holding the
  same numbers are still `==`, and have different element types.

## [0.1.1] — 2026-08-23

### Fixed

- The error raised for an unregistered type in a file `GeometricMachineLearning` wrote no longer
  offers a remedy that cannot work. Such a group carries a `gml_type` attribute and holds the type's
  fields under their own names, with no `storage` for `rebuild` to take, so registering the
  type with `register_parameter_type!` is the only way into it — the message now says that instead of
  suggesting a prototype ([#11]).
- Passing a prototype for one of those leaves raised `KeyError: "storage"` from inside the reader.
  `load` now checks for the storage before reaching for it and raises an `ArgumentError` naming the
  tagged type and the registration it needs ([#11]).

### Added

- The branch of the HDF5 reader that loads those files is covered by the test suite, including what
  it hands a registered reconstructor: the group's fields, the same object in both argument
  positions, reachable by name only. One registered name serves both that layout and the current
  one, which is the contract `GeometricOptimizers` is written against ([#11]).
- Coverage for a structured leaf that keeps everything in its storage and has no
  `parameter_metadata` — a manifold element, in the types this package is written for. The writer
  leaves the metadata group out of the file and the reader hands the reconstructor an empty
  `NamedTuple`; both halves of that agreement are now pinned ([#11]).

### Changed

- `docs/src/hdf5.md` says what the corrected errors say: that a `gml_type` file has to go through the
  registry, that the group's fields reach the reconstructor in both positions, and that the metadata
  group is written only when the type has metadata to record ([#11]).

## [0.1.0] — 2026-08-21

The initial release: the `NetworkParameters` container and its flat `FlatParameters` form, the
`freeparameters`/`rebuild` leaf protocol that structured parameter types extend, the tree walks,
`flatten`/`unflatten` with their derivative rules, and HDF5 storage through a package extension.

[#11]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/11
[#12]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/12
[#13]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/13
[#15]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/15
[#16]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/16
[#18]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/18
[#19]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/19
[#22]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/22
[0.2.5]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.5
[0.2.4]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.4
[0.2.3]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.3
[0.2.2]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.2
[0.2.1]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.1
[0.2.0]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.2.0
[0.1.1]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.1
[0.1.0]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/releases/tag/v0.1.0
