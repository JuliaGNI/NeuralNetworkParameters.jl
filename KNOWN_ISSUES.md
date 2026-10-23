# Known issues

What is known to be broken or incomplete and is not fixed yet. Delete an entry when its issue is
fixed; the fix goes in `CHANGELOG.md`.

## Upstream

### K1 · Revise prints EMFILE errors in the test log

- location: `test/quality/jet.jl:20`
- evidence: JET 0.12 loads Revise, and Revise's file watcher runs out of file handles. On Julia
  1.13.1 with JET 0.12.2, `grep -c 'UNHANDLED TASK ERROR.*EMFILE'` counts 12 blocks in the log of
  `run-tests.jl <repository> full` on a `git archive` copy of this tree, and 0 in the same run on
  a copy of 7772467, the commit before `test/quality/jet.jl` was added. These blocks are not test
  failures, and the test totals of every other file do not change.
- kind: upstream
- found: 2026-10-01

### K2 · Zygote takes minutes, or overflows its stack, on a set of 369 leaves

- location: `ext/ZygoteRulesExt.jl`, and every path that `Zygote.gradient` takes through a set
- evidence: `scripts/zygote_wide_set.jl` takes the first `Zygote.gradient` of a loss over a flat set
  of 369 leaves of 8 × 8, on one path per process: a `NetworkParameters` (`nnp`), a bare
  `NamedTuple` (`nt`), or the flat vector through `unflatten` (`flat`). On Julia 1.13.1 with Zygote
  0.7.13, `julia --project=. -e 'using TestEnv; TestEnv.activate();
  include("scripts/zygote_wide_set.jl")' nnp 369` prints `nnp 369: gradient taken, first call
  293.3 s; Julia 1.13.1, Zygote 0.7.13`. The script does not reproduce the stack overflow. On Julia
  1.12.7 with Zygote 0.7.13 and NeuralNetworkParameters 0.4.2, the same set
  raised a `StackOverflowError` in Zygote's recursive `_unapply` over an `NTuple{110}` on all three
  paths, and the first call took 3.95, 5.9, 14.6 and 51.6 s at 16, 32, 64 and 128 leaves; that run
  of 2026-10-05 used a script outside this repository, of which `scripts/zygote_wide_set.jl` keeps
  the shape of the set and the loss. The script on Julia 1.12.7 with this source had not finished
  after 12 minutes. No design of this package changes it, because the bare `NamedTuple` overflowed
  too.
- kind: upstream
- found: 2026-10-05

## Defects

### K3 · A storage `NamedTuple` built from one field of the leaf gets a zero gradient

- location: `src/derivatives.jl`, `_storage_cotangent(_, ::NamedTuple, ::NamedTuple)`
- evidence: a leaf that keeps its blocks in one `Tuple` field `b` and returns them as
  `freeparameters(s) = (x = s.b[1], y = s.b[2])` gets an all-zero gradient, with no error, for a
  loss that reads `q.L.b`. The storage keys `x` and `y` are matched against the field names of the
  structural tangent (`b`), so each block is a hole. The `storage_gradient` docstring says that a
  storage that is not the leaf's fields raises. `julia --project=. -e 'using TestEnv;
  TestEnv.activate(); include("scripts/tuple_field_storage.jl")'` prints
  `NTBox Float32: Float32[0.0, 0.0, 0.0, 0.0]` and `NTBox Float64: [0.0, 0.0, 0.0, 0.0]`, where
  the gradient is `[1, 1, 2, 2]`. The same holds on 0.5.0 (`17d8356`). A fix changes a value.
- kind: defect
- found: 2026-10-11

### K4 · A set with a `nothing` leaf has no flat form, so its flat gradient raises

- location: `src/layout.jl`, `_layout(x, offset)`
- evidence: `flatten(NetworkParameters((a = [1.0, 2.0], n = nothing)))` raises an `ArgumentError`
  that begins "no method of `freeparameters` for `Nothing`. A leaf of a parameter set has", on
  0.5.0 (`17d8356`) as on the current source. So a flat Zygote gradient through `unflatten`, and a loss that
  reads `flatten(q)` of such a set, raise too. The structural gradient of the same set is taken,
  and its `nothing` leaf stays `nothing`, as the `map_cotangent` docstring says. No docstring says
  whether a `nothing` leaf has a flat form.
- kind: found late
- found: 2026-10-11

### K9 · A branch of the wrong kind raises a `MethodError`, not an `ArgumentError`

- location: `src/flatten.jl:116`
- evidence: `_check_layout_keys(_, _) = nothing` accepts a `NamedTuple` branch against a
  `TupleLayout`, and a `Tuple` branch against a `NestedLayout`, so the mismatch reaches
  `_flatten!` or `_unflatten!`, which have no method for it. With `l` the layout of
  `NetworkParameters((a = ([1.0, 2.0], [3.0]),))` and
  `bad = NetworkParameters((a = (x = [9.0, 9.0], y = [9.0]),))`, both `flatten!(copy(v), bad, l)`
  and `unflatten!(bad, l, v)` raise a `MethodError` on Julia 1.13.1. Nothing is written before the
  error. A key or width mismatch raises an `ArgumentError` that names the set and the layout.
- kind: defect
- found: #42

### K10 · The first call of `map_cotangent` compiles twice as long as on 0.5.0

- location: `src/derivatives.jl:263`
- evidence: `julia --startup-file=no --project=. scripts/map_cotangent_cost.jl`, one cold process
  per run, on Julia 1.13.1, at 369 leaves: the first call takes 0.50 s and 0.51 s on 0.5.0
  (`17d8356`), and 1.12 s and 1.20 s on the head of #42. The second call allocates 21 520 B on both.
  The cause is not known, and 8 and 64 leaves are not measured.
- kind: defect
- found: #42

### K11 · A storage walk of four or more sets inside a walk of fewer sets allocates at every leaf

- location: `src/walk.jl`, the leaf method of `_foreach_storage` and of `foldstorage`
- evidence: `julia --project=. scripts/nested_walk_cost.jl wide`, Julia 1.13.1, on a set of 3
  leaves, prints 192 B for `mapstorage!` of 4 sets in `mapstorage!` of 2 and 48 B for `foldstorage`
  of 4 in `foldstorage` of 2, in `Float32` and in `Float64`, where `foreachparameters` and
  `foldparameters` print 0 B. The storage walks call their leaf method at each leaf, because
  `freeparameters` decides at run time whether a leaf is its own storage, and inference widens that
  leaf method on its second time on the stack. Issue #44.
- kind: defect
- found: #43

### K12 · `mapstorage` in the function of another `mapstorage` does not infer its result type

- location: `src/walk.jl`, the methods of `mapstorage`
- evidence: `julia --startup-file=no --project=. scripts/nested_walk_cost.jl nestedmap`, Julia
  1.13.2, prints `concrete = false` for `mapstorage in mapstorage, two leaves` in `Float32` and in
  `Float64`, where the three rows of `mapparameters` in `mapparameters` print `concrete = true`.
  `mapstorage` recurses through its own methods and `_map_zip` at run time, as `mapparameters` did on
  0.5.2. GeometricOptimizers calls no `mapstorage` in its `src/`.
- kind: defect
- found: 2026-10-23

## Tests

### K5 · Test comments carry history and measurements

- location: `test/flatten.jl:99`
- evidence: `test/flatten.jl:99` says "`unflatten` used to be …".
  `test/integration/wide_branches.jl:204-206` gives "1.37 s and 13.44 s" on Julia 1.11.
  `test/derivatives.jl:203-210` says the comparison "was dispatched dynamically" and that two costs
  "now sit at 96". The same lines are on 0.5.0 (`17d8356`).
- kind: docs
- found: 2026-10-11

### K6 · An allocation test compares two non-zero readings with `==`

- location: `test/derivatives.jl:218`
- evidence: the testset "the pullback does not pay for the fields a structured leaf carries"
  asserts `cost_padded == cost_sym`, two `@allocated` readings that the comment above it puts at
  96 bytes each. An exact `==` of
  two non-zero readings fails where the figures are quantised differently, as on Windows; the
  test needs a difference under a stated tolerance. The same line is on 0.5.0 (`17d8356`).
- kind: defect
- found: 2026-10-11

### K8 · The bound of the flat-rule allocation test leaves a margin of 2 %

- location: `test/derivatives.jl:1130`
- evidence: the testset "the flat rule allocates one vector" asserts
  `abs(rule - vector) < vector ÷ 2`, where each side is already the least of five readings. On
  Julia 1.11.9 on macOS in CI, the Float64 difference read 1 032 224 B (the first attempt of
  "Julia min - macOS-latest" on `5c929e8`) and 507 936 B (on `e0d0cad`), against a bound of
  1 056 784 B. Both values are 2^k − 16 352. The Float32 bound is 532 496 B, so the same 1 MiB
  extra fails there. 120 local trials on Julia 1.11.9 aarch64, with 1 and 2 threads, read 32 B
  each. The cause is not known.
- kind: defect
- found: #42

### K13 · The wide-branch test file runs longer than 60 s cold on Julia 1.11

- location: `test/integration/wide_branches.jl`
- evidence: `julia +1.11 --startup-file=no run-tests.jl <repository> core` on Julia 1.11.9 reports
  `D6: test/integration/wide_branches.jl runs 68.7 s cold, above the 60 s of a core file` on 0.5.2
  (`f35e7f1`), under a load average of about 22. On Julia 1.13.2 the same file runs 36.7 s. The
  limit is the test convention's for a file of the `core` group; a slower file goes to `slow`. The
  time under a lower load is not measured.
- kind: defect
- found: 2026-10-23
