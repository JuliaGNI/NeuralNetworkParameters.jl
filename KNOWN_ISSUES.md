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

### K2 · Zygote cannot differentiate a set of 369 leaves

- location: `ext/ZygoteRulesExt.jl`, and every path that `Zygote.gradient` takes through a set
- evidence: on Julia 1.12.7 with Zygote 0.7.13 and NeuralNetworkParameters 0.4.2, `Zygote.gradient`
  of a loss over a flat set of 369 leaves of 8 × 8 raises a `StackOverflowError` in Zygote's
  recursive `_unapply` over an `NTuple{110}`. It raises on every path: a `NetworkParameters`, a bare
  `NamedTuple`, and the flat vector through `unflatten`. The first call takes 3.95, 5.9, 14.6 and
  51.6 s at 16, 32, 64 and 128 leaves. Measured on 2026-10-05 with
  `~/Research/.scratch/perf/NNP-design/bench_grad.jl`; the log is `log_grad.txt` beside it, lines
  21–24 and 29. No design of this package changes it, because the bare `NamedTuple` fails too.
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
