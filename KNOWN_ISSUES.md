# Known issues

What is known to be broken or incomplete and is not fixed yet. Delete an entry when its issue is
fixed; the fix goes in `CHANGELOG.md`.

## Upstream

### K1 · Revise prints EMFILE errors in the test log

- location: `test/quality/jet.jl:18`
- evidence: JET 0.12 loads Revise, and Revise's file watcher runs out of file handles. On Julia
  1.13.1 with JET 0.12.2, `grep -c 'UNHANDLED TASK ERROR.*EMFILE'` counts 12 blocks in the log of
  `run-tests.jl <repository> full` on a `git archive` copy of the branch `tests/part-j6`, and 0 in the
  same run on a copy of `origin/main` (7772467). These blocks are not test failures, and the test
  totals of every other file do not change.
- kind: upstream
- found: 2026-10-01
