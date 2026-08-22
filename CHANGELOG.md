# Changelog

## Unreleased

## 1.1.0 - 2026-08-22

- Added disjoint multi-HSP extraction and pair-level collapse reports with
  total score, max score, and an aggregate BLAST-style E-value.
- Added `scripts/release.py` to preflight a release and publish it by
  creating the GitHub release that starts the PyPI and Anaconda.org
  workflows.

## 1.0.1 - 2026-08-13

- Republished the 1.0.0 aligner from one git tag so PyPI and the personal
  Anaconda channel ship the same source tree.
- Publish conda packages from the GitHub release tag.

## 1.0.0 - 2026-08-12

- First standalone production release.
- Added standalone PyPI and Anaconda distribution metadata and publishing
  workflows.
- Licensed under PolyForm Noncommercial 1.0.0.
- Exact local affine-gap dynamic programming with traceback and score-only paths.
- Numba acceleration with equivalent reference fallback.
- Validated scoring configuration, deterministic ranking, RNA rendering, CLI,
  and quadratic-work preflight.
- Configurable identifier, sequence, and structure metadata columns.
- Alignment rendering with summary identities and conserved-pair markers.
