# Operations guide

## Determinism

The Numba and pure-Python paths execute the same scalar recurrence and are
tested for exact score and traceback parity. Set
`GINFINITY_SW_NO_NUMBA=1` before importing the package to force the reference
path for diagnosis.

## Performance

Numba compiles the core on first use and caches it. Warm the service with one
small alignment during startup if first-request latency matters. Use
`traceback=False` with the legacy `rank` API when only the strongest score is
required. `rank_pairs` and the CLI pair report intentionally trace each
qualified HSP so that total score, max score, and the aggregate E-value are
available.

The multi-HSP path repeats dynamic programming up to `max_alignments` times
per pair. Lower `max_alignments` or raise `min_score` for large archives when
the pair-level report does not need every weak HSP.

## Memory safety

The `max_cells` preflight is mandatory on public-facing inputs. Traceback uses
multiple arrays proportional to `Lq × Lt`; score-only mode uses linear DP
row memory after the score matrix is formed.

## Deployment

Pin the package Git tag or full commit and the NumPy/Numba versions. The exact
validated versions are in `requirements.lock`. Build the wheel in a trusted
pipeline and promote that same artifact across environments.
