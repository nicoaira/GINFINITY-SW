# Operations guide

## Determinism

The Numba and pure-Python paths execute the same scalar recurrence and are
tested for exact score and traceback parity. Set
`GINFINITY_SW_NO_NUMBA=1` before importing the package to force the reference
path for diagnosis.

## Performance

Numba compiles the core on first use and caches it. Warm the service with one
small alignment during startup if first-request latency matters. Use
`traceback=False` for ranking, where only the score is required.

## Memory safety

The `max_cells` preflight is mandatory on public-facing inputs. Traceback uses
multiple arrays proportional to `Lq × Lt`; score-only mode uses linear DP
row memory after the score matrix is formed.

## Deployment

Pin the package Git tag or full commit and the NumPy/Numba versions. The exact
validated versions are in `requirements.lock`. Build the wheel in a trusted
pipeline and promote that same artifact across environments.
