# Contributing

Algorithm changes require exact parity tests for the accelerated and reference
paths, known-answer locality tests, cell-budget tests, and a documented
coordinate-contract review.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e . pytest build
.venv/bin/python -m pytest -q
.venv/bin/python -m build
```

Keep this distribution independent of any encoder or checkpoint. Add a
changelog entry for every user-visible change. Publish a release with
`python scripts/release.py` after the version bump is on `main`; see
[docs/PUBLISHING.md](docs/PUBLISHING.md).
