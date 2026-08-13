# Algorithm and coordinate contract

For query embedding `q[i]` and target embedding `t[j]`, the base similarity is
their cosine. `ScoringParameters` transforms it as:

```text
S[i,j] = clip(gamma * (cos(q[i], t[j]) - mu) / sigma,
              score_min, score_max) - score_offset
```

The exact local affine-gap recurrence is:

```text
E[i,j] = max(H[i,j-1] - gap_open, E[i,j-1] - gap_extend)
F[i,j] = max(H[i-1,j] - gap_open, F[i-1,j] - gap_extend)
H[i,j] = max(0, H[i-1,j-1] + S[i,j], E[i,j], F[i,j])
```

Opening from `H` permits adjacent opposite-direction gaps. Ties follow a stable
order: local restart, diagonal, query gap, target gap; a later choice replaces
an earlier one only when strictly greater.

Spans are 0-based and half-open. Each `columns` element is `(query_index,
target_index)`; `-1` in either position represents a gap. An empty alignment
has score `0`, spans `(0, 0)`, and no columns.

The score-only path skips traceback storage and produces the same score as the
traceback path.
