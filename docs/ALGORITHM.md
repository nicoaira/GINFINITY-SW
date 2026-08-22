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

## Multiple local HSPs and pair reports

`align_multiple` repeatedly traces the best remaining local alignment. After
each HSP, every query and target residue used by that traceback is blocked, so
the next HSP cannot count an already-used residue. Extraction stops at
`max_alignments`, or when the best remaining HSP is below `min_score`. HSPs
below `min_match_count` are discarded and masked so a weaker, longer HSP can
still be considered. This produces disjoint local HSPs for separate conserved
modules; it does not manufacture a synthetic path across unrelated regions.

`collapse_alignments` groups the HSPs for one query-target pair into an
`AlignmentSet` without discarding their individual tracebacks:

```text
total_score = sum(hsp.score for hsp in alignments)
max_score   = max(hsp.score for hsp in alignments)
```

The pair-level E-value is computed once from the aggregate score:

```text
E = K * query_length * target_length * exp(-lambda * total_score)
```

`EValueParameters` exposes `lambda` and `K` for model/background-specific
calibration. The defaults are a transparent approximation for transformed
cosine scores, not universal Karlin--Altschul calibration constants. JSON
reports retain the HSP list under `alignments` while placing `total_score`,
`max_score`, and `evalue` in the same pair-level result.
