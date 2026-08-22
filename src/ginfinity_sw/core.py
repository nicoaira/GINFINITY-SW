"""Exact local affine-gap Smith–Waterman over vector sequences."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from math import exp, isfinite, log
from typing import Iterable

import numpy as np

try:
    if os.environ.get("GINFINITY_SW_NO_NUMBA") or os.environ.get("SW_NO_NUMBA"):
        raise ImportError("compiled path disabled")
    from numba import njit as _njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function


NEGATIVE_INFINITY = -1.0e30


@dataclass(frozen=True, slots=True)
class ScoringParameters:
    """Similarity transform and affine-gap costs."""

    mu: float
    sigma: float = 1.0
    gamma: float = 1.5
    score_min: float = -4.0
    score_max: float = 8.0
    gap_open: float = 12.0
    gap_extend: float = 2.0
    score_offset: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("all scoring parameters must be finite")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.score_min >= self.score_max:
            raise ValueError("score_min must be smaller than score_max")
        if self.gap_open < 0 or self.gap_extend < 0:
            raise ValueError("gap costs must be non-negative")


@dataclass(frozen=True, slots=True)
class EValueParameters:
    """Parameters for the BLAST-style pair-level E-value approximation.

    ``lambda_`` and ``k`` correspond to the Karlin--Altschul ``lambda`` and
    ``K`` constants.  They are deliberately separate from the substitution
    scoring parameters: cosine-derived scores do not have universal
    Karlin--Altschul constants, so callers can calibrate these values for a
    particular model/background without changing alignment behavior.
    """

    lambda_: float = 1.0
    k: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.lambda_) or self.lambda_ <= 0:
            raise ValueError("lambda_ must be a finite positive number")
        if not isfinite(self.k) or self.k <= 0:
            raise ValueError("k must be a finite positive number")

    def to_dict(self) -> dict[str, float]:
        """Return JSON-friendly names matching the statistical notation."""
        return {"lambda": self.lambda_, "k": self.k}


@dataclass(frozen=True, slots=True)
class Alignment:
    """One local alignment with 0-based half-open spans."""

    score: float
    query_span: tuple[int, int]
    target_span: tuple[int, int]
    columns: tuple[tuple[int, int], ...]
    rows_processed: int

    @property
    def matches(self) -> tuple[tuple[int, int], ...]:
        return tuple((query, target) for query, target in self.columns
                     if query >= 0 and target >= 0)

    @property
    def match_count(self) -> int:
        return len(self.matches)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "query_span": list(self.query_span),
            "target_span": list(self.target_span),
            "columns": [list(column) for column in self.columns],
            "match_count": self.match_count,
            "rows_processed": self.rows_processed,
        }


@dataclass(frozen=True, slots=True)
class AlignmentSet:
    """All non-overlapping local HSPs for one query-target pair.

    A pair is reported as one result, while ``alignments`` retains each local
    traceback.  The aggregate score is the sum of HSP scores, and ``max_score``
    is the strongest individual HSP, mirroring BLAST-style pair summaries.
    The E-value is computed once for the set from ``total_score``.
    """

    alignments: tuple[Alignment, ...]
    query_length: int
    target_length: int
    evalue_parameters: EValueParameters = EValueParameters()

    def __post_init__(self) -> None:
        if self.query_length < 0 or self.target_length < 0:
            raise ValueError("sequence lengths must be non-negative")
        if not isinstance(self.evalue_parameters, EValueParameters):
            raise TypeError("evalue_parameters must be EValueParameters")
        for alignment in self.alignments:
            if not isinstance(alignment, Alignment):
                raise TypeError("alignments must contain Alignment values")

    @property
    def alignment_count(self) -> int:
        return len(self.alignments)

    @property
    def total_score(self) -> float:
        return float(sum(alignment.score for alignment in self.alignments))

    @property
    def max_score(self) -> float:
        return float(max(
            (alignment.score for alignment in self.alignments),
            default=0.0))

    @property
    def score(self) -> float:
        """Legacy score alias for the strongest HSP."""
        return self.max_score

    @property
    def match_count(self) -> int:
        return sum(alignment.match_count for alignment in self.alignments)

    @property
    def query_span(self) -> tuple[int, int]:
        if not self.alignments:
            return (0, 0)
        return (
            min(alignment.query_span[0] for alignment in self.alignments),
            max(alignment.query_span[1] for alignment in self.alignments),
        )

    @property
    def target_span(self) -> tuple[int, int]:
        if not self.alignments:
            return (0, 0)
        return (
            min(alignment.target_span[0] for alignment in self.alignments),
            max(alignment.target_span[1] for alignment in self.alignments),
        )

    @property
    def search_space(self) -> int:
        return self.query_length * self.target_length

    @property
    def evalue(self) -> float:
        return pair_evalue(
            self.total_score,
            self.query_length,
            self.target_length,
            parameters=self.evalue_parameters,
        )

    def to_dict(self) -> dict:
        """Return one pair-level report row with the individual HSPs nested."""
        return {
            "score": self.score,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "evalue": self.evalue,
            "alignment_count": self.alignment_count,
            "match_count": self.match_count,
            "query_length": self.query_length,
            "target_length": self.target_length,
            "search_space": self.search_space,
            "query_span": list(self.query_span),
            "target_span": list(self.target_span),
            "evalue_parameters": self.evalue_parameters.to_dict(),
            "alignments": [alignment.to_dict()
                           for alignment in self.alignments],
        }


# A descriptive alias for callers that prefer the report-oriented name.
PairAlignment = AlignmentSet


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(value)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = _matrix(embeddings, "embeddings")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def similarity_matrix(query_embeddings: np.ndarray,
                      target_embeddings: np.ndarray) -> np.ndarray:
    query = normalize_embeddings(query_embeddings)
    target = normalize_embeddings(target_embeddings)
    if query.shape[1] != target.shape[1]:
        raise ValueError(
            "query and target embedding dimensions must be equal")
    return np.ascontiguousarray(query @ target.T, dtype=np.float64)


def transform_scores(similarity: np.ndarray,
                     parameters: ScoringParameters) -> np.ndarray:
    similarity = _matrix(similarity, "similarity")
    scores = parameters.gamma * (
        similarity - parameters.mu) / parameters.sigma
    scores = np.clip(scores, parameters.score_min, parameters.score_max)
    if parameters.score_offset:
        scores = scores - parameters.score_offset
    return np.ascontiguousarray(scores, dtype=np.float64)


def pair_evalue(total_score: float, query_length: int, target_length: int,
                *, parameters: EValueParameters | None = None) -> float:
    """Compute one BLAST-style E-value for a query-target HSP set.

    The aggregate score is used exactly once:

    ``E = K * query_length * target_length * exp(-lambda * total_score)``.

    This is an approximation for the package's transformed cosine scores, not
    a claim that the defaults are calibrated for every embedding model.  Use
    model/background-specific ``EValueParameters`` when statistical
    calibration is available.
    """
    if parameters is None:
        parameters = EValueParameters()
    if not isinstance(parameters, EValueParameters):
        raise TypeError("parameters must be EValueParameters")
    if isinstance(query_length, bool) or not isinstance(
            query_length, (int, np.integer)):
        raise TypeError("query_length must be an integer")
    if isinstance(target_length, bool) or not isinstance(
            target_length, (int, np.integer)):
        raise TypeError("target_length must be an integer")
    if query_length < 0 or target_length < 0:
        raise ValueError("sequence lengths must be non-negative")
    if not isfinite(total_score):
        raise ValueError("total_score must be finite")
    search_space = int(query_length) * int(target_length)
    if search_space == 0:
        return 0.0
    log_evalue = (
        log(parameters.k) + log(search_space)
        - parameters.lambda_ * float(total_score)
    )
    if log_evalue >= log(np.finfo(np.float64).max):
        return float("inf")
    return float(exp(log_evalue))


def _resolve_evalue_parameters(
    parameters: EValueParameters | None,
    lambda_: float | None,
    k: float | None,
) -> EValueParameters:
    if parameters is not None and (lambda_ is not None or k is not None):
        raise ValueError(
            "pass either evalue_parameters or lambda_/k, not both")
    if parameters is not None:
        if not isinstance(parameters, EValueParameters):
            raise TypeError("evalue_parameters must be EValueParameters")
        return parameters
    return EValueParameters(
        lambda_=1.0 if lambda_ is None else lambda_,
        k=1.0 if k is None else k,
    )


def _infer_length(alignments: tuple[Alignment, ...], *, query: bool) -> int:
    end = 0
    for alignment in alignments:
        span = alignment.query_span if query else alignment.target_span
        end = max(end, int(span[1]))
    return end


def collapse_alignments(
    alignments: Iterable[Alignment], *,
    query_length: int | None = None,
    target_length: int | None = None,
    evalue_parameters: EValueParameters | None = None,
    lambda_: float | None = None,
    k: float | None = None,
) -> AlignmentSet:
    """Collapse one pair's HSPs into a single reportable result.

    The input is assumed to already belong to one query-target pair.  HSPs
    are sorted by their query and target spans for deterministic reporting;
    their tracebacks are not combined into a synthetic path.  ``query_length``
    and ``target_length`` should be supplied for a correct search space.  If
    omitted, they are conservatively inferred from the observed HSP spans.
    """
    values = tuple(alignments)
    if any(not isinstance(value, Alignment) for value in values):
        raise TypeError("alignments must contain Alignment values")
    values = tuple(sorted(
        values,
        key=lambda value: (
            value.query_span[0], value.target_span[0],
            value.query_span[1], value.target_span[1], -value.score,
        ),
    ))
    if query_length is None:
        query_length = _infer_length(values, query=True)
    if target_length is None:
        target_length = _infer_length(values, query=False)
    if isinstance(query_length, bool) or not isinstance(
            query_length, (int, np.integer)):
        raise TypeError("query_length must be an integer")
    if isinstance(target_length, bool) or not isinstance(
            target_length, (int, np.integer)):
        raise TypeError("target_length must be an integer")
    if query_length < 0 or target_length < 0:
        raise ValueError("sequence lengths must be non-negative")
    resolved = _resolve_evalue_parameters(evalue_parameters, lambda_, k)
    return AlignmentSet(
        alignments=values,
        query_length=int(query_length),
        target_length=int(target_length),
        evalue_parameters=resolved,
    )


@_njit(cache=True)
def _core(scores, gap_open, gap_extend, traceback, blocked_query,
          blocked_target):
    query_length, target_length = scores.shape
    previous_h = np.zeros(target_length + 1)
    current_h = np.zeros(target_length + 1)
    previous_f = np.full(target_length + 1, NEGATIVE_INFINITY)
    current_f = np.full(target_length + 1, NEGATIVE_INFINITY)
    if traceback:
        h_pointer = np.zeros(
            (query_length + 1, target_length + 1), dtype=np.int8)
        e_open = np.zeros(
            (query_length + 1, target_length + 1), dtype=np.bool_)
        f_open = np.zeros(
            (query_length + 1, target_length + 1), dtype=np.bool_)
    else:
        h_pointer = np.zeros((1, 1), dtype=np.int8)
        e_open = np.zeros((1, 1), dtype=np.bool_)
        f_open = np.zeros((1, 1), dtype=np.bool_)

    best = 0.0
    best_i = 0
    best_j = 0
    for i in range(1, query_length + 1):
        current_h[:] = 0.0
        current_f[:] = NEGATIVE_INFINITY
        if blocked_query[i - 1]:
            swap = previous_h
            previous_h = current_h
            current_h = swap
            swap = previous_f
            previous_f = current_f
            current_f = swap
            continue
        e = NEGATIVE_INFINITY
        h_left = 0.0
        for j in range(1, target_length + 1):
            if blocked_target[j - 1]:
                current_h[j] = 0.0
                current_f[j] = NEGATIVE_INFINITY
                e = NEGATIVE_INFINITY
                h_left = 0.0
                continue
            open_e = h_left - gap_open
            extend_e = e - gap_extend
            if open_e >= extend_e:
                e = open_e
                if traceback:
                    e_open[i, j] = True
            else:
                e = extend_e
            open_f = previous_h[j] - gap_open
            extend_f = previous_f[j] - gap_extend
            if open_f >= extend_f:
                f = open_f
                if traceback:
                    f_open[i, j] = True
            else:
                f = extend_f
            current_f[j] = f

            value = 0.0
            pointer = 0
            diagonal = previous_h[j - 1] + scores[i - 1, j - 1]
            if diagonal > value:
                value = diagonal
                pointer = 1
            if e > value:
                value = e
                pointer = 2
            if f > value:
                value = f
                pointer = 3
            current_h[j] = value
            if traceback:
                h_pointer[i, j] = pointer
            h_left = value
            if value > best:
                best = value
                best_i = i
                best_j = j
        swap = previous_h
        previous_h = current_h
        current_h = swap
        swap = previous_f
        previous_f = current_f
        current_f = swap

    buffer = (np.empty((query_length + target_length + 2, 2), dtype=np.int64)
              if traceback else np.empty((0, 2), dtype=np.int64))
    count = 0
    if traceback and best > 0.0:
        i, j, state = best_i, best_j, 0
        while i > 0 and j > 0:
            if state == 0:
                pointer = h_pointer[i, j]
                if pointer == 0:
                    break
                if pointer == 1:
                    buffer[count, 0] = i - 1
                    buffer[count, 1] = j - 1
                    count += 1
                    i -= 1
                    j -= 1
                elif pointer == 2:
                    state = 1
                else:
                    state = 2
            elif state == 1:
                buffer[count, 0] = -1
                buffer[count, 1] = j - 1
                count += 1
                j -= 1
                state = 0 if e_open[i, j + 1] else 1
            else:
                buffer[count, 0] = i - 1
                buffer[count, 1] = -1
                count += 1
                i -= 1
                state = 0 if f_open[i + 1, j] else 2
    return best, count, buffer


def _blocked_mask(value: np.ndarray | None, length: int,
                  name: str) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=np.bool_)
    value = np.asarray(value, dtype=np.bool_)
    if value.ndim != 1 or value.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},)")
    return np.ascontiguousarray(value)


def align_scores(scores: np.ndarray, parameters: ScoringParameters, *,
                 traceback: bool = True,
                 max_cells: int = 16_777_216,
                 blocked_query: np.ndarray | None = None,
                 blocked_target: np.ndarray | None = None) -> Alignment:
    scores = _matrix(scores, "scores")
    blocked_query = _blocked_mask(
        blocked_query, scores.shape[0], "blocked_query")
    blocked_target = _blocked_mask(
        blocked_target, scores.shape[1], "blocked_target")
    cells = int(scores.shape[0]) * int(scores.shape[1])
    if cells > max_cells:
        raise ValueError(
            f"alignment needs {cells:,} cells; maximum is {max_cells:,}")
    if not cells:
        return Alignment(0.0, (0, 0), (0, 0), (), 0)
    best, count, buffer = _core(
        scores, float(parameters.gap_open),
        float(parameters.gap_extend), bool(traceback),
        blocked_query, blocked_target)
    if not traceback or best <= 0.0:
        return Alignment(float(best), (0, 0), (0, 0), (), scores.shape[0])
    columns = tuple(
        (int(buffer[index, 0]), int(buffer[index, 1]))
        for index in range(count - 1, -1, -1))
    query_indices = [query for query, _ in columns if query >= 0]
    target_indices = [target for _, target in columns if target >= 0]
    query_span = ((min(query_indices), max(query_indices) + 1)
                  if query_indices else (0, 0))
    target_span = ((min(target_indices), max(target_indices) + 1)
                   if target_indices else (0, 0))
    return Alignment(float(best), query_span, target_span,
                     columns, scores.shape[0])


def align(query_embeddings: np.ndarray, target_embeddings: np.ndarray, *,
          params: ScoringParameters,
          traceback: bool = True,
          max_cells: int = 16_777_216) -> Alignment:
    """Align two ordered embedding matrices with exact local DP."""
    query_embeddings = _matrix(query_embeddings, "query_embeddings")
    target_embeddings = _matrix(target_embeddings, "target_embeddings")
    cells = query_embeddings.shape[0] * target_embeddings.shape[0]
    if cells > max_cells:
        raise ValueError(
            f"alignment needs {cells:,} cells; maximum is {max_cells:,}")
    scores = transform_scores(similarity_matrix(
        query_embeddings, target_embeddings), params)
    return align_scores(scores, params, traceback=traceback,
                        max_cells=max_cells)


def align_multiple(query_embeddings: np.ndarray,
                   target_embeddings: np.ndarray, *,
                   params: ScoringParameters,
                   max_alignments: int = 16,
                   min_score: float = 0.0,
                   min_match_count: int = 1,
                   max_cells: int = 16_777_216,
                   evalue_parameters: EValueParameters | None = None,
                   lambda_: float | None = None,
                   k: float | None = None) -> AlignmentSet:
    """Extract disjoint local HSPs and collapse them into one pair result.

    The best remaining local alignment is traced, then every query and target
    residue used by that HSP is blocked before the next extraction.  This
    prevents one physical residue from contributing to multiple HSP scores
    while still allowing separate conserved modules from the same pair.
    Extraction stops at ``max_alignments`` or when the best remaining HSP is
    below ``min_score``. HSPs below ``min_match_count`` are discarded and
    masked so a weaker, longer HSP can still be considered.
    """
    if isinstance(max_alignments, bool) or not isinstance(
            max_alignments, (int, np.integer)):
        raise TypeError("max_alignments must be an integer")
    if max_alignments < 0:
        raise ValueError("max_alignments must be non-negative")
    if not isfinite(min_score):
        raise ValueError("min_score must be finite")
    if isinstance(min_match_count, bool) or not isinstance(
            min_match_count, (int, np.integer)):
        raise TypeError("min_match_count must be an integer")
    if min_match_count < 1:
        raise ValueError("min_match_count must be positive")
    query_embeddings = _matrix(query_embeddings, "query_embeddings")
    target_embeddings = _matrix(target_embeddings, "target_embeddings")
    if query_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError(
            "query and target embedding dimensions must be equal")
    cells = query_embeddings.shape[0] * target_embeddings.shape[0]
    if cells > max_cells:
        raise ValueError(
            f"alignment needs {cells:,} cells; maximum is {max_cells:,}")
    resolved = _resolve_evalue_parameters(evalue_parameters, lambda_, k)
    scores = transform_scores(similarity_matrix(
        query_embeddings, target_embeddings), params)
    blocked_query = np.zeros(query_embeddings.shape[0], dtype=np.bool_)
    blocked_target = np.zeros(target_embeddings.shape[0], dtype=np.bool_)
    found: list[Alignment] = []
    while len(found) < int(max_alignments):
        current = align_scores(
            scores,
            params,
            traceback=True,
            max_cells=max_cells,
            blocked_query=blocked_query,
            blocked_target=blocked_target,
        )
        if current.score <= min_score:
            break
        query_used = [query for query, _ in current.columns if query >= 0]
        target_used = [target for _, target in current.columns if target >= 0]
        if not query_used or not target_used:
            break
        blocked_query[np.asarray(query_used, dtype=np.intp)] = True
        blocked_target[np.asarray(target_used, dtype=np.intp)] = True
        if current.match_count >= min_match_count:
            found.append(current)
    return collapse_alignments(
        found,
        query_length=query_embeddings.shape[0],
        target_length=target_embeddings.shape[0],
        evalue_parameters=resolved,
    )


def rank(query_embeddings: np.ndarray,
         candidates: Iterable[tuple[str, np.ndarray]], *,
         params: ScoringParameters,
         max_cells: int = 16_777_216) -> list[tuple[str, float]]:
    """Rank candidate embedding matrices by score, then identifier."""
    rows = [(identifier, align(
        query_embeddings, embedding, params=params, traceback=False,
        max_cells=max_cells).score) for identifier, embedding in candidates]
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def rank_pairs(query_embeddings: np.ndarray,
               candidates: Iterable[tuple[str, np.ndarray]], *,
               params: ScoringParameters,
               max_alignments: int = 16,
               min_score: float = 0.0,
               min_match_count: int = 1,
               max_cells: int = 16_777_216,
               evalue_parameters: EValueParameters | None = None,
               lambda_: float | None = None,
               k: float | None = None) -> list[tuple[str, AlignmentSet]]:
    """Rank candidates using one collapsed multi-HSP result per pair.

    E-value is the primary ordering key, followed by aggregate and strongest
    HSP score.  ``rank`` remains available for the original single-HSP score
    ranking API.
    """
    rows = [(
        identifier,
        align_multiple(
            query_embeddings,
            embedding,
            params=params,
            max_alignments=max_alignments,
            min_score=min_score,
            min_match_count=min_match_count,
            max_cells=max_cells,
            evalue_parameters=evalue_parameters,
            lambda_=lambda_,
            k=k,
        ),
    ) for identifier, embedding in candidates]
    return sorted(rows, key=lambda row: (
        row[1].evalue, -row[1].total_score, -row[1].max_score, row[0]))
