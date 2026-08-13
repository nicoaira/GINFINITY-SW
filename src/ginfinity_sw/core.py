"""Exact local affine-gap Smith–Waterman over vector sequences."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
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


@_njit(cache=True)
def _core(scores, gap_open, gap_extend, traceback):
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
        e = NEGATIVE_INFINITY
        h_left = 0.0
        for j in range(1, target_length + 1):
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


def align_scores(scores: np.ndarray, parameters: ScoringParameters, *,
                 traceback: bool = True,
                 max_cells: int = 16_777_216) -> Alignment:
    scores = _matrix(scores, "scores")
    cells = int(scores.shape[0]) * int(scores.shape[1])
    if cells > max_cells:
        raise ValueError(
            f"alignment needs {cells:,} cells; maximum is {max_cells:,}")
    if not cells:
        return Alignment(0.0, (0, 0), (0, 0), (), 0)
    best, count, buffer = _core(
        scores, float(parameters.gap_open),
        float(parameters.gap_extend), bool(traceback))
    if not traceback or best <= 0.0:
        return Alignment(float(best), (0, 0), (0, 0), (), scores.shape[0])
    columns = tuple(
        (int(buffer[index, 0]), int(buffer[index, 1]))
        for index in range(count - 1, -1, -1))
    matches = [(query, target) for query, target in columns
               if query >= 0 and target >= 0]
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


def rank(query_embeddings: np.ndarray,
         candidates: Iterable[tuple[str, np.ndarray]], *,
         params: ScoringParameters,
         max_cells: int = 16_777_216) -> list[tuple[str, float]]:
    """Rank candidate embedding matrices by score, then identifier."""
    rows = [(identifier, align(
        query_embeddings, embedding, params=params, traceback=False,
        max_cells=max_cells).score) for identifier, embedding in candidates]
    return sorted(rows, key=lambda row: (-row[1], row[0]))
