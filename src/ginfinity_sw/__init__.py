"""Public API for GINFINITY-SW."""

from .core import (HAVE_NUMBA, Alignment, ScoringParameters, align,
                   align_scores, normalize_embeddings, rank,
                   similarity_matrix, transform_scores)
from .formatting import format_alignment
from .metadata import read_metadata_table

__version__ = "1.0.1"

__all__ = [
    "Alignment", "HAVE_NUMBA", "ScoringParameters", "align", "align_scores",
    "format_alignment", "normalize_embeddings", "rank", "read_metadata_table",
    "similarity_matrix",
    "transform_scores", "__version__",
]
