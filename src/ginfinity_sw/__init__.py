"""Public API for GINFINITY-SW."""

from .core import (
    HAVE_NUMBA,
    Alignment,
    AlignmentSet,
    EValueParameters,
    PairAlignment,
    ScoringParameters,
    align,
    align_multiple,
    align_scores,
    collapse_alignments,
    normalize_embeddings,
    pair_evalue,
    rank,
    rank_pairs,
    similarity_matrix,
    transform_scores,
)
from .formatting import format_alignment, format_alignment_set
from .metadata import read_metadata_table

__version__ = "1.1.0"

__all__ = [
    "Alignment", "AlignmentSet", "EValueParameters", "HAVE_NUMBA",
    "PairAlignment", "ScoringParameters", "align", "align_multiple",
    "align_scores", "collapse_alignments", "format_alignment",
    "format_alignment_set", "normalize_embeddings", "pair_evalue", "rank",
    "rank_pairs", "read_metadata_table", "similarity_matrix",
    "transform_scores", "__version__",
]
