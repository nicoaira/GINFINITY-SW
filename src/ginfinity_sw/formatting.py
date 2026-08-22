"""Human-readable alignment rendering."""
from __future__ import annotations

from .core import Alignment, AlignmentSet


def _state(character: str) -> int:
    return 0 if character == "(" else (2 if character == ")" else 1)


def _pair_table(structure: str) -> list[int]:
    partners = [-1] * len(structure)
    stack: list[int] = []
    for index, character in enumerate(structure):
        if character == "(":
            stack.append(index)
        elif character == ")":
            if not stack:
                raise ValueError("unmatched ')' in structure")
            opening = stack.pop()
            partners[opening] = index
            partners[index] = opening
    if stack:
        raise ValueError("unmatched '(' in structure")
    return partners


def _conserved_pair_symbols(
    alignment: Alignment,
    query_structure: str,
    target_structure: str,
) -> tuple[dict[int, str], int]:
    query_partners = _pair_table(query_structure)
    target_partners = _pair_table(target_structure)
    mapping = {
        query: target for query, target in alignment.columns
        if query >= 0 and target >= 0
    }
    markers: dict[int, str] = {}
    count = 0
    for opening, closing in enumerate(query_partners):
        if closing <= opening or opening not in mapping or closing not in mapping:
            continue
        target_opening = mapping[opening]
        target_closing = mapping[closing]
        if target_partners[target_opening] == target_closing:
            markers[opening] = "<"
            markers[closing] = ">"
            count += 1
    return markers, count


def format_alignment(alignment: Alignment, query_sequence: str,
                     query_structure: str, target_sequence: str,
                     target_structure: str, *, width: int = 60) -> str:
    """Render a six-line RNA alignment using 1-based inclusive coordinates."""
    if not alignment.columns:
        return "No positive-scoring local alignment."
    if len(query_sequence) != len(query_structure):
        raise ValueError("query sequence/structure length mismatch")
    if len(target_sequence) != len(target_structure):
        raise ValueError("target sequence/structure length mismatch")
    query_chars: list[str] = []
    query_states: list[str] = []
    target_chars: list[str] = []
    target_states: list[str] = []
    symbols: list[str] = []
    conserved_markers, conserved_pair_count = _conserved_pair_symbols(
        alignment, query_structure, target_structure)
    pair_symbols: list[str] = []
    base_matches = 0
    state_matches = 0
    ungapped = 0
    for query, target in alignment.columns:
        query_base = query_sequence[query] if query >= 0 else "-"
        query_struct = query_structure[query] if query >= 0 else "-"
        target_base = target_sequence[target] if target >= 0 else "-"
        target_struct = target_structure[target] if target >= 0 else "-"
        query_chars.append(query_base)
        query_states.append(query_struct)
        target_chars.append(target_base)
        target_states.append(target_struct)
        pair_symbols.append(conserved_markers.get(query, " "))
        if query < 0 or target < 0:
            symbols.append(" ")
            continue
        ungapped += 1
        base_match = query_base.upper().replace("T", "U") == target_base.upper().replace("T", "U")
        state_match = _state(query_struct) == _state(target_struct)
        base_matches += int(base_match)
        state_matches += int(state_match)
        if base_match and state_match:
            symbols.append("|")
        elif state_match and _state(query_struct) != 1:
            symbols.append("+")
        elif state_match:
            symbols.append("~")
        elif base_match:
            symbols.append(":")
        else:
            symbols.append(".")
    lines = [
        f"Score = {alignment.score:.6g}",
        (f"Aligned columns = {len(alignment.columns)} ({ungapped} ungapped); "
         f"base identity = {100.0 * base_matches / max(ungapped, 1):.1f}%; "
         f"structure identity = "
         f"{100.0 * state_matches / max(ungapped, 1):.1f}%; "
         f"conserved pairs = {conserved_pair_count}"),
        (f"Query span = {alignment.query_span}; Target span = "
         f"{alignment.target_span} (0-based, half-open)"),
        "",
    ]
    query_position = alignment.query_span[0]
    target_position = alignment.target_span[0]
    for start in range(0, len(query_chars), width):
        stop = min(start + width, len(query_chars))
        query_chunk = query_chars[start:stop]
        target_chunk = target_chars[start:stop]
        query_start = query_position + 1
        target_start = target_position + 1
        query_position += sum(base != "-" for base in query_chunk)
        target_position += sum(base != "-" for base in target_chunk)
        lines.extend([
            f"Query {query_start:>5} {''.join(query_chunk)} {query_position}",
            f"            {''.join(query_states[start:stop])}",
            f"            {''.join(pair_symbols[start:stop])}",
            f"            {''.join(symbols[start:stop])}",
            f"            {''.join(target_states[start:stop])}",
            f"Sbjct {target_start:>5} {''.join(target_chunk)} {target_position}",
            "",
        ])
    return "\n".join(lines).rstrip()


def format_alignment_set(alignment_set: AlignmentSet, query_sequence: str,
                         query_structure: str, target_sequence: str,
                         target_structure: str, *, width: int = 60) -> str:
    """Render one BLAST-style pair summary followed by all HSP blocks.

    The individual tracebacks stay separate because disjoint local HSPs do
    not form one valid Smith--Waterman path.  They are nevertheless presented
    under one query-target result, with aggregate and maximum scores in the
    same summary.
    """
    if not isinstance(alignment_set, AlignmentSet):
        raise TypeError("alignment_set must be AlignmentSet")
    summary = [
        f"Total score = {alignment_set.total_score:.6g}",
        f"Max score = {alignment_set.max_score:.6g}",
        f"E-value = {alignment_set.evalue:.6g}",
        (f"Alignments = {alignment_set.alignment_count}; "
         f"search space = {alignment_set.search_space}"),
    ]
    if not alignment_set.alignments:
        summary.append("No positive-scoring local alignments.")
        return "\n".join(summary)
    blocks = []
    for index, alignment in enumerate(alignment_set.alignments, start=1):
        blocks.append(
            f"HSP {index}\n" + format_alignment(
                alignment,
                query_sequence,
                query_structure,
                target_sequence,
                target_structure,
                width=width,
            )
        )
    return "\n".join(summary) + "\n\n" + "\n\n".join(blocks)
