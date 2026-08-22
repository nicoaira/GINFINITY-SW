"""Command-line ranking and alignment for NPZ embedding archives."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .core import EValueParameters, ScoringParameters, align_multiple, rank_pairs
from .formatting import format_alignment_set
from .metadata import read_metadata_table


def _parameters(path: Path) -> tuple[ScoringParameters, EValueParameters]:
    value = json.loads(path.read_text())
    scoring = value.get("scoring_parameters", value)
    if "scoring_parameters" not in value:
        scoring = {
            key: item for key, item in scoring.items()
            if key not in {"evalue_parameters", "evalue_lambda", "evalue_k"}
        }
    evalue = value.get("evalue_parameters", {})
    if not isinstance(evalue, dict):
        raise ValueError("evalue_parameters must be an object")
    lambda_value = evalue.get(
        "lambda",
        evalue.get("lambda_", value.get("evalue_lambda", 1.0)),
    )
    k = evalue.get("k", value.get("evalue_k", 1.0))
    return ScoringParameters(**scoring), EValueParameters(
        lambda_=lambda_value,
        k=k,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ginfinity-sw")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--target")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cells", type=int, default=16_777_216)
    parser.add_argument(
        "--max-alignments", type=int, default=16,
        help="maximum disjoint local HSPs to report per pair",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="minimum score for an additional HSP",
    )
    parser.add_argument(
        "--min-match-count", type=int, default=1,
        help="minimum ungapped columns for an additional HSP",
    )
    parser.add_argument(
        "--evalue-lambda", type=float,
        help="override the pair-level E-value lambda constant",
    )
    parser.add_argument(
        "--evalue-k", type=float,
        help="override the pair-level E-value K constant",
    )
    parser.add_argument("--id-column", default="transcript_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--structure-column", default="secondary_structure")
    parser.add_argument("--delimiter", default="\t")
    args = parser.parse_args(argv)
    try:
        metadata = read_metadata_table(
            args.metadata,
            identifier_column=args.id_column,
            sequence_column=args.sequence_column,
            structure_column=args.structure_column,
            delimiter=args.delimiter,
        )
        params, evalue_parameters = _parameters(args.parameters)
        if args.evalue_lambda is not None or args.evalue_k is not None:
            evalue_parameters = EValueParameters(
                lambda_=(
                    evalue_parameters.lambda_
                    if args.evalue_lambda is None else args.evalue_lambda
                ),
                k=(
                    evalue_parameters.k
                    if args.evalue_k is None else args.evalue_k
                ),
            )
        with np.load(args.embeddings) as archive:
            if set(archive.files) != set(metadata):
                missing_metadata = sorted(set(archive.files) - set(metadata))
                missing_embeddings = sorted(set(metadata) - set(archive.files))
                raise ValueError(
                    "embedding/metadata identifier mismatch; "
                    f"missing metadata={missing_metadata}, "
                    f"missing embeddings={missing_embeddings}")
            for identifier in archive.files:
                value = archive[identifier]
                if value.ndim != 2 or value.shape[0] != len(metadata[identifier][0]):
                    raise ValueError(
                        f"embedding shape does not match metadata for {identifier!r}")
            if args.query not in archive:
                raise ValueError(f"query {args.query!r} is absent from embeddings")
            query = np.asarray(archive[args.query])
            if args.target:
                if args.target not in archive:
                    raise ValueError(
                        f"target {args.target!r} is absent from embeddings")
                result = align_multiple(
                    query,
                    archive[args.target],
                    params=params,
                    max_alignments=args.max_alignments,
                    min_score=args.min_score,
                    min_match_count=args.min_match_count,
                    max_cells=args.max_cells,
                    evalue_parameters=evalue_parameters,
                )
                q_sequence, q_structure = metadata[args.query]
                t_sequence, t_structure = metadata[args.target]
                rendered = format_alignment_set(
                    result,
                    q_sequence,
                    q_structure,
                    t_sequence,
                    t_structure,
                )
                payload = {"query": args.query, "target": args.target,
                           **result.to_dict(), "formatted": rendered}
            else:
                candidates = ((identifier, archive[identifier])
                              for identifier in archive.files
                              if identifier != args.query)
                payload = {"query": args.query, "ranking": [
                    {
                        "rank": index,
                        "target": identifier,
                        "score": summary.max_score,
                        "total_score": summary.total_score,
                        "max_score": summary.max_score,
                        "evalue": summary.evalue,
                        "alignment_count": summary.alignment_count,
                        "match_count": summary.match_count,
                    }
                    for index, (identifier, summary) in enumerate(
                        rank_pairs(
                            query,
                            candidates,
                            params=params,
                            max_alignments=args.max_alignments,
                            min_score=args.min_score,
                            min_match_count=args.min_match_count,
                            max_cells=args.max_cells,
                            evalue_parameters=evalue_parameters,
                        ),
                        start=1,
                    )]}
        text = json.dumps(payload, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")
        return 0
    except Exception as error:
        print(f"ginfinity-sw: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
