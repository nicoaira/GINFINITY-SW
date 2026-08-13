"""Command-line ranking and alignment for NPZ embedding archives."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .core import ScoringParameters, align, rank
from .formatting import format_alignment
from .metadata import read_metadata_table


def _parameters(path: Path) -> ScoringParameters:
    value = json.loads(path.read_text())
    value = value.get("scoring_parameters", value)
    return ScoringParameters(**value)


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
        params = _parameters(args.parameters)
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
                result = align(query, archive[args.target], params=params,
                               max_cells=args.max_cells)
                q_sequence, q_structure = metadata[args.query]
                t_sequence, t_structure = metadata[args.target]
                rendered = format_alignment(
                    result, q_sequence, q_structure,
                    t_sequence, t_structure)
                payload = {"query": args.query, "target": args.target,
                           **result.to_dict(), "formatted": rendered}
            else:
                candidates = ((identifier, archive[identifier])
                              for identifier in archive.files
                              if identifier != args.query)
                payload = {"query": args.query, "ranking": [
                    {"rank": index, "target": identifier, "score": score}
                    for index, (identifier, score) in enumerate(
                        rank(query, candidates, params=params,
                             max_cells=args.max_cells), start=1)]}
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
