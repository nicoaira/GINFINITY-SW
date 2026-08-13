"""Delimited sequence/structure metadata input for alignment rendering."""
from __future__ import annotations

import csv
from pathlib import Path


def read_metadata_table(
    path: str | Path,
    *,
    identifier_column: str = "transcript_id",
    sequence_column: str = "sequence",
    structure_column: str = "secondary_structure",
    delimiter: str = "\t",
) -> dict[str, tuple[str, str]]:
    """Read identifier-to-sequence/structure metadata from a delimited table."""
    columns = (identifier_column, sequence_column, structure_column)
    if any(not column for column in columns):
        raise ValueError("column names cannot be empty")
    if len(set(columns)) != 3:
        raise ValueError("identifier, sequence, and structure columns must differ")
    if len(delimiter) != 1:
        raise ValueError("delimiter must be exactly one character")
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"empty metadata table: {path}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"duplicate column name in metadata table: {path}")
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"metadata table {path} is missing column(s): "
                + ", ".join(missing))
        result: dict[str, tuple[str, str]] = {}
        for row in reader:
            if None in row:
                raise ValueError(
                    f"metadata table {path} line {reader.line_num} has extra fields")
            identifier = row[identifier_column]
            sequence = row[sequence_column]
            structure = row[structure_column]
            if not all(isinstance(value, str)
                       for value in (identifier, sequence, structure)):
                raise ValueError(
                    f"missing metadata value at line {reader.line_num}")
            if identifier in result:
                raise ValueError(f"duplicate metadata identifier {identifier!r}")
            if not identifier:
                raise ValueError(
                    f"empty metadata identifier at line {reader.line_num}")
            if len(sequence) != len(structure):
                raise ValueError(f"metadata length mismatch for {identifier!r}")
            result[identifier] = (sequence, structure)
    if not result:
        raise ValueError(f"metadata table contains no records: {path}")
    return result


__all__ = ["read_metadata_table"]
