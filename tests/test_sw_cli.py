import json

import numpy as np

from ginfinity_sw.cli import main


def test_cli_aligns_and_ranks(tmp_path):
    embeddings = tmp_path / "embeddings.npz"
    query = np.eye(4)
    np.savez(embeddings, query=query, mate=query, other=-query)
    metadata = tmp_path / "molecules.tsv"
    metadata.write_text(
        "transcript_id\tsequence\tsecondary_structure\n"
        "query\tACGU\t....\nmate\tACGU\t....\nother\tUGCA\t....\n")
    parameters = tmp_path / "alignment.json"
    parameters.write_text(json.dumps({
        "mu": 0.2, "sigma": 1.0, "gamma": 5.0,
        "score_min": -4.0, "score_max": 8.0,
        "gap_open": 6.0, "gap_extend": 1.0, "score_offset": 0.0,
    }))
    output = tmp_path / "ranking.json"
    assert main([
        "--embeddings", str(embeddings), "--metadata", str(metadata),
        "--parameters", str(parameters), "--query", "query",
        "--output", str(output),
    ]) == 0
    got = json.loads(output.read_text())
    assert got["ranking"][0]["target"] == "mate"

    output = tmp_path / "alignment-output.json"
    assert main([
        "--embeddings", str(embeddings), "--metadata", str(metadata),
        "--parameters", str(parameters), "--query", "query",
        "--target", "mate", "--output", str(output),
    ]) == 0
    got = json.loads(output.read_text())
    assert got["match_count"] == 4
    assert "Query" in got["formatted"]


def test_cli_rejects_metadata_archive_drift(tmp_path, capsys):
    embeddings = tmp_path / "embeddings.npz"
    np.savez(embeddings, only=np.eye(4))
    metadata = tmp_path / "molecules.tsv"
    metadata.write_text(
        "transcript_id\tsequence\tsecondary_structure\n"
        "different\tACGU\t....\n")
    parameters = tmp_path / "alignment.json"
    parameters.write_text(json.dumps({
        "mu": 0.2, "sigma": 1.0, "gamma": 5.0,
        "score_min": -4.0, "score_max": 8.0,
        "gap_open": 6.0, "gap_extend": 1.0, "score_offset": 0.0,
    }))
    assert main([
        "--embeddings", str(embeddings), "--metadata", str(metadata),
        "--parameters", str(parameters), "--query", "only",
    ]) == 2
    assert "identifier mismatch" in capsys.readouterr().err


def test_cli_accepts_configurable_metadata_columns(tmp_path):
    embeddings = tmp_path / "embeddings.npz"
    query = np.eye(4)
    np.savez(embeddings, query=query, mate=query)
    metadata = tmp_path / "structures.csv"
    metadata.write_text(
        "name,bases,fold,source\n"
        "query,ACGU,....,example\n"
        "mate,ACGU,....,example\n")
    parameters = tmp_path / "alignment.json"
    parameters.write_text(json.dumps({
        "mu": 0.2, "sigma": 1.0, "gamma": 5.0,
        "score_min": -4.0, "score_max": 8.0,
        "gap_open": 6.0, "gap_extend": 1.0, "score_offset": 0.0,
    }))
    output = tmp_path / "alignment-output.json"
    assert main([
        "--embeddings", str(embeddings), "--metadata", str(metadata),
        "--parameters", str(parameters), "--query", "query",
        "--target", "mate", "--output", str(output),
        "--id-column", "name", "--sequence-column", "bases",
        "--structure-column", "fold", "--delimiter", ",",
    ]) == 0
    assert json.loads(output.read_text())["match_count"] == 4
