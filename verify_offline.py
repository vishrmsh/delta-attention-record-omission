"""Verify the anonymous artifact without installing dependencies or running models."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_provenance_coverage(observed):
    """Bind every surviving scientific payload/source to its recorded treatment."""
    metadata = json.loads((ROOT / "provenance.json").read_text())
    followup = metadata["followup_provenance"]
    assert digest(ROOT / followup["path"]) == followup["sha256"], "follow-up provenance map checksum"
    records = metadata["copied_files"] + metadata.get("artifact_support_files", [])
    paths = [item["path"] for item in records]
    assert len(paths) == len(set(paths)), "duplicate source provenance path"
    for item in records:
        path = ROOT / item["path"]
        assert item["path"] in observed, f"source provenance file missing: {item['path']}"
        assert digest(path) == item["sha256"], f"source provenance checksum: {item['path']}"
        assert path.stat().st_size == item["bytes"], f"source provenance size: {item['path']}"
        if item.get("treatment") == "verbatim":
            assert item["origin_sha256"] == item["sha256"], f"verbatim source provenance: {item['path']}"
    from verify_followup import provenance
    covered = set(paths) | set(provenance())
    containers = {"code", "benchmarks", "reports", "artifacts", "followup"}
    scientific = {
        name for name, path in observed.items()
        if Path(name).parts[0] in containers and path.suffix in (".py", ".json", ".jsonl")
        and name != "followup/PROVENANCE.json"
    }
    assert scientific <= covered, f"scientific source provenance missing: {sorted(scientific - covered)}"
    print(f"PASS source provenance coverage: {len(scientific)} scientific data/source files")


def verify_controlled_separability():
    """Check the released aggregate scope and its rounded manuscript endpoints."""
    name = "benchmarks/kda_separability_controlled_summary_v1.json"
    evidence = json.loads((ROOT / name).read_text())
    assert set(evidence) == {"schema", "contains_source_text", "scope", "sources"}
    assert evidence["schema"] == "kda-controlled-separability-summary-v1"
    assert evidence["contains_source_text"] is False
    assert set(evidence["sources"]) == {"cds", "notes"}
    metadata = json.loads((ROOT / "provenance.json").read_text())
    origin = next(item for item in metadata["copied_files"] if item["path"] == name)
    rules = {"separable", "decay", "decay_corrected", "kda", "kda_corrected"}
    expected_rows = {
        "cds": ("2.1e-06", "0.008", "0.078", "6.4e-06", "0.116", "0.300", "0.089", "0.300"),
        "notes": ("1.4e-05", "0.015", "0.109", "7.0e-05", "0.122", "0.489", "0.081", "0.494"),
    }
    for source, entry in evidence["sources"].items():
        assert set(entry) == {"source_report_sha256", "protocol", "summary"}
        fingerprint = entry["source_report_sha256"]
        assert len(fingerprint) == 64 and all(c in "0123456789abcdef" for c in fingerprint)
        assert fingerprint == origin["source_report_sha256"][source]
        protocol, summary = entry["protocol"], entry["summary"]
        assert set(protocol) == {"kda_layers", "isolated_layers", "metric"}
        assert protocol["metric"] == "||dS_A - dS_B||_F / max(||dS_A||_F, ||dS_B||_F)"
        assert protocol["kda_layers"] == [i for i in range(26) if i % 4 != 3]
        assert protocol["isolated_layers"] == [0, 13, 25]
        assert set(summary) == {
            "end_to_end_rel", "surviving_contribution_norm_fraction", "isolated",
            "reference_vs_kernel_max", "n_configs", "n_kda_layers", "isolated_layers",
        }
        assert summary["n_configs"] == 6 and summary["n_kda_layers"] == 20
        assert summary["isolated_layers"] == protocol["isolated_layers"]
        assert set(summary["isolated"]) == rules
        groups = [summary[key] for key in ("end_to_end_rel", "surviving_contribution_norm_fraction")]
        groups.extend(summary["isolated"].values())
        for values in groups:
            assert set(values) == {"min", "median", "max"}
            assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())
            assert values["min"] <= values["median"] <= values["max"]
        assert 0 <= summary["reference_vs_kernel_max"] < 1e-6
        isolated = summary["isolated"]
        row = (f"{isolated['separable']['max']:.1e}",
               f"{isolated['decay']['min']:.3f}", f"{isolated['decay']['max']:.3f}",
               f"{isolated['decay_corrected']['max']:.1e}",
               f"{isolated['kda']['min']:.3f}", f"{isolated['kda']['max']:.3f}",
               f"{isolated['kda_corrected']['min']:.3f}", f"{isolated['kda_corrected']['max']:.3f}")
        assert row == expected_rows[source], f"controlled separability manuscript row: {source}"
    isolated = [entry["summary"]["isolated"] for entry in evidence["sources"].values()]
    assert f"{100 * max(row['decay']['max'] for row in isolated):.1f}" == "10.9"
    for rule, expected in (("kda", ("12", "49")), ("kda_corrected", ("8", "49"))):
        actual = (f"{100 * min(row[rule]['min'] for row in isolated):.0f}",
                  f"{100 * max(row[rule]['max'] for row in isolated):.0f}")
        assert actual == expected, f"controlled separability pooled endpoints: {rule}"
    print("PASS controlled separability aggregate schema, provenance and rounded manuscript endpoints")
    print("Controlled-data source reports and inference are outside this offline verification.")


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    expected = {entry["path"]: entry for entry in manifest["files"]}
    assert len(expected) == len(manifest["files"]), "duplicate manifest inventory path"
    observed = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.relative_to(ROOT).parts[0] != ".git"
        and path.relative_to(ROOT).as_posix() != "MANIFEST.json"
    }
    assert set(expected) == set(observed), "artifact file inventory differs"
    for name, path in observed.items():
        assert not path.is_symlink(), f"symlink: {name}"
        assert digest(path) == expected[name]["sha256"], f"checksum: {name}"
        assert path.stat().st_size == expected[name]["bytes"], f"size: {name}"
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
        if path.suffix in (".json", ".jsonl"):
            strings = [path.read_text()] if path.suffix == ".json" else path.read_text().splitlines()
            for value in strings:
                json.loads(value, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    print(f"PASS inventory, SHA-256, Python syntax and strict JSON: {len(observed)} files")
    verify_provenance_coverage(observed)
    verify_controlled_separability()

    environment = dict(os.environ, PYTHONPATH=str(ROOT / "code"), PYTHONDONTWRITEBYTECODE="1")
    tasks = [
        ("kimi_sv.summarize_forcing_sweep", "reports/kimi_forcing_sweep", "kda_forcing_sweep_summary_v1.json"),
        ("kimi_sv.summarize_masked_sampling", "reports/kimi_masked_sampling", "kda_masked_sampling_summary_v1.json"),
        ("recurrent_families.summarize", "reports/recurrent_families", "recurrent_families_summary_v1.json"),
    ]
    with tempfile.TemporaryDirectory(prefix="kda-offline-") as temporary:
        for module, reports, name in tasks:
            output = Path(temporary) / name
            completed = subprocess.run(
                [sys.executable, "-B", "-m", module, "--root", str(ROOT / reports), "--out", str(output)],
                cwd=ROOT, env=environment, capture_output=True, text=True,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            regenerated = json.loads(output.read_text())
            shipped = json.loads((ROOT / "benchmarks" / name).read_text())
            assert regenerated == shipped, f"regenerated summary differs: {name}"
            print(f"PASS complete JSON regeneration: {name}")
    qwen = json.loads((ROOT / "benchmarks/qwen35_replay_result_v3.json").read_text())
    assert qwen["manifest_sha256"] == digest(ROOT / "benchmarks/qwen35_replay_v3.json")
    assert qwen["implementation_sha256"] == digest(ROOT / "code/gemma_sv/qwen_replay_audit.py")
    assert all(qwen["checks"].values())
    print("PASS recorded Qwen manifest/implementation fingerprints and aggregate checks")
    from verify_followup import verify
    verify()
    print("No models, downloads, credentials or controlled data were used.")


if __name__ == "__main__":
    main()
