"""Verify the anonymous artifact without installing dependencies or running models."""
from __future__ import annotations

import ast
import hashlib
import json
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
