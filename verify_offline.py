"""Verify the anonymous artifact without installing dependencies or running models."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    expected = {entry["path"]: entry for entry in manifest["files"]}
    observed = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.relative_to(ROOT).parts[0] != ".git"
        and path.name != "MANIFEST.json"
    }
    assert set(expected) == set(observed), "artifact file inventory differs"
    for name, path in observed.items():
        assert not path.is_symlink(), f"symlink: {name}"
        assert digest(path) == expected[name]["sha256"], f"checksum: {name}"
        assert path.stat().st_size == expected[name]["bytes"], f"size: {name}"
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
        if path.suffix == ".json":
            json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    print(f"PASS inventory, SHA-256, Python syntax and strict JSON: {len(observed)} files")

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
        output = Path(temporary) / "kda_sweep_macros.tex"
        command = [sys.executable, "-B", "-m", "kimi_sv.build_kda_sweep_macros"]
        for flag, name in [
            ("sweep", "kda_forcing_sweep_summary_v1.json"),
            ("families", "recurrent_families_summary_v1.json"),
            ("sampling", "kda_masked_sampling_summary_v1.json"),
            ("mla", "kda_mla_logical_equality_v1.json"),
        ]:
            command += ["--" + flag, str(ROOT / "benchmarks" / name)]
        command += ["--out", str(output)]
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert output.read_bytes() == (ROOT / "expected/kda_sweep_macros.tex").read_bytes(), "paper macros differ"
        assert "PENDING" not in output.read_text(), "pending paper macro"
        print("PASS byte-identical numerical paper macros; no PENDING values")

    qwen = json.loads((ROOT / "benchmarks/qwen35_replay_result_v3.json").read_text())
    assert qwen["manifest_sha256"] == digest(ROOT / "benchmarks/qwen35_replay_v3.json")
    assert qwen["implementation_sha256"] == digest(ROOT / "code/gemma_sv/qwen_replay_audit.py")
    assert all(qwen["checks"].values())
    print("PASS recorded Qwen manifest/implementation fingerprints and aggregate checks")
    print("No models, downloads, credentials or controlled data were used.")


if __name__ == "__main__":
    main()
