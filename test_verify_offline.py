"""End-to-end inventory regression checks using isolated artifact copies."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


class OfflineVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kda-verifier-test-")
        self.addCleanup(self.temporary.cleanup)
        self.artifact = Path(self.temporary.name) / "artifact"
        shutil.copytree(ROOT, self.artifact, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    def verify(self, succeeds, message=None):
        result = subprocess.run(
            [sys.executable, "-B", "verify_offline.py"], cwd=self.artifact,
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode == 0, succeeds, output)
        if message:
            self.assertIn(message, output)

    def test_extracted_archive(self):
        self.verify(True, "PASS recorded Qwen")

    def test_git_directory(self):
        metadata = self.artifact / ".git" / "objects"
        metadata.mkdir(parents=True)
        (metadata / "unlisted-object").write_bytes(b"git metadata")
        self.verify(True, "PASS recorded Qwen")

    def test_worktree_git_file(self):
        (self.artifact / ".git").write_text("gitdir: ../repository/worktrees/review\n")
        self.verify(True, "PASS recorded Qwen")

    def test_unexpected_hidden_file(self):
        (self.artifact / ".unexpected").write_text("unlisted artifact content\n")
        self.verify(False, "artifact file inventory differs")

    def test_nested_git_is_not_exempt(self):
        metadata = self.artifact / "code" / ".git"
        metadata.mkdir()
        (metadata / "unexpected").write_text("unlisted nested content\n")
        self.verify(False, "artifact file inventory differs")

    def test_missing_payload(self):
        (self.artifact / "expected" / "kda_sweep_macros.tex").unlink()
        self.verify(False, "artifact file inventory differs")

    def test_changed_payload(self):
        payload = self.artifact / "expected" / "kda_sweep_macros.tex"
        payload.write_bytes(payload.read_bytes() + b"\n% tampered\n")
        self.verify(False, "checksum: expected/kda_sweep_macros.tex")


    def test_nested_manifest_is_not_exempt(self):
        (self.artifact / "followup" / "MANIFEST.json").write_text("{}\n")
        self.verify(False, "artifact file inventory differs")

    def test_followup_tamper_detected_after_inventory_rehash(self):
        relative = "followup/raw/kimi8_v2/kda-fu-evaluation-000.json"
        payload = self.artifact / relative
        row = json.loads(payload.read_text())
        row["conditions"]["present"]["target_probes"][0]["greedy_text"] = "tampered synthetic answer"
        payload.write_text(json.dumps(row))
        manifest_path = self.artifact / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            if item["path"] == relative:
                item["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
                item["bytes"] = payload.stat().st_size
        manifest_path.write_text(json.dumps(manifest))
        self.verify(False, "follow-up release checksum")

    def test_duplicate_manifest_path_rejected(self):
        path = self.artifact / "MANIFEST.json"
        manifest = json.loads(path.read_text())
        manifest["files"].append(manifest["files"][0])
        path.write_text(json.dumps(manifest))
        self.verify(False, "duplicate manifest inventory path")


if __name__ == "__main__":
    unittest.main()
