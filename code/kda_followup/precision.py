"""Strict Kimi precision provenance and pinned original-weight acquisition.

Reading headers never loads a model or downloads weights. A config containing
``dtype=bfloat16`` is insufficient: the cached MLX 8-bit model has that field.
Only matching the official pinned shard SHA256 values establishes original
unquantized provenance; floating-point tensors alone could be dequantized.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

OFFICIAL_REPO = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
OFFICIAL_REVISION = "e1df551a447157d4658b573f9a695d57658590e9"
OFFICIAL_WEIGHT_BYTES = 98_248_224_120
METADATA_PATH = Path(__file__).with_name("kimi_bf16_upstream_metadata.json")
PRECISION_EVALUATION_INDICES = (0, 1, 2, 3, 6, 7, 8, 9)


def freeze_subset(manifest_path: Path, out: Path) -> dict[str, Any]:
    """Preselect four short and four long contexts without consulting results."""
    from .manifest import load_manifest, canonical_bytes

    manifest = load_manifest(manifest_path)
    ids = [f"kda-fu-evaluation-{i:03d}" for i in PRECISION_EVALUATION_INDICES]
    by_id = {r["record_id"]: r for r in manifest["records"]}
    chosen = [by_id[record_id] for record_id in ids]
    if Counter(r["suffix_class"] for r in chosen) != {"short": 4, "long": 4}:
        raise ValueError("precision subset must contain four short and four long contexts")
    plan = {"schema": "kda-precision-matched-subset-v1", "manifest_path": str(manifest_path.absolute()),
            "manifest_sha256": manifest["manifest_sha256"], "record_ids": ids,
            "selection_rule": "fixed evaluation indices 0,1,2,3,6,7,8,9, before examining evaluation output",
            "record_sha256": {r["record_id"]: hashlib.sha256(canonical_bytes(r)).hexdigest() for r in chosen},
            "conditions": manifest["protocol"]["conditions"], "suffix_strata": dict(Counter(r["suffix_class"] for r in chosen)),
            "bf16_repo": OFFICIAL_REPO, "bf16_revision": OFFICIAL_REVISION,
            "comparison": "same frozen record contexts, probe/target token ids, decoding, conditions; each precision versus its own never-stored reference and repeat-rebuild floor"}
    plan["plan_sha256"] = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    if out.exists() and json.loads(out.read_text()) != plan:
        raise ValueError("refusing to replace an existing precision subset")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def verify_tokenizers(plan_path: Path, quantized: Path, bf16: Path, out: Path) -> dict[str, Any]:
    """Load tokenizers only and compare every exact token in the matched suite."""
    from mlx_lm.utils import load_tokenizer
    from kimi_sv.protocol import encode, stop_ids
    from .manifest import load_manifest, canonical_bytes
    from .run import token_plan

    plan = json.loads(plan_path.read_text())
    manifest = load_manifest(plan["manifest_path"])
    if manifest["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("precision manifest changed after preselection")
    tokenizers = []
    for path in (quantized, bf16):
        config = json.loads((path / "config.json").read_text())
        tokenizers.append(load_tokenizer(path, {"trust_remote_code": True, "local_files_only": True}, eos_token_ids=config.get("eos_token_id")))
    records = {r["record_id"]: r for r in manifest["records"]}
    rows = []
    for record_id in plan["record_ids"]:
        record = records[record_id]
        if hashlib.sha256(canonical_bytes(record)).hexdigest() != plan["record_sha256"][record_id]:
            raise ValueError(f"precision record changed: {record_id}")
        encodings = []
        for tokenizer in tokenizers:
            prefix, target, suffix, forget, neutral = token_plan(tokenizer, record)
            items = {"prefix": [p[0].tolist() for p in prefix], "target": target[0].tolist(),
                     "suffix": suffix[0].tolist(), "forget": forget[0].tolist(),
                     "neutral": neutral[0].tolist(), "stop_ids": stop_ids(tokenizer)}
            for group in ("target_probes", "retained_probes", "continuations"):
                items[group] = [{"prompt": encode(tokenizer, "\n" + p["prompt"])[0].tolist(),
                                 "target": encode(tokenizer, p["target"])[0].tolist()} for p in record[group]]
            encodings.append(items)
        equal = encodings[0] == encodings[1]
        rows.append({"record_id": record_id, "all_token_ids_equal": equal,
                     "token_plan_sha256_8bit": hashlib.sha256(canonical_bytes(encodings[0])).hexdigest(),
                     "token_plan_sha256_bf16": hashlib.sha256(canonical_bytes(encodings[1])).hexdigest(),
                     "token_counts": {key: len(encodings[0][key]) for key in ("target", "suffix", "forget", "neutral")}})
    result = {"schema": "kda-precision-tokenizer-equality-v1", "plan_sha256": plan["plan_sha256"],
              "model_8bit": str(quantized), "model_bf16": str(bf16), "records": rows,
              "all_equal": all(row["all_token_ids_equal"] for row in rows)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_equal"]:
        raise ValueError("tokenization differs between precisions; do not claim token-matched comparison")
    return result


def summarize_pair(plan_path: Path, directory_8bit: Path, directory_bf16: Path,
                   token_check: Path, audit_8bit: Path, audit_bf16: Path, out: Path) -> dict[str, Any]:
    """Paired intervals for each precision against its own omission reference."""
    import numpy as np
    from .analyze import _metrics, _validate_result, _normalize_greedy, paired_mean_interval
    from .manifest import load_manifest
    from .maintenance_report import wilson

    plan = json.loads(plan_path.read_text())
    manifest = load_manifest(plan["manifest_path"])
    token_audit = json.loads(token_check.read_text())
    if not token_audit["all_equal"] or token_audit["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("paired precision analysis needs successful matching tokenizer audit")
    audits = {"8bit": json.loads(audit_8bit.read_text()), "bf16": json.loads(audit_bf16.read_text())}
    for precision, audit in audits.items():
        require_precision(audit, precision)
    derived = audits["8bit"].get("kind") == "derived_in_memory_quantization"
    if derived:
        source = audits["8bit"]["source_original_bf16_audit"]
        if source["official_weight_checks"] != audits["bf16"]["official_weight_checks"] or source["config_sha256"] != audits["bf16"]["config_sha256"]:
            raise ValueError("derived 8-bit model and BF16 model do not share the verified original master")
    manifest_records = {r["record_id"]: r for r in manifest["records"]}
    results = {}
    configurations = {}
    for precision, directory in (("8bit", directory_8bit), ("bf16", directory_bf16)):
        config = json.loads((directory / "run_config.json").read_text())
        configurations[precision] = config
        if config["tiny"] or config["manifest_sha256"] != plan["manifest_sha256"]:
            raise ValueError("precision analysis refuses tiny or mismatched-manifest runs")
        config_audit = inspect_snapshot(Path(config.get("source_model_path", config["model_id"])))
        expected_config = audits[precision].get("source_snapshot_config_sha256", audits[precision]["config_sha256"])
        expected_bytes = audits[precision].get("source_weight_bytes", audits[precision].get("weight_bytes"))
        if config_audit["config_sha256"] != expected_config or config_audit["weight_bytes"] != expected_bytes:
            raise ValueError("run model metadata differs from precision provenance audit")
        if precision == "8bit" and derived and (config.get("model_variant") != "derived_in_memory_quantization" or config["model_id"] != audits[precision]["model_id"]):
            raise ValueError("run configuration lacks the derived quantization provenance")
        results[precision] = []
        for record_id in plan["record_ids"]:
            row = json.loads((directory / f"{record_id}.json").read_text())
            if row["status"] != "complete":
                raise ValueError(f"incomplete {precision} result: {record_id}")
            if row.get("tiny") or row["model_id"] != config["model_id"]:
                raise ValueError("precision row differs from real run model configuration")
            _validate_result(row, manifest_records[record_id], manifest)
            results[precision].append(_normalize_greedy(row, manifest_records[record_id]))
    for key in ("runtime", "max_new_tokens"):
        if configurations["8bit"][key] != configurations["bf16"][key]:
            raise ValueError(f"precision run configurations have unmatched {key}")
    if configurations["8bit"].get("evaluator_sha256", configurations["8bit"]["runner_sha256"]) != configurations["bf16"].get("evaluator_sha256", configurations["bf16"]["runner_sha256"]):
        raise ValueError("precision conditions were produced by different main evaluator versions")
    if any(a["tokens"] != b["tokens"] for a, b in zip(results["8bit"], results["bf16"])):
        raise ValueError("observed input token plans differ across precision runs")
    seed = manifest["protocol"]["bootstrap"]["seed"]
    n = len(plan["record_ids"])
    precision_summary = {}
    deltas = {}
    for precision, rows in results.items():
        extracted = {condition: [_metrics(row["conditions"][condition]) for row in rows] for condition in plan["conditions"]}
        summaries = {}
        absolute = {}
        answer_rates = {}
        deltas[precision] = {}
        for condition, values in extracted.items():
            summaries[condition] = {}
            absolute[condition] = {}
            deltas[precision][condition] = {}
            for metric in values[0]:
                data = np.asarray([v[metric] for v in values], dtype=float)
                ref = np.asarray([v[metric] for v in extracted["never_stored"]], dtype=float)
                deltas[precision][condition][metric] = data - ref
                summaries[condition][metric] = paired_mean_interval(data, ref, seed=seed)
                absolute[condition][metric] = paired_mean_interval(data, np.zeros(n), seed=seed)
            answer_rates[condition] = {"target_exact": wilson(sum(v["target_direct_exact"] == 1 for v in values), n),
                                       "target_contains": wilson(sum(v["target_direct_contains"] == 1 for v in values), n),
                                       "all_three_retained_exact": wilson(sum(v["retained_exact"] == 1 for v in values), n),
                                       "all_three_retained_contain": wilson(sum(v["retained_contains"] == 1 for v in values), n)}
        precision_summary[precision] = {"conditions_paired_difference_vs_own_never": summaries,
            "absolute_metrics": absolute, "answer_rates_wilson_by_context": answer_rates,
            "qualified_records": sum(r["qualification"]["qualified"] for r in rows),
            "qualification_wilson": wilson(sum(r["qualification"]["qualified"] for r in rows), n),
            "replay_complete_state_and_logits_equal": sum(r["state_checks"]["replay_vs_never"]["all_equal"] for r in rows),
            "replay_state_max_abs_diff": max(r["state_checks"]["replay_vs_never"]["max_abs_diff"] for r in rows),
            "replay_logits_max_abs_diff": max(r["state_checks"]["replay_vs_never"]["logits"]["max_abs_diff"] for r in rows),
            "repeat_numerical_floor_complete_state_and_logits_equal": sum(r["state_checks"]["repeat_never_vs_never"]["all_equal"] for r in rows),
            "transport_residual_relative_to_present": [r["state_checks"]["corrections"]["transport"]["global_relative_to_present_difference"] for r in rows]}
    interactions = {condition: {metric: paired_mean_interval(delta, deltas["8bit"][condition][metric], seed=seed)
                               for metric, delta in metrics.items()} for condition, metrics in deltas["bf16"].items()}
    result = {"schema": "kda-matched-precision-summary-v1", "status": "complete", "plan_sha256": plan["plan_sha256"],
              "comparison_basis": "same_verified_original_master_with_in_memory_quantization" if derived else "separately_pinned_released_artifacts_nominally_same_base",
              "8bit_variant": "derived_in_memory_quantization" if derived else "released_community_8bit",
              "source_directories": {"8bit": str(directory_8bit.absolute()), "bf16": str(directory_bf16.absolute())},
              "provenance_audits": {"8bit": str(audit_8bit.absolute()), "bf16": str(audit_bf16.absolute())},
              "record_ids": plan["record_ids"], "n_records": n, "all_records_included_regardless_of_qualification": True,
              "bootstrap_unit": "whole context paired across condition and precision", "bootstrap_replicates": 2000,
              "by_precision": precision_summary, "difference_of_own_reference_deltas_bf16_minus_8bit": interactions,
              "limitations": "Eight preselected synthetic contexts provide a small matched precision control for these artifacts and runtime; no general quantization or population-equivalence claim follows. Utility consists of three retained synthetic records and two continuations per context from a fixed six-passage bank, not a broad language benchmark. A zero observed replay difference is not a universal guarantee.",
              "lineage_note": "The 8-bit variant was quantized in memory from the exact SHA-verified BF16 master, using the recorded MLX recipe." if derived else "The community model card names the same base repository but does not pin the original master revision. This secondary comparison cannot attribute differences causally to quantization alone."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def official_metadata() -> dict[str, Any]:
    metadata = json.loads(METADATA_PATH.read_text())
    assert metadata["id"] == OFFICIAL_REPO
    assert metadata["sha"] == OFFICIAL_REVISION
    weights = [f for f in metadata["siblings"] if f["rfilename"].endswith(".safetensors")]
    assert len(weights) == 20
    assert sum(f["size"] for f in weights) == OFFICIAL_WEIGHT_BYTES
    return metadata


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 100_000_000:
            raise ValueError(f"invalid safetensors header size: {path}")
        header = json.loads(stream.read(size))
    return header


def inspect_snapshot(path: Path, *, verify_official: bool = False) -> dict[str, Any]:
    path = Path(path).expanduser().absolute()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"local snapshot lacks config.json: {path}")
    config = json.loads(config_path.read_text())
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        expected = sorted(set(index["weight_map"].values()))
        files = [path / name for name in expected]
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files or any(not f.is_file() for f in files):
        raise ValueError("snapshot has missing or no weight shards")
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    quantized_names = []
    for shard in files:
        for name, meta in read_header(shard).items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            counts[dtype] += 1
            sizes[dtype] += meta["data_offsets"][1] - meta["data_offsets"][0]
            samples.setdefault(dtype, [])
            if len(samples[dtype]) < 3:
                samples[dtype].append(name)
            if name.endswith(".scales") or (name.endswith(".weight") and dtype in {"U32", "I32", "U8", "I8"}):
                quantized_names.append(name)
    quant = config.get("quantization") or config.get("quantization_config")
    floating = bool(counts.get("BF16")) and set(counts) <= {"BF16", "F32"}
    out = {
        "path": str(path), "snapshot_revision_if_hf_cache": path.name if path.parent.name == "snapshots" else None,
        "config_sha256": sha256_file(config_path), "model_type": config.get("model_type"),
        "config_dtype": config.get("dtype", config.get("torch_dtype")),
        "quantization": quant, "weight_files": len(files),
        "weight_bytes": sum(f.stat().st_size for f in files),
        "tensor_dtype_counts": dict(counts), "tensor_dtype_bytes": dict(sizes),
        "tensor_examples": samples, "quantization_tensor_count": len(quantized_names),
        "bf16_tensor_format": floating and not quant and not quantized_names,
        "original_bf16_provenance": "unverified",
        "source": f"https://huggingface.co/{OFFICIAL_REPO}/tree/{OFFICIAL_REVISION}",
    }
    if verify_official:
        official_files = [f for f in official_metadata()["siblings"] if f["rfilename"].endswith(".safetensors")]
        if set(f.name for f in files) != set(f["rfilename"] for f in official_files):
            raise ValueError("shard inventory differs from the pinned original bf16 release")
        checks = []
        for expected_file in official_files:
            local = path / expected_file["rfilename"]
            if local.stat().st_size != expected_file["size"]:
                raise ValueError(f"official shard size mismatch: {local.name}")
            checksum = sha256_file(local)
            if checksum != expected_file["lfs"]["sha256"]:
                raise ValueError(f"official shard SHA256 mismatch: {local.name}")
            checks.append({"file": local.name, "bytes": local.stat().st_size, "sha256": checksum})
            print(f"verified original shard {len(checks)}/{len(official_files)}: {local.name}", flush=True)
        if not out["bf16_tensor_format"]:
            raise ValueError("official bf16 control contains unexpected quantization or tensor dtypes")
        out["original_bf16_provenance"] = "all_pinned_official_weight_shas_verified"
        out["official_weight_checks"] = checks
    return out


def require_precision(audit: dict[str, Any], expected: str) -> None:
    if audit["model_type"] != "kimi_linear":
        raise ValueError("precision control requires Kimi Linear weights")
    if expected == "bf16":
        if not audit["bf16_tensor_format"] or audit["original_bf16_provenance"] != "all_pinned_official_weight_shas_verified":
            raise ValueError("bf16 control requires original hash-verified weights, not dtype metadata or dequantized 8-bit weights")
    elif expected == "8bit":
        if not audit.get("quantization") or audit["quantization"].get("bits") != 8 or not audit["quantization_tensor_count"]:
            raise ValueError("8bit control requires actual packed 8-bit quantized weights")
    else:
        raise ValueError(f"unknown precision {expected}")


def download_official(cache_dir: Path, out: Path, workers: int = 2) -> dict[str, Any]:
    """Download only pinned model/tokenizer files, then verify every weight SHA.

    This is the one operation in this module that accesses the network. It is
    opt-in via the ``download`` subcommand and must be launched with filesystem
    authorization for its explicit cache destination.
    """
    from huggingface_hub import snapshot_download

    metadata = official_metadata()
    allowed = [f["rfilename"] for f in metadata["siblings"] if "/" not in f["rfilename"] and f["rfilename"] != ".gitattributes"]
    print(json.dumps({"repo": OFFICIAL_REPO, "revision": OFFICIAL_REVISION,
                      "cache_dir": str(cache_dir), "weight_bytes": OFFICIAL_WEIGHT_BYTES,
                      "files": allowed}), flush=True)
    local = Path(snapshot_download(repo_id=OFFICIAL_REPO, revision=OFFICIAL_REVISION,
                                 cache_dir=str(cache_dir), allow_patterns=allowed,
                                 max_workers=workers))
    for expected in metadata["siblings"]:
        if expected["rfilename"] in allowed and (local / expected["rfilename"]).stat().st_size != expected["size"]:
            raise ValueError(f"pinned file size mismatch: {expected['rfilename']}")
    audit = inspect_snapshot(local, verify_official=True)
    require_precision(audit, "bf16")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"original bf16 snapshot verified: {local}", flush=True)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect")
    inspect.add_argument("--model", type=Path, required=True)
    inspect.add_argument("--expected", choices=("bf16", "8bit"))
    inspect.add_argument("--verify-official", action="store_true")
    inspect.add_argument("--out", type=Path, required=True)
    download = subs.add_parser("download")
    download.add_argument("--cache-dir", type=Path, required=True)
    download.add_argument("--out", type=Path, required=True)
    download.add_argument("--workers", type=int, default=2)
    subset = subs.add_parser("freeze-subset")
    subset.add_argument("--manifest", type=Path, required=True)
    subset.add_argument("--out", type=Path, required=True)
    tokenizer = subs.add_parser("verify-tokenizers")
    tokenizer.add_argument("--plan", type=Path, required=True)
    tokenizer.add_argument("--model-8bit", type=Path, required=True)
    tokenizer.add_argument("--model-bf16", type=Path, required=True)
    tokenizer.add_argument("--out", type=Path, required=True)
    summarize = subs.add_parser("summarize-pair")
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--results-8bit", type=Path, required=True)
    summarize.add_argument("--results-bf16", type=Path, required=True)
    summarize.add_argument("--token-check", type=Path, required=True)
    summarize.add_argument("--audit-8bit", type=Path, required=True)
    summarize.add_argument("--audit-bf16", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        download_official(args.cache_dir, args.out, args.workers)
    elif args.command == "freeze-subset":
        print(json.dumps(freeze_subset(args.manifest, args.out), indent=2))
    elif args.command == "verify-tokenizers":
        print(json.dumps(verify_tokenizers(args.plan, args.model_8bit, args.model_bf16, args.out), indent=2))
    elif args.command == "summarize-pair":
        result = summarize_pair(args.plan, args.results_8bit, args.results_bf16, args.token_check,
                                args.audit_8bit, args.audit_bf16, args.out)
        print(json.dumps({"status": result["status"], "n_records": result["n_records"], "out": str(args.out)}))
    else:
        audit = inspect_snapshot(args.model, verify_official=args.verify_official)
        if args.expected:
            require_precision(audit, args.expected)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
