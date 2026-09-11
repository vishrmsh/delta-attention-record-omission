"""Evaluate an 8-bit model quantized in memory from the verified BF16 master.

No weight files are saved. The runtime inventories actual packed tensors and
records the original shard provenance and exact MLX quantization recipe before
evaluating the frozen subset through the unchanged main evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import time
from typing import Any


def quantize_with_provenance(model: Any, config: dict[str, Any], source_audit: dict[str, Any] | None) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import quantize_model
    from mlx_lm.models import kimi_linear
    from .manifest import canonical_bytes

    t0 = time.perf_counter()
    # Upstream's wrapper checks to_quantized support and group divisibility
    # before delegating to model.quant_predicate. Passing that predicate directly
    # to nn.quantize would wrongly select normalization and container modules.
    model, quantized_config = quantize_model(model, config, group_size=64, bits=8,
                                           mode="affine", quant_predicate=model.quant_predicate)
    mx.eval(model.parameters())
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    packed = []
    scales = []
    for name, value in tree_flatten(model.parameters()):
        counts[str(value.dtype)] += 1
        sizes[str(value.dtype)] += value.nbytes
        if name.endswith(".weight") and value.dtype == mx.uint32:
            packed.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes})
        if name.endswith(".scales"):
            scales.append(name)
    if not packed or not scales:
        raise ValueError("quantization produced no actual packed weight/scales tensors")
    sources = {name: Path(inspect.getsourcefile(obj)) for name, obj in
               (("mlx_nn_quantize", nn.quantize), ("mlx_lm_quantize_model", quantize_model), ("kimi_linear_model", kimi_linear.Model))}
    audit = {"schema": "kda-derived-8bit-provenance-v1", "kind": "derived_in_memory_quantization",
             "model_type": "kimi_linear", "config_dtype": config.get("dtype", "tiny-random"),
             "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
             "quantization_recipe": "mlx_lm.utils.quantize_model with group_size=64, bits=8, mode=affine, model.quant_predicate; wrapper calls mlx.nn.quantize after support/divisibility checks",
             "quantized_config": quantized_config,
             "config_sha256": hashlib.sha256(canonical_bytes(quantized_config)).hexdigest(),
             "quantization_tensor_count": len(packed) + len(scales), "packed_weight_tensor_count": len(packed),
             "tensor_dtype_counts": dict(counts), "tensor_dtype_bytes": dict(sizes),
             "live_parameter_bytes": sum(sizes.values()), "packed_weights": packed,
             "bf16_tensor_format": False, "original_bf16_provenance": "not_applicable_quantized_variant",
             "source_original_bf16_audit": source_audit,
             "source_snapshot_config_sha256": None if source_audit is None else source_audit["config_sha256"],
             "source_weight_bytes": None if source_audit is None else source_audit["weight_bytes"],
             "weight_files_saved": False, "quantization_seconds": time.perf_counter() - t0,
             "runtime": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
             "implementation_files": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sources.items()}}
    return model, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-provenance", type=Path, required=True)
    parser.add_argument("--floor-out-dir", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args(argv)
    from .manifest import load_manifest, canonical_bytes
    from .precision import inspect_snapshot, require_precision
    manifest = load_manifest(args.manifest)
    plan = json.loads(args.plan.read_text())
    if plan["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("source manifest differs from frozen precision plan")
    by_id = {r["record_id"]: r for r in manifest["records"]}
    rows = [by_id[record_id] for record_id in plan["record_ids"]]
    for row in rows:
        if hashlib.sha256(canonical_bytes(row)).hexdigest() != plan["record_sha256"][row["record_id"]]:
            raise ValueError("precision record changed after preselection")
    if args.limit:
        rows = rows[:args.limit]
    if not args.tiny and args.max_new_tokens != manifest["protocol"]["decoding"]["max_new_tokens"]:
        raise ValueError("real runs must preserve the frozen decoding budget")
    source_audit = None
    if not args.tiny:
        if args.source_provenance is None:
            raise ValueError("original BF16 source provenance is required")
        source_audit = json.loads(args.source_provenance.read_text())
        require_precision(source_audit, "bf16")
        current = inspect_snapshot(Path(args.model))
        if current["config_sha256"] != source_audit["config_sha256"] or current["weight_bytes"] != source_audit["weight_bytes"]:
            raise ValueError("source snapshot differs from original BF16 provenance")
    import mlx.core as mx
    from kimi_sv.protocol import load_model
    from kimi_sv.mla_mask import install_mla_mask
    from .run import evaluate_record
    from .transport_floor import diagnostic_record
    model_id = args.model + "::derived-8bit-group64-affine"
    config = {"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"], "plan_sha256": plan["plan_sha256"],
              "model_id": model_id, "source_model_path": args.model, "model_variant": "derived_in_memory_quantization",
              "tiny": args.tiny, "max_new_tokens": args.max_new_tokens,
              "runtime": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
              "platform": platform.platform(), "device": mx.device_info(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(Path(inspect.getsourcefile(evaluate_record)).read_bytes()).hexdigest(),
              "floor_evaluator_sha256": hashlib.sha256(Path(inspect.getsourcefile(diagnostic_record)).read_bytes()).hexdigest(),
              "source_provenance_sha256": None if args.source_provenance is None else hashlib.sha256(args.source_provenance.read_bytes()).hexdigest()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("refusing to resume a different derived precision configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    pending = [r for r in rows if not (args.out_dir / f"{r['record_id']}.json").exists()]
    pending_floor = [] if args.floor_out_dir is None else [r for r in rows if not (args.floor_out_dir / f"{r['record_id']}.json").exists()]
    if not pending and not pending_floor:
        print("No pending derived-8bit records", flush=True)
        return 0
    mx.set_cache_limit(2 * 1024**3)
    model, tokenizer, _ = load_model(args.model, tiny=args.tiny, lazy=True)
    source_config = {"model_type": "kimi_linear"} if args.tiny else json.loads((Path(args.model) / "config.json").read_text())
    model, provenance = quantize_with_provenance(model, source_config, source_audit)
    provenance.update(model_id=model_id, source_model_path=args.model, tiny_plumbing_only=args.tiny,
                      runner_sha256=config["runner_sha256"], evaluator_sha256=config["evaluator_sha256"])
    if not args.tiny:
        require_precision(provenance, "8bit")
    args.out_provenance.parent.mkdir(parents=True, exist_ok=True)
    args.out_provenance.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    install_mla_mask(model)
    print(f"Derived8bit: {provenance['packed_weight_tensor_count']} packed tensors, {provenance['live_parameter_bytes']} live bytes; {len(pending)} records", flush=True)
    for row in pending:
        started = time.perf_counter()
        result = evaluate_record(model, tokenizer, row, tiny=args.tiny, max_new_tokens=args.max_new_tokens)
        result.update(schema_version=1, manifest_sha256=manifest["manifest_sha256"], record_id=row["record_id"],
                      model_id=model_id, tiny=args.tiny, split=row["split"], seconds=time.perf_counter() - started,
                      model_variant="derived_in_memory_quantization")
        with (args.out_dir / f"{row['record_id']}.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"{row['record_id']}: {result['seconds']:.1f}s replay_equal={result['state_checks']['replay_vs_never']['all_equal']}", flush=True)
        mx.clear_cache()
    if args.floor_out_dir is not None:
        args.floor_out_dir.mkdir(parents=True, exist_ok=True)
        floor_config = {"schema": "kda-transport-arithmetic-floor-v1", "precision": "8bit", "model": model_id,
                        "model_variant": "derived_in_memory_quantization", "tiny_plumbing_only": args.tiny,
                        "manifest_sha256": manifest["manifest_sha256"], "plan_sha256": plan["plan_sha256"],
                        "record_ids": [r["record_id"] for r in rows], "runner_sha256": config["floor_evaluator_sha256"],
                        "precision_provenance_path": str(args.out_provenance.absolute()), "runtime": config["runtime"]}
        (args.floor_out_dir / "run_config.json").write_text(json.dumps(floor_config, indent=2) + "\n")
        for row in pending_floor:
            result = diagnostic_record(model, tokenizer, row, tiny=args.tiny)
            result.update(manifest_sha256=manifest["manifest_sha256"], precision="8bit", model_id=model_id,
                          model_variant="derived_in_memory_quantization")
            with (args.floor_out_dir / f"{row['record_id']}.json").open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.write("\n")
            print(f"{row['record_id']} arithmetic: residual/floor={result['global']['residual_to_max_empirical_floor_ratio']}; {result['seconds']:.1f}s", flush=True)
            mx.clear_cache()
        floors = [json.loads((args.floor_out_dir / f"{row['record_id']}.json").read_text()) for row in rows]
        ratios = [r["global"]["residual_to_reconstruction_error_ratio"] for r in floors]
        combined = [r["global"]["residual_to_max_empirical_floor_ratio"] for r in floors]
        summary = {"schema": "kda-transport-arithmetic-floor-summary-v1", "status": "complete", "precision": "8bit",
                   "model_variant": "derived_in_memory_quantization", "tiny_plumbing_only": args.tiny,
                   "n_records": len(floors), "record_ids": [r["record_id"] for r in floors],
                   "global_residual_to_reconstruction_error_ratios": ratios,
                   "global_residual_to_max_empirical_floor_ratios": combined,
                   "minimum_global_combined_floor_ratio": min((r for r in combined if r is not None), default=None),
                   "maximum_global_combined_floor_ratio": max((r for r in combined if r is not None), default=None),
                   "max_present_reconstruction_relative_error": max(r["global"]["present_reconstruction_relative_error"] for r in floors),
                   "max_omitted_reconstruction_relative_error": max(r["global"]["omitted_reconstruction_relative_error"] for r in floors)}
        (args.floor_out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
