"""Native-versus-serial KDA arithmetic diagnostics for a fixed precision subset.

Capture each branch's actual kernel inputs, then independently reconstruct the
present and omitted recurrent states in fp32. These discrepancies measure the
arithmetic reconstruction error for the transport calculation. They complement
the repeat-never control, which measures only deterministic repeatability.
This does not claim a rigorous bound on every possible floating-point error.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from typing import Any


def _global(per_layer: list[float]) -> float:
    return math.sqrt(sum(value * value for value in per_layer))


def _ratio(numerator: float, denominator: float) -> float | None:
    # null means no finite ratio can be estimated from a zero observed floor.
    return numerator / denominator if denominator > 0.0 else None


def diagnostic_record(model: Any, tokenizer: Any, record: dict[str, Any], *, tiny: bool = False) -> dict[str, Any]:
    import mlx.core as mx
    from kimi_sv.eval_forcing_sweep import AllLayerTap, _step_transition, _step_write, layer_norms, run_context
    from kimi_sv.state import classify_layers
    from .run import token_plan

    started = time.perf_counter()
    kda, _ = classify_layers(model)
    prefix, target, suffix, _, _ = token_plan(tokenizer, record, tiny=tiny)
    n = int(suffix.shape[1])
    tap = AllLayerTap(len(kda))
    tap.install()
    try:
        present = run_context(model, kda, prefix + [target], suffix, [0, n], tap, mark_after=len(prefix) - 1)
        omitted = run_context(model, kda, prefix, suffix, [0, n], tap)
    finally:
        tap.uninstall()
    serial_present = present.states["cut:0"]
    serial_omitted = omitted.states["cut:0"]
    frozen_omitted = omitted.states["cut:0"]
    transported = serial_present - serial_omitted
    for token in range(n):
        kp, vp, gp, bp = (present.inputs[key][:, token].astype(mx.float32) for key in ("k", "v", "g", "beta"))
        ko, vo, go, bo = (omitted.inputs[key][:, token].astype(mx.float32) for key in ("k", "v", "g", "beta"))
        serial_present = _step_transition(kp, gp, bp, serial_present) + _step_write(kp, vp, bp)
        serial_omitted = _step_transition(ko, go, bo, serial_omitted) + _step_write(ko, vo, bo)
        # Reuse serial_present as f_p. f_o starts from the omitted initial
        # state but sees the SAME present inputs/writes as f_p thereafter.
        # In exact arithmetic f_p - f_o equals the propagated receipt.
        frozen_omitted = _step_transition(kp, gp, bp, frozen_omitted) + _step_write(kp, vp, bp)
        transported = _step_transition(kp, gp, bp, transported)
        mx.eval(serial_present, serial_omitted, frozen_omitted, transported)
    native_present, native_omitted = present.states[f"cut:{n}"], omitted.states[f"cut:{n}"]
    native_residual = native_present - transported - native_omitted
    serial_residual = serial_present - transported - serial_omitted
    present_error = serial_present - native_present
    omitted_error = serial_omitted - native_omitted
    vectors = {
        "native_transport_residual": native_residual,
        "serial_transport_residual": serial_residual,
        "present_reconstruction_error": present_error,
        "omitted_reconstruction_error": omitted_error,
        "frozen_input_identity_error": serial_present - frozen_omitted - transported,
        "residual_change_from_reconstruction": native_residual - serial_residual,
        "native_present_minus_omitted": native_present - native_omitted,
        "native_present": native_present,
        "native_omitted": native_omitted,
    }
    norms = {key: layer_norms(value) for key, value in vectors.items()}
    maximum_errors = {key: float(mx.max(mx.abs(value)).item()) for key, value in vectors.items()
                      if "error" in key or "residual" in key}
    rows = []
    for j, layer in enumerate(kda):
        row = {"layer": layer, **{key + "_frobenius": values[j] for key, values in norms.items()}}
        floor = norms["present_reconstruction_error"][j] + norms["omitted_reconstruction_error"][j]
        combined_floor = max(floor, norms["frozen_input_identity_error"][j])
        row.update(reconstruction_error_triangle_sum=floor,
                   residual_to_reconstruction_error_ratio=_ratio(norms["native_transport_residual"][j], floor),
                   max_empirical_floor=combined_floor,
                   residual_to_max_empirical_floor_ratio=_ratio(norms["native_transport_residual"][j], combined_floor),
                   native_residual_relative_to_present_difference=_ratio(norms["native_transport_residual"][j], norms["native_present_minus_omitted"][j]),
                   present_reconstruction_relative_error=_ratio(norms["present_reconstruction_error"][j], norms["native_present"][j]),
                   omitted_reconstruction_relative_error=_ratio(norms["omitted_reconstruction_error"][j], norms["native_omitted"][j]))
        rows.append(row)
    global_norms = {key: _global(values) for key, values in norms.items()}
    floor = global_norms["present_reconstruction_error"] + global_norms["omitted_reconstruction_error"]
    combined_floor = max(floor, global_norms["frozen_input_identity_error"])
    aggregate = {key + "_frobenius": value for key, value in global_norms.items()}
    aggregate.update(reconstruction_error_triangle_sum=floor,
                     residual_to_reconstruction_error_ratio=_ratio(global_norms["native_transport_residual"], floor),
                     max_empirical_floor=combined_floor,
                     residual_to_max_empirical_floor_ratio=_ratio(global_norms["native_transport_residual"], combined_floor),
                     native_residual_relative_to_present_difference=_ratio(global_norms["native_transport_residual"], global_norms["native_present_minus_omitted"]),
                     present_reconstruction_relative_error=_ratio(global_norms["present_reconstruction_error"], global_norms["native_present"]),
                     omitted_reconstruction_relative_error=_ratio(global_norms["omitted_reconstruction_error"], global_norms["native_omitted"]))
    return {"status": "complete", "record_id": record["record_id"], "suffix_tokens": n,
            "suffix_class": record["suffix_class"], "kda_layers": len(kda), "per_layer": rows,
            "global": aggregate, "max_abs_errors": maximum_errors, "seconds": time.perf_counter() - started,
            "recurrent_cache_dtypes": [str(present.cache[i].state[3].dtype) for i in kda]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", choices=("8bit", "bf16"), required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    from .manifest import load_manifest, canonical_bytes
    from .precision import inspect_snapshot, require_precision
    manifest = load_manifest(args.manifest)
    plan = json.loads(args.plan.read_text())
    if plan["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("frozen precision plan and manifest differ")
    by_id = {row["record_id"]: row for row in manifest["records"]}
    rows = [by_id[record_id] for record_id in plan["record_ids"]]
    for row in rows:
        if hashlib.sha256(canonical_bytes(row)).hexdigest() != plan["record_sha256"][row["record_id"]]:
            raise ValueError("frozen precision record changed")
    if args.limit:
        rows = rows[:args.limit]
    provenance = None
    if not args.tiny:
        if not args.provenance:
            raise ValueError("real runs require the precision provenance audit")
        provenance = json.loads(args.provenance.read_text())
        require_precision(provenance, args.precision)
        current = inspect_snapshot(Path(args.model))
        if current["config_sha256"] != provenance["config_sha256"] or current["weight_bytes"] != provenance["weight_bytes"]:
            raise ValueError("current model does not match precision provenance audit")
    config = {"schema": "kda-transport-arithmetic-floor-v1", "model": args.model,
              "precision": args.precision, "tiny_plumbing_only": args.tiny,
              "manifest_sha256": manifest["manifest_sha256"], "plan_sha256": plan["plan_sha256"],
              "record_ids": [row["record_id"] for row in rows],
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "runtime": {package: importlib.metadata.version(package) for package in ("mlx", "mlx-lm", "numpy")},
              "precision_provenance": provenance,
              "definitions": {
                  "reconstruction": "Start from each branch's native post-target/omission state; reconstruct suffix from that branch's actual captured k,v,g,beta in serial fp32 arithmetic.",
                  "transport": "Propagate native initial present-minus-omitted receipt through present-branch captured transitions, in the same serial fp32 arithmetic.",
                  "native_residual": "native_present_final - transported_receipt - native_omitted_final",
                  "floor": "Observed sum of present and omitted reconstruction error norms; numerical diagnostic, not a proved bound for all transport floating-point error.",
                  "frozen_input_identity": "Propagate f_p and f_o from native present/omitted initial states through identical captured present k,v,g,beta, including writes. Compare f_p - f_o with the separately propagated receipt.",
                  "max_empirical_floor": "max(present reconstruction error norm + omitted reconstruction error norm, frozen-input identity error norm); an observed diagnostic, not a rigorous bound.",
                  "ratio_null": "A null ratio indicates a zero observed denominator.",
                  "scope": "Recurrent KDA arrays only; complete hybrid state and readout audits remain in the main evaluation."}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("refusing to resume a different diagnostic configuration")
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    pending = [row for row in rows if not (args.out_dir / f"{row['record_id']}.json").exists()]
    if pending:
        import mlx.core as mx
        from kimi_sv.protocol import load_model
        from kimi_sv.mla_mask import install_mla_mask
        mx.set_cache_limit(2 * 1024**3)
        model, tokenizer, _ = load_model(args.model, tiny=args.tiny, lazy=not args.tiny)
        install_mla_mask(model)
        for row in pending:
            result = diagnostic_record(model, tokenizer, row, tiny=args.tiny)
            result["manifest_sha256"] = manifest["manifest_sha256"]
            result["precision"] = args.precision
            path = args.out_dir / f"{row['record_id']}.json"
            with path.open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.write("\n")
            print(f"{row['record_id']}: residual/floor={result['global']['residual_to_reconstruction_error_ratio']}; {result['seconds']:.1f}s", flush=True)
            mx.clear_cache()
    complete = [json.loads((args.out_dir / f"{row['record_id']}.json").read_text()) for row in rows]
    ratios = [row["global"]["residual_to_reconstruction_error_ratio"] for row in complete]
    combined_ratios = [row["global"]["residual_to_max_empirical_floor_ratio"] for row in complete]
    summary = {"schema": "kda-transport-arithmetic-floor-summary-v1", "status": "complete",
               "precision": args.precision, "tiny_plumbing_only": args.tiny, "n_records": len(complete),
               "record_ids": [r["record_id"] for r in complete],
               "global_residual_to_reconstruction_error_ratios": ratios,
               "global_residual_to_max_empirical_floor_ratios": combined_ratios,
               "minimum_global_ratio": min((v for v in ratios if v is not None), default=None),
               "maximum_global_ratio": max((v for v in ratios if v is not None), default=None),
               "minimum_global_combined_floor_ratio": min((v for v in combined_ratios if v is not None), default=None),
               "maximum_global_combined_floor_ratio": max((v for v in combined_ratios if v is not None), default=None),
               "max_present_reconstruction_relative_error": max(r["global"]["present_reconstruction_relative_error"] for r in complete),
               "max_omitted_reconstruction_relative_error": max(r["global"]["omitted_reconstruction_relative_error"] for r in complete)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
