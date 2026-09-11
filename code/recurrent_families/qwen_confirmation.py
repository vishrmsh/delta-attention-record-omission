"""Preselected Qwen cohort: fresh cached weights, full replay state, receipt transport.

Freeze before evaluating: python -m recurrent_families.qwen_confirmation freeze
Run: python -m recurrent_families.qwen_confirmation run --device mps
No network requests, result replacement, or outcome-dependent record selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any

import numpy as np

from gemma_sv import qwen_replay_audit as audit

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "benchmarks/qwen_confirmation_20260911.json"
OUTDIR = ROOT / "results/qwen_confirmation_20260911"
REVISION = "57370f0ea82c3cca33558a95212e032c344e5fd5"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots" / REVISION
SEED = 2026091105


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return audit._sha256_file(path)


def make_records(tokenizer, n: int = 32) -> list[dict[str, Any]]:
    """Synthetic records generated before model loading; all entries are retained."""
    rng = random.Random(SEED)
    domains = ["observatory", "library", "greenhouse", "railway", "museum", "workshop", "aquarium", "theater"]
    objects = ["lantern", "maple", "quartz", "copper", "falcon", "mosaic", "ribbon", "willow", "pebble", "violet", "badger", "silver"]
    activities = ["inventory", "cleaning", "delivery", "inspection", "training", "maintenance"]
    rows = []
    for i in range(n):
        domain = domains[i % len(domains)]
        code = f"{rng.choice(objects)} {rng.choice(objects)} {rng.randrange(1000,9999)}"
        prefix = f"User: This fictional {domain} ledger contains independent synthetic records. Use only the listed facts.\nAssistant: I will track the supplied ledger.\n"
        for j in range(2 + i % 4):
            prefix += f"User: Retained entry R{i:02d}-{j} has {rng.randrange(3,28)} {rng.choice(objects)} markers.\nAssistant: Retained entry R{i:02d}-{j} is recorded.\n"
        victim = f"User: Temporary entry T{i:02d} has access phrase {code}.\nAssistant: Entry T{i:02d} has access phrase {code}.\n"
        # Independent suffixes share no deleted phrase or entry identifier.
        suffixes = {}
        for label, minimum_tokens in (("short", 64), ("long", 512)):
            suffix = ""
            j = 0
            while len(audit._input_ids(tokenizer, suffix)) < minimum_tokens:
                suffix += f"User: The {rng.choice(domains)} {rng.choice(activities)} log lists {rng.randrange(10,90)} {rng.choice(objects)} labels for batch {j}.\nAssistant: The unrelated batch {j} observation is recorded.\n"
                j += 1
            suffixes[label] = suffix
        texts = {"prefix": prefix, "victim": victim, **suffixes}
        tokens = {key: audit._input_ids(tokenizer, value) for key, value in texts.items()}
        boundaries = {label: audit._input_ids(tokenizer, prefix + victim + suffixes[label]) == tokens["prefix"] + tokens["victim"] + tokens[label] and audit._input_ids(tokenizer, prefix + suffixes[label]) == tokens["prefix"] + tokens[label] for label in suffixes}
        if not all(boundaries.values()):
            raise ValueError(f"raw/segment tokenization mismatch for record {i}")
        rows.append({"record_id": f"qwen-independent-{i:02d}", "domain": domain, "target_phrase": code, "text": texts, "token_ids": tokens, "token_counts": {key: len(value) for key, value in tokens.items()}, "raw_segment_boundaries_match": boundaries})
    return rows


def freeze(path: Path) -> None:
    from transformers import AutoTokenizer
    if path.exists():
        raise FileExistsError(f"Frozen manifest already exists: {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    prior = json.loads((ROOT.parent / "gemma_sv/benchmarks/qwen35_replay_v3.json").read_text())
    manifest = {
        "schema": "qwen-independent-cohort-v1", "status": "frozen-before-new-cohort-evaluation", "created_utc": utc_now(),
        "seed": SEED, "n_records": 32, "model": prior["model"], "architecture_contract": prior["architecture_contract"],
        "local_model_directory": str(MODEL_DIR), "model_config_sha256": sha(MODEL_DIR / "config.json"), "weight_index_sha256": sha(MODEL_DIR / "model.safetensors.index.json"),
        "design": {"source": "new synthetic cohort generated without evaluating model outputs", "independent_of_prior_three_scenarios": True, "selection": "all 32 fixed records; no qualification filtering", "suffix_minimum_tokens": {"short": 64, "long": 512}, "matched_trajectories": ["record present", "checkpoint restore and suffix replay", "fresh never stored", "fresh never stored repeat"], "transport_probe_layers": [0,16,30], "complete_state_probe": "All 24 conv arrays, 24 recurrent arrays, 8 key arrays, 8 value arrays, final logits, cache length, initialization flags and rope bookkeeping", "chunk_schedule": "each prefix, victim and suffix is a complete segment; replay and omission schedules match", "claimed_scope": "schedule-matched complete active-state replay and selected-layer receipt transport; not a behavioral deletion or population generalization study"},
        "acceptance": {"replay_and_repeat_max_abs": 0.0, "complete_array_count": 65, "complete_flag_count": 82, "frozen_transport_state_scaled_error_max": 0.005, "forcing_decomposition_relative_error_max": 0.005, "live_recurrence_state_scaled_error_max": 0.005, "native_mismatch_to_numerical_floor_min": 5.0, "nonzero_native_effect_min": 1e-6},
        "analysis": {"unit": "whole record; short and long suffixes and layers remain paired", "bootstrap_replicates": 10000, "bootstrap_seed": SEED + 1, "confidence_level": 0.95, "binomial_intervals": "Wilson 95% over whole-record success (both suffixes); bootstrap also provided", "transport_relative_denominators": "Report both effect-relative and omitted-state-relative errors; long-suffix vanishing frozen effects can destabilize effect-relative ratios", "numerical_floor": "maximum frozen identity error and recurrence/live mismatch in absolute Frobenius norm; lower bounded at 1e-12"},
        "records": make_records(tokenizer),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"frozen_manifest": str(path), "sha256": sha(path), "n_records":len(manifest["records"]), "token_ranges":{key: [min(r["token_counts"][key] for r in manifest["records"]), max(r["token_counts"][key] for r in manifest["records"])] for key in ("prefix", "victim", "short", "long")}}), flush=True)


def receipt_analysis(prefix_state, boundary_state, p, o):
    """Carry native forcing and frozen controls online, comparing to live kernel states."""
    import torch
    pin, oin = p["initial"].float(), o["initial"].float()
    fp, fo = pin.clone(), oin.clone()
    native_o = oin.clone()
    transported = pin - oin
    ledger = transported.clone()
    static = transported.clone()
    tf, wf = torch.zeros_like(transported), torch.zeros_like(transported)
    forcing_abs = 0.0
    forcing_scale = 0.0
    for i in range(p["k"].shape[1]):
        pk,pv,pg,pb = (p[key][:,i] for key in ("k","v","g","beta"))
        ok,ov,og,ob = (o[key][:,i] for key in ("k","v","g","beta"))
        tpo, too = audit._transition(native_o,pk,pg,pb), audit._transition(native_o,ok,og,ob)
        wp, wo = audit._write(pk,pv,pb), audit._write(ok,ov,ob)
        tf = audit._transition(tf,pk,pg,pb) + tpo - too
        wf = audit._transition(wf,pk,pg,pb) + wp - wo
        transported = audit._transition(transported,pk,pg,pb)
        ledger = ledger * pg.exp().unsqueeze(-1).unsqueeze(-1)
        fp = audit._transition(fp,pk,pg,pb) + wp
        fo = audit._transition(fo,pk,pg,pb) + wp
        native_o = too + wo
        direct = fp - native_o
        forcing_abs = max(forcing_abs, float((transported + tf + wf - direct).norm()))
        forcing_scale = max(forcing_scale, float(direct.norm()))
    pfinal, ofinal = p["final"].float(), o["final"].float()
    frozen = fp - fo
    native = pfinal - ofinal
    scale = max(float(ofinal.norm()), 1e-12)
    live_scale = max(scale, float(pfinal.norm()))
    frozen_abs = float((transported - frozen).norm())
    live_abs = max(float((fp-pfinal).norm()), float((native_o-ofinal).norm()))
    native_abs = float((transported-native).norm())
    floor = max(frozen_abs, live_abs, 1e-12)
    return {
        "boundary_effect_norm":float(static.norm()), "native_effect_norm":float(native.norm()), "frozen_effect_norm":float(frozen.norm()), "omitted_state_norm":scale,
        "frozen_transport_relative_residual":frozen_abs / max(float(frozen.norm()),1e-12),
        "frozen_transport_state_scaled_error":frozen_abs / scale,
        "frozen_static_state_scaled_error":float((static-frozen).norm())/scale,
        "frozen_ledger_state_scaled_error":float((ledger-frozen).norm())/scale,
        "native_transport_relative_residual":native_abs/max(float(native.norm()),1e-12),
        "native_transport_state_scaled_error":native_abs/scale,
        "native_mismatch_to_numerical_floor":native_abs/floor,
        "numerical_floor_absolute":floor,
        "live_recurrence_state_scaled_error":live_abs/live_scale,
        "present_initial_state_relative_residual":audit._relative_residual(pin,boundary_state),
        "omitted_initial_state_relative_residual":audit._relative_residual(oin,prefix_state),
        "forcing_decomposition_relative_error":forcing_abs/max(forcing_scale,1e-12),
        "transition_forcing_norm":float(tf.norm()), "write_forcing_norm":float(wf.norm()),
        "finite":all(bool(torch.isfinite(x).all()) for x in (transported, frozen, native, pfinal, ofinal, tf, wf)),
    }


def suffix_case(model, record, label, layers, device):
    tokens = record["token_ids"]
    cache = audit._new_cache(model)
    audit.prefill(model,cache,tokens["prefix"],device)
    mark = audit.snapshot_cache(model,cache)
    prefix = audit._recurrent_by_layer(cache,layers)
    audit.prefill(model,cache,tokens["victim"],device)
    boundary = audit._recurrent_by_layer(cache,layers)
    with audit.capture_delta_inputs(model,layers) as pc:
        pc.enabled = True
        out = audit.prefill(model,cache,tokens[label],device)
    present = audit._state_snapshot(model,cache,out.logits[:,-1])
    audit.restore_cache(model,cache,mark)
    out = audit.prefill(model,cache,tokens[label],device)
    replay = audit._state_snapshot(model,cache,out.logits[:,-1])
    omitted_cache = audit._new_cache(model)
    audit.prefill(model,omitted_cache,tokens["prefix"],device)
    with audit.capture_delta_inputs(model,layers) as oc:
        oc.enabled = True
        out = audit.prefill(model,omitted_cache,tokens[label],device)
    omitted = audit._state_snapshot(model,omitted_cache,out.logits[:,-1])
    repeat_cache = audit._new_cache(model)
    audit.prefill(model,repeat_cache,tokens["prefix"],device)
    out = audit.prefill(model,repeat_cache,tokens[label],device)
    repeat = audit._state_snapshot(model,repeat_cache,out.logits[:,-1])
    receipt = {}
    for layer in layers:
        if len(pc.rows[layer]) != 1 or len(oc.rows[layer]) != 1:
            raise ValueError("Expected exactly one live kernel capture per suffix/layer")
        receipt[str(layer)] = receipt_analysis(prefix[layer],boundary[layer],pc.rows[layer][0],oc.rows[layer][0])
    return {"suffix":label, "suffix_tokens":len(tokens[label]), "present_vs_omitted":audit.compare_snapshots(omitted,present,strict_shapes=False), "replay_vs_fresh_omission":audit.compare_snapshots(omitted,replay), "repeat_vs_fresh_omission":audit.compare_snapshots(omitted,repeat), "receipt_analysis":receipt}


def evaluate_checks(case, acceptance):
    receipts = list(case["receipt_analysis"].values())
    checks = {}
    for label in ("replay", "repeat"):
        row = case[f"{label}_vs_fresh_omission"]
        checks[f"{label}_complete_state_exact"] = all((row["all_arrays_exact"],row["all_flags_equal"],row["all_values_finite"],row["max_abs"] == acceptance["replay_and_repeat_max_abs"],row["array_count"] == acceptance["complete_array_count"],row["flag_count"] == acceptance["complete_flag_count"],row["shape_mismatch_count"] == 0))
    checks["frozen_transport_closes"] = all(r["frozen_transport_state_scaled_error"] <= acceptance["frozen_transport_state_scaled_error_max"] for r in receipts)
    checks["forcing_decomposition_closes"] = all(r["forcing_decomposition_relative_error"] <= acceptance["forcing_decomposition_relative_error_max"] for r in receipts)
    checks["live_recurrence_conforms"] = all(max(r["live_recurrence_state_scaled_error"],r["present_initial_state_relative_residual"],r["omitted_initial_state_relative_residual"]) <= acceptance["live_recurrence_state_scaled_error_max"] for r in receipts)
    checks["native_transport_mismatch_above_floor"] = any(r["native_effect_norm"] >= acceptance["nonzero_native_effect_min"] and r["native_mismatch_to_numerical_floor"] >= acceptance["native_mismatch_to_numerical_floor_min"] for r in receipts)
    checks["finite"] = all(r["finite"] for r in receipts) and case["present_vs_omitted"]["all_values_finite"]
    return checks


def wilson(successes, n):
    if not n:
        return [None,None]
    z = 1.959963984540054
    p = successes/n
    middle = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,middle-half), min(1.,middle+half)]


def bootstrap_mean(values, index):
    vals = np.asarray(values,dtype=float)
    draws = vals[index].mean(axis=1)
    return {"mean":float(vals.mean()),"ci95":np.quantile(draws,[.025,.975]).tolist(),"n_records":len(vals)}


def summarize(rows, manifest):
    n = len(manifest["records"])
    by_id = {row["record_id"]:row for row in rows}
    ordered = [by_id.get(r["record_id"],{}) for r in manifest["records"]]
    rng = np.random.default_rng(manifest["analysis"]["bootstrap_seed"])
    index = rng.integers(0,n,size=(manifest["analysis"]["bootstrap_replicates"],n))
    result = {"preselected_records":n,"attempted_records":len(rows),"completed_records":sum(r.get("status")=="completed" for r in ordered),"failed_records":[r.get("record_id") for r in ordered if r.get("status") not in (None,"completed")],"whole_record_checks":{},"metrics_by_suffix":{},"paired_long_minus_short":{}}
    names = set().union(*(set(c.get("checks",{})) for r in ordered for c in r.get("cases",[])))
    for name in sorted(names):
        successes = [int(r.get("status")=="completed" and len(r.get("cases",[]))==2 and all(c["checks"].get(name,False) for c in r["cases"])) for r in ordered]
        result["whole_record_checks"][name] = {**bootstrap_mean(successes,index),"successes":sum(successes),"denominator_including_failures":n,"wilson_ci95":wilson(sum(successes),n)}
    completed = [r for r in ordered if r.get("status")=="completed" and len(r.get("cases",[]))==2]
    if completed:
        m = len(completed)
        mi = rng.integers(0,m,size=(manifest["analysis"]["bootstrap_replicates"],m))
        metric_names = ["native_transport_relative_residual","native_transport_state_scaled_error","frozen_transport_state_scaled_error","frozen_transport_relative_residual","frozen_static_state_scaled_error","frozen_ledger_state_scaled_error","live_recurrence_state_scaled_error","forcing_decomposition_relative_error","native_mismatch_to_numerical_floor"]
        for metric in metric_names:
            paired = {}
            for label in ("short","long"):
                vals = [float(np.mean([v[metric] for c in r["cases"] if c["suffix"]==label for v in c["receipt_analysis"].values()])) for r in completed]
                paired[label] = vals
                result["metrics_by_suffix"].setdefault(label,{})[metric] = bootstrap_mean(vals,mi)
            result["paired_long_minus_short"][metric] = bootstrap_mean(np.asarray(paired["long"])-np.asarray(paired["short"]),mi)
        result["continuous_metrics_note"] = f"Means average selected layers within each record, then bootstrap whole records. Continuous summaries have {m}/{n} complete records; operational failures are retained in all-record binary denominators."
    return result


def run(manifest_path, outdir, device):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    outdir.mkdir(parents=True,exist_ok=True)
    log_path = outdir / "records.jsonl"
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior evaluation: {log_path}")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable: run with approved unsandboxed GPU access")
    audit._seed_everything(manifest["seed"])
    torch.set_num_threads(4)
    metadata = {"schema":"qwen-independent-cohort-results-v1","started_utc":utc_now(),"status":"loading_model","manifest_path":str(manifest_path),"manifest_sha256":manifest_hash,"implementation_sha256":sha(Path(__file__)),"base_audit_implementation_sha256":sha(Path(audit.__file__)),"environment":audit._environment(),"device":device,"model":manifest["model"],"local_files_only":True,"tiny_random_model":False,"records":[]}
    audit._atomic_write(outdir / "results.json",metadata)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(manifest["local_model_directory"],dtype=torch.bfloat16,attn_implementation="sdpa",low_cpu_mem_usage=True,use_safetensors=True,local_files_only=True).eval().to(device)
    model.requires_grad_(False)
    layers = audit._validate_architecture(model,manifest,tiny=False)
    metadata["kernel_implementation"] = audit._kernel_implementation()
    metadata["status"] = "running"
    started = time.perf_counter()
    with log_path.open("x") as log:
        for record in manifest["records"]:
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
                raise RuntimeError("Frozen manifest changed during evaluation")
            row = {"record_id":record["record_id"],"token_counts":record["token_counts"],"started_utc":utc_now(),"status":"running","cases":[]}
            tick = time.perf_counter()
            try:
                for label in ("short","long"):
                    with torch.inference_mode():
                        case = suffix_case(model,record,label,layers,device)
                    case["checks"] = evaluate_checks(case,manifest["acceptance"])
                    row["cases"].append(case)
                    if device == "mps":
                        torch.mps.empty_cache()
                row["status"] = "completed"
                row["all_checks_pass"] = all(all(c["checks"].values()) for c in row["cases"])
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
                row["all_checks_pass"] = False
                if device == "mps":
                    torch.mps.empty_cache()
            row["elapsed_seconds"] = time.perf_counter()-tick
            log.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
            log.flush()
            os.fsync(log.fileno())
            metadata["records"].append(row)
            metadata["elapsed_seconds"] = time.perf_counter()-started
            audit._atomic_write(outdir / "results.json",metadata)
            print(json.dumps({"record_id":row["record_id"],"status":row["status"],"all_checks_pass":row["all_checks_pass"],"elapsed_seconds":row["elapsed_seconds"],"completed":len(metadata["records"]),"total":len(manifest["records"])}),flush=True)
    metadata["status"] = "completed" if all(r["all_checks_pass"] for r in metadata["records"]) else "completed_with_failed_checks"
    metadata["ended_utc"] = utc_now()
    metadata["summary"] = summarize(metadata["records"],manifest)
    audit._atomic_write(outdir / "results.json",metadata)
    audit._atomic_write(outdir / "summary.json",metadata["summary"])
    print(json.dumps({"status":metadata["status"],"outdir":str(outdir),"elapsed_seconds":metadata["elapsed_seconds"]}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["freeze","run","summarize"])
    parser.add_argument("--manifest",type=Path,default=MANIFEST)
    parser.add_argument("--outdir",type=Path,default=OUTDIR)
    parser.add_argument("--device",default="mps")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.manifest)
    elif args.action == "run":
        run(args.manifest,args.outdir,args.device)
    else:
        manifest = json.loads(args.manifest.read_text())
        rows = [json.loads(line) for line in (args.outdir/"records.jsonl").read_text().splitlines()]
        audit._atomic_write(args.outdir/"summary.json",summarize(rows,manifest))


if __name__ == "__main__":
    main()
