"""Independent, resumable Kimi cohort evaluation for the SaTML follow-up.

Every condition is evaluated on every frozen record, regardless of admission.
The original sweep's native mask and transition algebra are reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from pathlib import Path

import mlx.core as mx

from kimi_sv.eval_forcing_sweep import (
    AllLayerTap, _step_transition, layer_norms, run_context, with_recurrent,
)
from kimi_sv.eval_masked_sampling import NullTap
from kimi_sv.mla_mask import install_mla_mask, set_mla_drop_positions
from kimi_sv.probes import teacher_forced
from kimi_sv.protocol import encode, load_model, stop_ids
from kimi_sv.records import RecordMemory
from kimi_sv.state import classify_layers, restore, snapshot
from .manifest import load_manifest


def compare_cache(model, left, right):
    """Compare all active KDA/conv and logical MLA arrays, shapes and offsets."""
    kda, mla = classify_layers(model)
    rows = []
    for i in kda:
        ls, rs = left[i].state, right[i].state
        if len(ls) != len(rs):
            raise ValueError("KDA state arity mismatch")
        for j, (a, b) in enumerate(zip(ls, rs)):
            rows.append(_array_diff(f"layer:{i}/array:{j}", a, b))
    offsets = []
    for i in mla:
        a, b = left[i], right[i]
        na, nb = int(a.offset), int(b.offset)
        offsets.append({"layer": i, "candidate": na, "reference": nb, "equal": na == nb})
        for key in ("keys", "values"):
            aa, bb = getattr(a, key), getattr(b, key)
            rows.append(_array_diff(f"layer:{i}/{key}",
                                    None if aa is None else aa[..., :na, :],
                                    None if bb is None else bb[..., :nb, :]))
    finite_diffs = [r["max_abs_diff"] for r in rows if r["max_abs_diff"] is not None]
    return {"arrays": rows, "offsets": offsets, "n_arrays": len(rows),
            "all_equal": all(r["equal"] for r in rows + offsets),
            "max_abs_diff": max(finite_diffs, default=0.0)}


def _array_diff(name, a, b):
    shapes_match = (a is None and b is None) or (
        a is not None and b is not None and a.shape == b.shape)
    diff = None
    if shapes_match:
        diff = 0.0 if a is None or a.size == 0 else float(
            mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
    return {"name": name, "shape_candidate": None if a is None else list(a.shape),
            "shape_reference": None if b is None else list(b.shape),
            "max_abs_diff": diff, "equal": shapes_match and diff == 0.0}


def corrections(present, omitted, suffix_tokens):
    """Transport the receipt through captured present transitions, in fp32.

    Measure residuals against the actual omitted state. Do not reconstruct that
    reference with different arithmetic or substitute forcing norms for errors.
    """
    last = f"cut:{suffix_tokens}"
    receipt = present.states["cut:0"] - omitted.states["cut:0"]
    transported = receipt
    ledger = mx.ones_like(present.inputs["g"][:, 0].astype(mx.float32))
    start = time.perf_counter()
    for t in range(suffix_tokens):
        k = present.inputs["k"][:, t].astype(mx.float32)
        g = present.inputs["g"][:, t].astype(mx.float32)
        beta = present.inputs["beta"][:, t].astype(mx.float32)
        transported = _step_transition(k, g, beta, transported)
        ledger = ledger * g
        mx.eval(transported, ledger)
    sp, so = present.states[last], omitted.states[last]
    states = {"static": sp - receipt, "ledger": sp - receipt * ledger[..., None, :],
              "transport": sp - transported}
    direct_norms, ref_norms = layer_norms(sp - so), layer_norms(so)
    summaries = {}
    for name, state in {"present": sp, **states}.items():
        norms = layer_norms(state - so)
        summaries[name] = {
            "per_layer_residual_norm": norms,
            "per_layer_relative_to_present_difference": [n / max(d, 1e-12) for n, d in zip(norms, direct_norms)],
            "per_layer_relative_to_reference": [n / max(d, 1e-12) for n, d in zip(norms, ref_norms)],
            "global_relative_to_present_difference": (sum(n*n for n in norms) / max(sum(n*n for n in direct_norms), 1e-24)) ** .5,
        }
    return states, summaries, time.perf_counter() - start


def token_plan(tokenizer, record, tiny=False):
    prefix = [encode(tokenizer, record["preamble"], special=True)]
    prefix += [encode(tokenizer, r["text"]) for r in record["prefix_records"]]
    prefix += [encode(tokenizer, r["text"]) for r in record["retained_records"]]
    target = encode(tokenizer, record["target"]["text"])
    suffix_text = "".join(r["text"] for r in record["suffix_records"])
    n = min(record["suffix_tokens"], 16) if tiny else record["suffix_tokens"]
    suffix = encode(tokenizer, suffix_text)
    if suffix.shape[1] < n:
        suffix = mx.concatenate([suffix] * (n // suffix.shape[1] + 1), axis=1)
    suffix = suffix[:, :n]
    forget = encode(tokenizer, "\n" + record["forget_instruction"] + "\n")
    neutral = encode(tokenizer, "\n" + record["neutral_instruction"] + "\n")
    # Fixed neutral prose repeated and truncated to exactly the forgetting
    # instruction's token budget; record the actual ids for reproducibility.
    neutral = mx.concatenate([neutral] * (forget.shape[1] // neutral.shape[1] + 1), axis=1)[:, :forget.shape[1]]
    return prefix, target, suffix, forget, neutral


def score_condition(model, tokenizer, cache, boundary, record, drop=None,
                    instruction=None, max_new_tokens=24):
    restore(model, cache, boundary)
    set_mla_drop_positions(model, drop)
    started = time.perf_counter()
    try:
        if instruction is not None:
            mx.eval(model(instruction, cache=cache))
        memory = RecordMemory(model, cache=cache)
        result = {"extra_instruction_tokens": 0 if instruction is None else int(instruction.shape[1])}
        for group in ("target_probes", "retained_probes", "continuations"):
            result[group] = []
            for j, probe in enumerate(record[group]):
                prompt = encode(tokenizer, "\n" + probe["prompt"])
                target = encode(tokenizer, probe["target"])
                row = {"probe_id": probe["probe_id"], **teacher_forced(memory, prompt, target).as_dict()}
                if (group == "target_probes" and j == 0) or group == "retained_probes":
                    toks = memory.generate(prompt, max_tokens=max_new_tokens, stop_ids=stop_ids(tokenizer))
                    answer = tokenizer.decode(toks).strip()
                    row.update(greedy_text=answer, greedy_exact=probe["target"].strip().casefold() in answer.casefold(),
                               generated_tokens=len(toks))
                result[group].append(row)
        result["seconds"] = time.perf_counter() - started
        return result
    finally:
        set_mla_drop_positions(model, None)
        restore(model, cache, boundary)


def evaluate_record(model, tokenizer, record, *, tiny=False, max_new_tokens=24):
    kda, mla = classify_layers(model)
    prefix, target, suffix, forget, neutral = token_plan(tokenizer, record, tiny)
    n = int(suffix.shape[1]); cuts = [0, n]
    tap = AllLayerTap(len(kda))
    tap.install()
    try:
        present = run_context(model, kda, prefix + [target], suffix, cuts, tap, mark_after=len(prefix)-1)
        # No need to capture omitted transitions: the direct actual state is
        # the reference and only present transitions define receipt transport.
        omitted = run_context(model, kda, prefix, suffix, cuts, NullTap())
    finally:
        tap.uninstall()
    corrected, state_summary, transport_seconds = corrections(present, omitted, n)
    present.inputs = {}; omitted.inputs = {}
    bp, bo = present.boundaries[f"cut:{n}"], omitted.boundaries[f"cut:{n}"]
    start = present.boundaries["prefix"].token_offset
    positions = list(range(start, start + int(target.shape[1])))
    conditions = {}
    def score(name, cache, boundary, drop=None, instruction=None):
        conditions[name] = score_condition(model, tokenizer, cache, boundary, record,
                                          drop, instruction, max_new_tokens)
    score("never_stored", omitted.cache, bo)
    for name, drop in (("present", None), ("masked", positions)):
        score(name, present.cache, bp, drop)
    for name, state in corrected.items():
        boundary = with_recurrent(bp, kda, state)
        score(name, present.cache, boundary)
        if name == "transport":
            score("mask_transport", present.cache, boundary, positions)
    score("prompt_forget", present.cache, bp, instruction=forget)
    score("prompt_neutral", present.cache, bp, instruction=neutral)
    repeat = run_context(model, kda, prefix, suffix, cuts, NullTap())
    repeat_state = compare_cache(model, repeat.cache, omitted.cache)
    score("never_stored_repeat", repeat.cache, repeat.boundaries[f"cut:{n}"])
    # Run replay last: restoring the prefix overwrites the former target rows.
    restore(model, present.cache, present.boundaries["prefix"])
    replay_start = time.perf_counter()
    mx.eval(model(suffix, cache=present.cache))
    replay_seconds = time.perf_counter() - replay_start
    replay_boundary = snapshot(model, present.cache)
    replay_state = compare_cache(model, present.cache, omitted.cache)
    audit_prompt = encode(tokenizer, "\n" + record["target_probes"][0]["prompt"])
    ref_logits = RecordMemory(model, cache=omitted.cache).probe(audit_prompt)
    for trace, state in ((present, replay_state), (repeat, repeat_state)):
        lp = RecordMemory(model, cache=trace.cache).probe(audit_prompt)
        state["logits"] = _array_diff("audit_next_token_logits", lp, ref_logits)
        state["all_equal"] = state["all_equal"] and state["logits"]["equal"]
    score("replay", present.cache, replay_boundary)
    p = conditions["present"]["target_probes"][0]
    o = conditions["never_stored"]["target_probes"][0]
    lift = p["mean_log_probability"] - o["mean_log_probability"]
    return {
        "status": "complete", "conditions": conditions,
        "qualification": {"qualified": lift >= .05 and p["first_token_rank"] <= 10,
                          "lift_nats": lift, "first_token_rank": p["first_token_rank"]},
        "state_checks": {"replay_vs_never": replay_state, "repeat_never_vs_never": repeat_state,
                         "corrections": state_summary},
        "tokens": {"prefix": start, "target": int(target.shape[1]), "suffix": n,
                   "forget_instruction": int(forget.shape[1]), "neutral_instruction": int(neutral.shape[1]),
                   "forget_ids": forget[0].tolist(), "neutral_ids": neutral[0].tolist()},
        "timing": {"transport_seconds": transport_seconds, "replay_seconds": replay_seconds,
                   "present_suffix_seconds": present.suffix_seconds, "never_suffix_seconds": omitted.suffix_seconds},
        "n_kda_layers": len(kda), "n_mla_layers": len(mla),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--record-ids", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=24)
    args = p.parse_args(argv)
    manifest = load_manifest(args.manifest)
    digest = manifest["manifest_sha256"]
    if not args.tiny and args.max_new_tokens != manifest["protocol"]["decoding"]["max_new_tokens"]:
        raise ValueError("Real runs must use the frozen decoding token budget")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    config = {"schema_version": 1, "manifest_sha256": digest, "model_id": args.model,
              "tiny": args.tiny, "max_new_tokens": args.max_new_tokens,
              "runtime": {k: importlib.metadata.version(k) for k in ("mlx", "mlx-lm", "numpy")},
              "platform": platform.platform(), "device": mx.device_info(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_path = out / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Refusing resume with changed configuration or runner source")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    rows = manifest["records"]
    if args.record_ids:
        selected = set(args.record_ids.split(",")); rows = [r for r in rows if r["record_id"] in selected]
        if {r["record_id"] for r in rows} != selected:
            raise ValueError("Unknown record ids")
    if args.limit: rows = rows[:args.limit]
    pending = [r for r in rows if not (out / (r["record_id"] + ".json")).exists()]
    if not pending:
        print("No pending records", flush=True); return 0
    mx.set_cache_limit(2 * 1024**3)
    start = time.perf_counter()
    model, tokenizer, label = load_model(args.model, tiny=args.tiny, lazy=True)
    install_mla_mask(model)
    print(f"Loaded {label}; {len(pending)} pending records", flush=True)
    errors = 0
    for idx, record in enumerate(pending):
        t0 = time.perf_counter()
        base = {"schema_version": 1, "manifest_sha256": digest, "record_id": record["record_id"],
                "model_id": args.model, "tiny": args.tiny, "split": record["split"]}
        try:
            result = evaluate_record(model, tokenizer, record, tiny=args.tiny,
                                     max_new_tokens=args.max_new_tokens)
        except Exception:
            errors += 1
            result = {"status": "error", "traceback": traceback.format_exc()}
            set_mla_drop_positions(model, None)
        result = {**base, **result, "seconds": time.perf_counter() - t0}
        dest = out / (record["record_id"] + ".json")
        with dest.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
        print(f"{idx+1}/{len(pending)} {record['record_id']} {result['status']} "
              f"{result['seconds']:.1f}s qualified={result.get('qualification',{}).get('qualified')} "
              f"replay_equal={result.get('state_checks',{}).get('replay_vs_never',{}).get('all_equal')}", flush=True)
        mx.clear_cache()
    print(f"Finished in {time.perf_counter()-start:.1f}s; errors={errors}", flush=True)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
