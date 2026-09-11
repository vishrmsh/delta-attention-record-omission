"""Preselected amendment and sequential-maintenance cohorts for Kimi Linear.

Twelve independent synthetic amendment targets are fixed before model loading.
Six disjoint target pairs also undergo both deletion orders, with a new record
admitted between deletions. Every transition is compared with a fresh reference
using its current surviving record sequence. Tiny mode checks plumbing only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Any

SEED = 202609117
PREAMBLE = "Internal case file. Each entry is one patient record. Answer questions using only these entries.\n\n"


def make_manifest(n: int = 12, seed: int = SEED) -> dict[str, Any]:
    if not 1 <= n <= 12:
        raise ValueError("the preselected cohort supports 1 through 12 targets")
    rng = random.Random(seed)
    used: set[str] = set()

    def code() -> str:
        while True:
            value = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5))
            value += f"-{rng.randrange(100, 1000)}"
            if value not in used:
                used.add(value)
                return value

    def record(index: int, label: str) -> dict[str, str]:
        ward = code()
        pid = f"{810000 + index}"
        return {"key": f"followup-{label}-{pid}",
                "text": f"Record {pid}. Patient: Person {pid}. Assigned ward code {ward}. Admission purpose: routine observation.\n\n",
                "question": f"Question: What is the ward code for patient {pid}?\nAnswer:",
                "secret": " " + ward}

    scenarios = []
    # Generate the full cohort before slicing: any smaller run is a fixed prefix.
    for i in range(12):
        old = record(10 * i, "target")
        new = {**old, "secret": " " + code()}
        new["text"] = old["text"].replace(old["secret"].strip(), new["secret"].strip())
        retained = [record(10 * i + j + 1, "retained") for j in range(3)]
        admission = record(10 * i + 8, "admitted")
        scenarios.append({"id": f"amend-{i:02d}", "old": old, "new": new,
                          "retained": retained, "admission": admission,
                          "target_position": (0, 1, 3)[i % 3],
                          "suffix_tokens": (0, 128, 512)[(i // 3) % 3]})
    payload = {"schema": "kda-maintenance-manifest-v1", "seed": seed,
               "source": "invented records generated without model feedback",
               "scenarios": scenarios[:n]}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def compare_cache(model: Any, left: Any, right: Any) -> dict[str, Any]:
    """Compare all recurrent/conv arrays and logical active MLA keys/values."""
    import mlx.core as mx
    from kimi_sv.state import classify_layers

    def arrays(a: Any, b: Any) -> dict[str, Any]:
        if a is None or b is None:
            return {"equal": a is None and b is None, "left_none": a is None,
                    "right_none": b is None, "max_abs_diff": 0.0 if a is b else None}
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"equal": False, "left_shape": list(a.shape), "right_shape": list(b.shape),
                    "left_dtype": str(a.dtype), "right_dtype": str(b.dtype), "max_abs_diff": None}
        delta = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) if a.size else 0.0
        return {"equal": bool(mx.array_equal(a, b).item()), "shape": list(a.shape),
                "dtype": str(a.dtype), "max_abs_diff": delta}

    a, b = getattr(left, "cache", left), getattr(right, "cache", right)
    kda, mla = classify_layers(model)
    states = {}
    for i in kda:
        if len(a[i].state) != len(b[i].state):
            raise ValueError("recurrent cache state arity differs")
        states[str(i)] = [arrays(x, y) for x, y in zip(a[i].state, b[i].state)]
    attention = {}
    for i in mla:
        ao, bo = int(a[i].offset), int(b[i].offset)
        attention[str(i)] = {"left_offset": ao, "right_offset": bo, "offset_equal": ao == bo,
                             "keys": arrays(a[i].keys[..., :ao, :], b[i].keys[..., :bo, :]),
                             "values": arrays(a[i].values[..., :ao, :], b[i].values[..., :bo, :])}
    checks = [x for layer in states.values() for x in layer]
    checks += [layer[key] for layer in attention.values() for key in ("keys", "values")]
    return {"all_equal": all(c["equal"] for c in checks) and all(c["offset_equal"] for c in attention.values()),
            "max_abs_diff": max((c["max_abs_diff"] or 0.0 for c in checks), default=0.0),
            "kda_and_conv": states, "logical_mla": attention}


def _tokens(tokenizer: Any, scenario: dict[str, Any], corrected: bool = False) -> list[Any]:
    from kimi_sv.protocol import case_records, filler_record, preamble_record
    records = case_records(tokenizer, scenario["retained"])
    target = case_records(tokenizer, [scenario["new" if corrected else "old"]])[0]
    records.insert(scenario["target_position"], target)
    records.insert(0, preamble_record(tokenizer, PREAMBLE))
    if scenario["suffix_tokens"]:
        records.append(filler_record(tokenizer, scenario["suffix_tokens"]))
    return records


def _score(memory: Any, tokenizer: Any, case: dict[str, str], greedy_tokens: int) -> dict[str, Any]:
    from kimi_sv.protocol import answer_text, probe_secret
    result = asdict(probe_secret(memory, tokenizer, case))
    result["target"] = case["secret"]
    if greedy_tokens:
        result["greedy_answer"] = answer_text(memory, tokenizer, case["question"], max_tokens=greedy_tokens)
        result["greedy_contains_target"] = case["secret"].strip().casefold() in result["greedy_answer"].casefold()
    return result


def _score_amendment(memory: Any, tokenizer: Any, scenario: dict[str, Any], greedy_tokens: int) -> dict[str, Any]:
    # One generation answers both old/new scoring targets because the question
    # is identical. Retained targets are explicitly unrelated to the correction.
    old = _score(memory, tokenizer, scenario["old"], greedy_tokens)
    new = _score(memory, tokenizer, scenario["new"], 0)
    if greedy_tokens:
        new["greedy_answer"] = old["greedy_answer"]
        new["greedy_contains_target"] = scenario["new"]["secret"].strip().casefold() in old["greedy_answer"].casefold()
    retained = {c["key"]: _score(memory, tokenizer, c, greedy_tokens) for c in scenario["retained"]}
    return {"old": old, "new": new, "retained": retained}


def _audit(model: Any, left: Any, right: Any, tokenizer: Any, question: str) -> dict[str, Any]:
    from kimi_sv.protocol import encode
    from kimi_sv.records import max_abs_diff
    cache = compare_cache(model, left, right)
    logits = max_abs_diff(left.probe(encode(tokenizer, "\n" + question)), right.probe(encode(tokenizer, "\n" + question)))
    return {"complete_cache": cache, "probe_logit_max_abs_diff": logits,
            "all_equal": cache["all_equal"] and logits == 0.0}


def run_amendment(model: Any, tokenizer: Any, scenario: dict[str, Any], greedy_tokens: int = 16) -> dict[str, Any]:
    from kimi_sv.protocol import build_memory, case_records
    started = time.perf_counter()
    heard = build_memory(model, _tokens(tokenizer, scenario))
    corrected_records = _tokens(tokenizer, scenario, corrected=True)
    reference = build_memory(model, corrected_records)
    repeated = build_memory(model, corrected_records)
    floor = _audit(model, repeated, reference, tokenizer, scenario["old"]["question"])
    del repeated
    never = build_memory(model, [r for r in corrected_records if r.key != scenario["old"]["key"]])
    never_old = _score(never, tokenizer, scenario["old"], 0)
    del never
    before = _score_amendment(heard, tokenizer, scenario, greedy_tokens)
    correct = _score_amendment(reference, tokenizer, scenario, greedy_tokens)
    operation = heard.amend(scenario["old"]["key"], case_records(tokenizer, [scenario["new"]])[0])
    audit = _audit(model, heard, reference, tokenizer, scenario["old"]["question"])
    after = _score_amendment(heard, tokenizer, scenario, greedy_tokens)
    lift = before["old"]["mean_log_probability"] - never_old["mean_log_probability"]
    return {"id": scenario["id"], "status": "complete", "seconds": time.perf_counter() - started,
            "tokens_as_heard": sum(r.n_tokens for r in _tokens(tokenizer, scenario)),
            "replacement_tokens": case_records(tokenizer, [scenario["new"]])[0].n_tokens,
            "qualification": {"old_target_lift_nats_vs_never": lift, "lift_exceeds_one_nat": lift > 1.0,
                              "retained_even_if_unqualified": True},
            "as_heard": before, "corrected_from_start": correct, "after_amendment": after,
            "never_stored_old_target": never_old, "operation": asdict(operation),
            "audit": audit, "numerical_floor_repeat_corrected_rebuild": floor}


def run_sequence(model: Any, tokenizer: Any, first: dict[str, Any], second: dict[str, Any], reverse: bool, greedy_tokens: int = 16) -> dict[str, Any]:
    from kimi_sv.protocol import build_memory, case_records, filler_record, preamble_record
    records = [preamble_record(tokenizer, PREAMBLE)]
    records += case_records(tokenizer, [first["retained"][0], first["old"], first["retained"][1], second["old"], first["retained"][2]])
    if first["suffix_tokens"]:
        records.append(filler_record(tokenizer, first["suffix_tokens"]))
    targets = [first["old"], second["old"]]
    order = list(reversed(targets)) if reverse else targets
    admission = case_records(tokenizer, [first["admission"]])[0]
    memory = build_memory(model, records)
    live = list(records)
    initial = {c["key"]: _score(memory, tokenizer, c, greedy_tokens) for c in targets}
    steps = []
    for action, case in [("delete", order[0]), ("admit", first["admission"]), ("delete", order[1])]:
        if action == "delete":
            report = asdict(memory.delete(case["key"]))
            live = [r for r in live if r.key != case["key"]]
        else:
            t0 = time.perf_counter()
            memory.ingest(admission)
            live.append(admission)
            report = {"key": admission.key, "seconds": time.perf_counter() - t0, "admitted_tokens": admission.n_tokens}
        reference = build_memory(model, live)
        audit = _audit(model, memory, reference, tokenizer, first["old"]["question"])
        repeated = build_memory(model, live)
        floor = _audit(model, repeated, reference, tokenizer, first["old"]["question"])
        del repeated
        probe_cases = targets + first["retained"] + ([first["admission"]] if admission.key in memory.keys else [])
        measured = {c["key"]: _score(memory, tokenizer, c, greedy_tokens) for c in probe_cases}
        expected = {c["key"]: _score(reference, tokenizer, c, greedy_tokens) for c in probe_cases}
        steps.append({"action": action, "record": case["key"], "operation": report,
                      "live_keys": list(memory.keys), "audit": audit,
                      "numerical_floor_repeat_rebuild": floor, "scores": measured, "fresh_reference_scores": expected})
        del reference
    return {"id": f"seq-{first['id']}-{second['id']}-{'BA' if reverse else 'AB'}",
            "status": "complete", "deletion_order": [c["key"] for c in order],
            "initial_target_scores": initial, "steps": steps,
            "all_transitions_equal": all(s["audit"]["all_equal"] for s in steps)}


def _save(out: Path, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit")
    parser.add_argument("--targets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--greedy-tokens", type=int, default=16)
    parser.add_argument("--mode", choices=("amendment", "sequential", "both"), default="both")
    parser.add_argument("--memory-limit-gb", type=float, default=160)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = make_manifest(args.targets, args.seed)
    report = {"schema": "kda-maintenance-cohort-v1", "status": "preselected", "model": args.model,
              "tiny_plumbing_only": args.tiny, "manifest": manifest,
              "protocol": {"mode": args.mode, "greedy_tokens": args.greedy_tokens,
                           "qualification": "all targets retained, old target mean logprob lift > 1 nat vs never-stored flagged",
                           "full_state": "every KDA recurrent and convolution array, logical active MLA keys/values and offsets",
                           "numerical_floor": "independent fresh rebuild repeated under identical chunking and precision"},
              "host": {"platform": platform.platform(), "machine": platform.machine()},
              "amendments": [], "sequences": []}
    if args.out.exists():
        if not args.resume:
            raise ValueError("output already exists; use --resume to preserve completed runs")
        old = json.loads(args.out.read_text())
        if old["manifest"] != manifest or old["protocol"] != report["protocol"] or old["model"] != args.model or old["tiny_plumbing_only"] != args.tiny:
            raise ValueError("resume protocol/model/cohort mismatch")
        report = old
    _save(args.out, report)
    if args.manifest_only:
        print(f"preselected {args.targets} amendment targets: {manifest['sha256']}")
        return 0
    import mlx.core as mx
    from kimi_sv.protocol import load_model
    mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    model, tokenizer, label = load_model(args.model, tiny=args.tiny, lazy=not args.tiny)
    report["loaded_model_label"] = label
    report["status"] = "running"
    _save(args.out, report)
    if args.mode in {"amendment", "both"}:
        completed = {row["id"] for row in report["amendments"]}
        for scenario in manifest["scenarios"]:
            if scenario["id"] in completed:
                continue
            row = run_amendment(model, tokenizer, scenario, args.greedy_tokens)
            report["amendments"].append(row)
            _save(args.out, report)
            print(f"{row['id']}: exact={row['audit']['all_equal']}; lift={row['qualification']['old_target_lift_nats_vs_never']:.3f}; {row['seconds']:.1f}s", flush=True)
            gc.collect()
            mx.clear_cache()
    if args.mode in {"sequential", "both"}:
        completed = {row["id"] for row in report["sequences"]}
        for i in range(0, len(manifest["scenarios"]) - 1, 2):
            first, second = manifest["scenarios"][i:i + 2]
            for reverse in (False, True):
                sid = f"seq-{first['id']}-{second['id']}-{'BA' if reverse else 'AB'}"
                if sid in completed:
                    continue
                row = run_sequence(model, tokenizer, first, second, reverse, args.greedy_tokens)
                report["sequences"].append(row)
                _save(args.out, report)
                print(f"{row['id']}: exact={row['all_transitions_equal']}", flush=True)
                gc.collect()
                mx.clear_cache()
    report["status"] = "complete"
    report["summary"] = {"amendments": len(report["amendments"]), "sequences": len(report["sequences"]),
                         "amendment_audits_all_equal": all(r["audit"]["all_equal"] for r in report["amendments"]),
                         "sequence_audits_all_equal": all(r["all_transitions_equal"] for r in report["sequences"]),
                         "amendment_qualified": sum(r["qualification"]["lift_exceeds_one_nat"] for r in report["amendments"])}
    _save(args.out, report)
    print(json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
