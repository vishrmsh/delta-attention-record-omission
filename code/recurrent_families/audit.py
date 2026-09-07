"""Receipt transport, forcing, and probe recovery across recurrent families.

The protocol mirrors ``kimi_sv.eval_forcing_sweep`` so that Kimi's KDA numbers
and the numbers here are read on the same axes. A victim record is placed after
``P`` prefix records and followed by a fixed suffix; a record-present and a
record-omitted context are advanced one suffix token at a time in lockstep, and
at every cut we record

* the state difference ``||S+ - S-||`` per layer and its size relative to the
  omitted state (the record's imprint);
* the exact decomposition of that difference into the transported boundary
  receipt, transition forcing, and write forcing, carried online so no
  per-token write term has to be stored;
* the residual left by three receipt classes (static, diagonal ledger, full
  transport) and by exact recomputation of the first ``k`` suffix tokens;
* two frozen-input controls that test Corollary 1 directly: the handwritten
  recurrence run from the present state with the present inputs must reach the
  model's own state (recurrence check), and run from the omitted state with the
  present inputs it defines a frozen counterfactual that the ledger (decay
  class) or the full transport (delta class) must reach exactly;
* teacher-forced recovery of the victim's secret from each corrected state.

Predictions for each family are written to ``benchmarks/family_predictions_v1.json``
before any run; the summarizer compares measured verdicts against them.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .adapters import ADAPTERS, FamilyAdapter, StepOps

DEFAULT_CUTS = (0, 16, 64, 256, 1024)
PREAMBLE = (
    "Internal case file. Each entry is one patient record. "
    "Answer questions using only these entries.\n\n"
)
MIN_SIGNAL_NATS = 0.05
MAX_FIRST_TOKEN_RANK = 10


def layer_norms(x: torch.Tensor) -> List[float]:
    flat = x.reshape(x.shape[0], -1).float()
    return torch.linalg.vector_norm(flat, dim=1).tolist()


def _ratio(num: Sequence[float], den: Sequence[float]) -> List[float]:
    return [a / max(b, 1e-12) for a, b in zip(num, den)]


class Decomposition:
    """Online carrier of (transported diff, transition forcing, write forcing, reconstructed omitted)."""

    def __init__(self, adapter: FamilyAdapter, s_plus: torch.Tensor, s_minus: torch.Tensor, start: int):
        self.adapter = adapter
        self.start = start
        self.diff = (s_plus - s_minus).clone()
        self.tf = torch.zeros_like(self.diff)
        self.wf = torch.zeros_like(self.diff)
        self.om = s_minus.clone()

    def update(self, ops_p: StepOps, ops_o: StepOps) -> None:
        a = self.adapter
        present_on_omitted = a.apply_transition(ops_p, self.om)
        omitted_transition = a.apply_transition(ops_o, self.om)
        self.diff = a.apply_transition(ops_p, self.diff)
        self.tf = a.apply_transition(ops_p, self.tf) + (present_on_omitted - omitted_transition)
        self.wf = a.apply_transition(ops_p, self.wf) + a.write_term(ops_p) - a.write_term(ops_o)
        self.om = omitted_transition + a.write_term(ops_o)

    def record(self, s_plus: torch.Tensor, s_minus: torch.Tensor) -> Dict[str, Any]:
        direct = s_plus - s_minus
        forcing = self.tf + self.wf
        rec = {
            "start": self.start,
            "direct_norm": layer_norms(direct),
            "omitted_norm": layer_norms(s_minus),
            "present_norm": layer_norms(s_plus),
            "transported_norm": layer_norms(self.diff),
            "transition_forcing_norm": layer_norms(self.tf),
            "write_forcing_norm": layer_norms(self.wf),
            "forcing_norm": layer_norms(forcing),
        }
        rec["decomposition_error"] = _ratio(layer_norms(self.diff + forcing - direct), rec["direct_norm"])
        rec["omitted_reconstruction_error"] = _ratio(layer_norms(self.om - s_minus), rec["omitted_norm"])
        rec["residual_full_rel_direct"] = _ratio(rec["forcing_norm"], rec["direct_norm"])
        rec["residual_full_rel_omitted"] = _ratio(rec["forcing_norm"], rec["omitted_norm"])
        return rec


def teacher_forced(adapter: FamilyAdapter, cache: Any, prompt: Sequence[int], target: Sequence[int]) -> Dict[str, Any]:
    """Score ``target`` after ``prompt`` from ``cache`` (which is consumed)."""
    logits = None
    for tok in prompt:
        logits, _ = adapter.step(tok, cache)
    logprobs: List[float] = []
    first_rank = None
    first_logprobs = None
    for i, tok in enumerate(target):
        lp = torch.log_softmax(logits, dim=-1)
        if i == 0:
            first_rank = int((lp > lp[tok]).sum().item()) + 1
            first_logprobs = lp.detach().clone()
        logprobs.append(float(lp[tok].item()))
        logits, _ = adapter.step(tok, cache)
    return {
        "mean_log_probability": sum(logprobs) / len(logprobs),
        "total_log_probability": sum(logprobs),
        "first_token_rank": first_rank,
        "first_token_probability": math.exp(logprobs[0]),
        "n_target_tokens": len(target),
        "_first_logprobs": first_logprobs,
    }


def kl(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    p = torch.exp(reference)
    return float((p * (reference - candidate)).sum().item())


@torch.inference_mode()
def evaluate_configuration(
    adapter: FamilyAdapter,
    *,
    prefix_ids: List[int],
    victim_ids: List[int],
    suffix_ids: List[int],
    cuts: Sequence[int],
    victim_case: Dict[str, str],
    retained_case: Optional[Dict[str, str]],
    behavior_trunc: str,
) -> Dict[str, Any]:
    cuts = sorted(set(int(c) for c in cuts) | {0})
    n_max = min(cuts[-1], len(suffix_ids))
    cuts = [c for c in cuts if c <= n_max]

    probed_trunc: Dict[int, set] = {}
    for c in cuts:
        keys = [k for k in cuts if 0 < k < c]
        if behavior_trunc == "all":
            probed_trunc[c] = set(keys)
        elif behavior_trunc == "none":
            probed_trunc[c] = set()
        else:
            probed_trunc[c] = {
                k for target in (c / 2, c / 8) for k in [max([t for t in keys if t <= target], default=None)] if k is not None
            }

    def release() -> None:
        # The naive SSD prefill materializes multi-GiB chunk tensors; the MPS
        # allocator keeps them cached unless told otherwise.
        if adapter.device == "mps":
            torch.mps.empty_cache()

    t0 = time.perf_counter()
    cache_p = adapter.prefill(prefix_ids + victim_ids)
    cache_o = adapter.prefill(prefix_ids)
    prefill_seconds = time.perf_counter() - t0
    release()
    states_p = {0: adapter.states(cache_p)}
    states_o = {0: adapter.states(cache_o)}
    snaps_p = {0: adapter.snapshot(cache_p)}
    snaps_o = {0: adapter.snapshot(cache_o)}
    receipt = states_p[0] - states_o[0]

    # Online carriers.
    decomps: Dict[int, Decomposition] = {0: Decomposition(adapter, states_p[0], states_o[0], 0)}
    ledger_receipt = receipt.clone()  # diagonal-ledger transport of the boundary receipt
    frozen_minus = states_o[0].clone()  # omitted init, present inputs
    frozen_plus = states_p[0].clone()  # present init, present inputs (recurrence check)
    transported_receipt_full = receipt.clone()
    state_records: Dict[str, Any] = {}
    truncated: Dict[str, Dict[str, Any]] = {}
    corrected: Dict[int, Dict[str, torch.Tensor]] = {}
    log_bytes = None
    step_seconds = 0.0
    transport_seconds = 0.0

    for t in range(n_max):
        tok = suffix_ids[t]
        t1 = time.perf_counter()
        _, ops_p = adapter.step(tok, cache_p, capture=True)
        _, ops_o = adapter.step(tok, cache_o, capture=True)
        step_seconds += time.perf_counter() - t1
        if log_bytes is None:
            log_bytes = adapter.transition_log_bytes_per_token(ops_p)
        t2 = time.perf_counter()
        for d in decomps.values():
            d.update(ops_p, ops_o)
        ledger_receipt = adapter.apply_ledger(ops_p, ledger_receipt)
        transported_receipt_full = adapter.apply_transition(ops_p, transported_receipt_full)
        frozen_minus = adapter.apply_transition(ops_p, frozen_minus) + adapter.write_term(ops_p)
        frozen_plus = adapter.apply_transition(ops_p, frozen_plus) + adapter.write_term(ops_p)
        transport_seconds += time.perf_counter() - t2
        c = t + 1
        if c in cuts:
            s_plus, s_minus = adapter.states(cache_p), adapter.states(cache_o)
            states_p[c], states_o[c] = s_plus, s_minus
            snaps_p[c], snaps_o[c] = adapter.snapshot(cache_p), adapter.snapshot(cache_o)
            rec = decomps[0].record(s_plus, s_minus)
            static_res = (s_plus - receipt) - s_minus
            ledger_res = (s_plus - ledger_receipt) - s_minus
            frozen_diff = s_plus - frozen_minus
            rec.update(
                {
                    "imprint_fraction": _ratio(rec["direct_norm"], rec["omitted_norm"]),
                    "residual_static_rel_direct": _ratio(layer_norms(static_res), rec["direct_norm"]),
                    "residual_static_rel_omitted": _ratio(layer_norms(static_res), rec["omitted_norm"]),
                    "residual_ledger_rel_direct": _ratio(layer_norms(ledger_res), rec["direct_norm"]),
                    "residual_ledger_rel_omitted": _ratio(layer_norms(ledger_res), rec["omitted_norm"]),
                    # Frozen-input controls (Corollary 1). Errors are reported both
                    # relative to the frozen difference (a share) and relative to the
                    # omitted state (an absolute scale). In layers whose decay drives
                    # the frozen difference toward zero over a long suffix, the share
                    # is a ratio of two vanishing norms; the state-relative form is
                    # the one to judge exactness by.
                    "recurrence_vs_model_error": _ratio(layer_norms(frozen_plus - s_plus), rec["present_norm"]),
                    "frozen_diff_norm": layer_norms(frozen_diff),
                    "frozen_diff_rel_omitted": _ratio(layer_norms(frozen_diff), rec["omitted_norm"]),
                    "frozen_full_transport_error": _ratio(
                        layer_norms(transported_receipt_full - frozen_diff), layer_norms(frozen_diff)
                    ),
                    "frozen_full_transport_error_rel_omitted": _ratio(
                        layer_norms(transported_receipt_full - frozen_diff), rec["omitted_norm"]
                    ),
                    "frozen_ledger_transport_error": _ratio(
                        layer_norms(ledger_receipt - frozen_diff), layer_norms(frozen_diff)
                    ),
                    "frozen_ledger_transport_error_rel_omitted": _ratio(
                        layer_norms(ledger_receipt - frozen_diff), rec["omitted_norm"]
                    ),
                    "frozen_static_error": _ratio(layer_norms(receipt - frozen_diff), layer_norms(frozen_diff)),
                    "frozen_static_error_rel_omitted": _ratio(layer_norms(receipt - frozen_diff), rec["omitted_norm"]),
                }
            )
            state_records[str(c)] = rec
            corrected[c] = {
                "static": s_plus - receipt,
                "ledger": s_plus - ledger_receipt,
                "full": s_plus - decomps[0].diff,
            }
            for k, d in decomps.items():
                if k == 0:
                    continue
                r = d.record(s_plus, s_minus)
                truncated.setdefault(str(k), {})[str(c)] = {
                    "residual_rel_direct": r["residual_full_rel_direct"],
                    "residual_rel_omitted": r["residual_full_rel_omitted"],
                    "forcing_norm": r["forcing_norm"],
                }
                if k in probed_trunc.get(c, set()):
                    corrected[c][f"trunc:{k}"] = s_plus - d.diff
            if c < cuts[-1]:
                decomps[c] = Decomposition(adapter, s_plus, s_minus, c)
            release()

    state_records["0"] = {
        "start": 0,
        "direct_norm": layer_norms(receipt),
        "omitted_norm": layer_norms(states_o[0]),
        "present_norm": layer_norms(states_p[0]),
        "imprint_fraction": _ratio(layer_norms(receipt), layer_norms(states_o[0])),
    }

    # Carriers are no longer needed once every cut has been recorded.
    decomps.clear()
    del frozen_minus, frozen_plus, transported_receipt_full, ledger_receipt
    release()

    # --- replay identity: the omitted prefill + suffix steps is the replay; check determinism ---
    cache_r = adapter.prefill(prefix_ids)
    for t in range(n_max):
        adapter.step(suffix_ids[t], cache_r)
    replay_diff = float((adapter.states(cache_r) - states_o[n_max]).abs().max().item()) if n_max > 0 else 0.0
    del cache_r
    release()

    # --- behavior --------------------------------------------------------------
    q_ids = adapter.encode("\n" + victim_case["question"])
    s_ids = adapter.encode(victim_case["secret"])
    rq_ids = adapter.encode("\n" + retained_case["question"]) if retained_case else None
    rs_ids = adapter.encode(retained_case["secret"]) if retained_case else None
    behavior: Dict[str, Any] = {}
    for c in cuts:
        conditions: Dict[str, Tuple[Any, Optional[torch.Tensor]]] = {
            "present": (snaps_p[c], states_p[c]),
            "omitted": (snaps_o[c], states_o[c]),
        }
        for name, stacked in corrected.get(c, {}).items():
            conditions[name] = (snaps_p[c], stacked)
        raw: Dict[str, Dict[str, Any]] = {}
        for name, (snap, stacked) in conditions.items():
            cache = adapter.restore(snap)
            if name not in ("present", "omitted"):
                adapter.inject(cache, stacked)
            victim = teacher_forced(adapter, cache, q_ids, s_ids)
            entry: Dict[str, Any] = {"victim": victim, "_stacked": stacked}
            if rq_ids:
                cache2 = adapter.restore(snap)
                if name not in ("present", "omitted"):
                    adapter.inject(cache2, stacked)
                entry["retained"] = teacher_forced(adapter, cache2, rq_ids, rs_ids)
            raw[name] = entry
        ref = raw["omitted"]["victim"]
        out: Dict[str, Any] = {}
        for name, entry in raw.items():
            v = entry["victim"]
            rec = {
                "victim": {k: val for k, val in v.items() if not k.startswith("_")},
                "lift_vs_omitted_nats": v["mean_log_probability"] - ref["mean_log_probability"],
                "lift_vs_present_nats": v["mean_log_probability"] - raw["present"]["victim"]["mean_log_probability"],
                "first_token_kl_omitted_to_condition": kl(ref["_first_logprobs"], v["_first_logprobs"]),
            }
            if entry["_stacked"] is not None:
                diff = layer_norms(entry["_stacked"] - states_o[c])
                rel = _ratio(diff, layer_norms(states_o[c]))
                rec["state_residual_vs_omitted"] = {"max": max(rel), "mean": sum(rel) / len(rel)}
            if "retained" in entry:
                r = entry["retained"]
                rec["retained"] = {k: val for k, val in r.items() if not k.startswith("_")}
                rec["retained_lift_vs_omitted_nats"] = (
                    r["mean_log_probability"] - raw["omitted"]["retained"]["mean_log_probability"]
                )
            out[name] = rec
        behavior[str(c)] = out
        corrected.pop(c, None)
        raw.clear()
        release()

    lift0 = behavior["0"]["present"]["lift_vs_omitted_nats"]
    rank0 = behavior["0"]["present"]["victim"]["first_token_rank"]
    return {
        "tokens": {
            "prefix_total": len(prefix_ids),
            "victim": len(victim_ids),
            "suffix_available": len(suffix_ids),
            "cuts": cuts,
        },
        "timing": {
            "prefill_seconds": prefill_seconds,
            "step_seconds_per_token_both_runs": step_seconds / max(n_max, 1),
            "transport_seconds_per_token_all_carriers": transport_seconds / max(n_max, 1),
        },
        "storage": {
            "transition_log_bytes_per_token_all_layers": log_bytes,
            "state_bytes_all_layers": int(states_p[0].numel() * 4),
        },
        "admission": {
            "lift_nats": lift0,
            "first_token_rank": rank0,
            "qualified": lift0 >= MIN_SIGNAL_NATS and rank0 <= MAX_FIRST_TOKEN_RANK,
        },
        "replay_check": {"cut": n_max, "max_abs_state_diff_vs_omitted": replay_diff},
        "state": state_records,
        "truncated": truncated,
        "behavior": behavior,
    }


def build_records(adapter: FamilyAdapter, n_cases: int, n_victims: int, max_prefix: int):
    from kimi_sv.protocol import generated_cases

    cases = generated_cases(n_cases)
    victims = cases[:n_victims]
    prefix_pool = cases[n_victims : n_victims + max_prefix]
    suffix_pool = cases[n_victims + max_prefix :]
    return victims, prefix_pool, suffix_pool


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--n-cases", type=int, default=192)
    ap.add_argument("--victims", default="0,1,2,3")
    ap.add_argument("--prefix-depths", default="0,4,16")
    ap.add_argument("--cuts", default=",".join(str(c) for c in DEFAULT_CUTS))
    ap.add_argument("--behavior-trunc", choices=("none", "half", "all"), default="half")
    ap.add_argument("--out-dir", default="outputs/recurrent_families_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    victims = [int(x) for x in args.victims.split(",")]
    depths = sorted({int(x) for x in args.prefix_depths.split(",")})
    cuts = sorted({int(x) for x in args.cuts.split(",")} | {0})

    adapter = ADAPTERS[args.family](device=args.device)
    t0 = time.perf_counter()
    adapter.load(args.repo)
    print(f"loaded {adapter.repo} ({args.family}) in {time.perf_counter() - t0:.1f}s on {args.device}", flush=True)

    victim_cases, prefix_pool, suffix_pool = build_records(adapter, args.n_cases, max(victims) + 1, max(depths))
    suffix_ids: List[int] = []
    used = 0
    for case in suffix_pool:
        suffix_ids += adapter.encode(case["text"])
        used += 1
        if len(suffix_ids) >= max(cuts):
            break
    out_dir = Path(args.out_dir) / args.family
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in victims:
        for p in depths:
            out = out_dir / f"victim{v:03d}-prefix{p:03d}.json"
            if out.exists() and not args.overwrite:
                print(f"exists; skipping {out}", flush=True)
                continue
            case = victim_cases[v]
            prefix_ids = adapter.encode(PREAMBLE, special=True)
            for rec in prefix_pool[:p]:
                prefix_ids += adapter.encode(rec["text"])
            victim_ids = adapter.encode(case["text"])
            retained = prefix_pool[p - 1] if p > 0 else None
            t1 = time.perf_counter()
            result = evaluate_configuration(
                adapter,
                prefix_ids=prefix_ids,
                victim_ids=victim_ids,
                suffix_ids=suffix_ids,
                cuts=cuts,
                victim_case=case,
                retained_case=retained,
                behavior_trunc=args.behavior_trunc,
            )
            report = {
                "schema": "recurrent-families-audit-v1",
                "family": args.family,
                "transition_class": adapter.transition_class,
                "has_attention": adapter.has_attention,
                "model": adapter.repo,
                "device": args.device,
                "host": {"platform": platform.platform(), "machine": platform.machine()},
                "source": "synthetic",
                "contains_source_text": False,
                "victim_index": v,
                "victim_key": case["key"],
                "prefix_depth": p,
                "suffix_records_used": used,
                "recurrent_layers": adapter.recurrent_layers(),
                "protocol": {"cuts": cuts, "behavior_trunc": args.behavior_trunc},
                "seconds": time.perf_counter() - t1,
                **result,
            }
            out.write_text(json.dumps(report, indent=1))
            s = report["state"][str(cuts[-1] if str(cuts[-1]) in report["state"] else max(int(k) for k in report["state"]))]
            fl = s.get("frozen_ledger_transport_error_rel_omitted", [float("nan")])
            ff = s.get("frozen_full_transport_error_rel_omitted", [float("nan")])
            rf = s.get("residual_full_rel_direct", [float("nan")])
            rc = s.get("recurrence_vs_model_error", [float("nan")])
            print(
                f"{args.family} victim {v} prefix {p}: {report['seconds'] / 60:.1f} min; admitted={report['admission']['qualified']};"
                f" recurrence err max {max(rc):.1e}; frozen ledger err/state max {max(fl):.2e}; frozen full err/state max {max(ff):.2e};"
                f" native forcing/direct median {sorted(rf)[len(rf) // 2]:.3f}; replay diff {report['replay_check']['max_abs_state_diff_vs_omitted']:.1e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
