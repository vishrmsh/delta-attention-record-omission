"""How a record's imprint on KDA state, and what a receipt can undo, change with the suffix.

The published transport audit (``eval_separability``) measured, at one suffix
length, that a boundary receipt transported through the record-present
transitions misses the record-omitted state by a relative residual of 0.064 to
1.010. This script asks the questions that audit left open, on the same model
and the same kernel-input capture:

1. Does the record's imprint on the recurrent state, ``||S+ - S-|| / ||S-||``,
   decay as more tokens follow it, or does it persist?
2. Does the share of the final difference that transport cannot explain (the
   accumulated forcing) grow, shrink, or hold steady with suffix length?
3. If a system recomputes only the first ``k`` suffix tokens exactly and
   transports the remaining difference, how fast does the residual fall in
   ``k``? This is the price curve for approximate deletion.
4. What does each approximately corrected state look like behaviorally: how
   much of the record's secret is still recoverable by a teacher-forced probe,
   a first-token rank, and a greedy answer, with and without the record's
   attention rows masked?

One configuration is a victim record placed after ``P`` prefix records and
followed by a fixed suffix. The suffix is ingested in chunks whose boundaries
are the requested cut points, so one record-present run and one record-omitted
run give the state at every suffix length at once. Kernel inputs (k, v, g,
beta) are captured for all 20 KDA layers on both runs, and the exact
decomposition of the state difference into transported receipt, transition
forcing, and write forcing is carried token by token with the norms recorded at
each cut. All twenty layers are propagated together as one stacked array.

Every number written is a norm, a ratio, a probability, a rank, or a token
count. Greedy answers are recorded only for the public sources.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

import mlx_lm.models.kimi_linear as kimi_linear_module
from mlx_lm.models.gated_delta import compute_g

from .mla_mask import install_mla_mask, set_mla_drop_positions
from .probes import ProbeStats, lift_nats, teacher_forced
from .protocol import (
    CASE_SOURCES,
    DEFAULT_MODEL,
    PUBLIC_SOURCES,
    answer_text,
    case_records,
    encode,
    load_model,
    load_source_cases,
    preamble_record,
    source_preamble,
)
from .records import Record, RecordMemory
from .state import Boundary, classify_layers, restore, snapshot

RECURRENT = 3
DEFAULT_CUTS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
MIN_SIGNAL_NATS = 0.05
MAX_FIRST_TOKEN_RANK = 10


# --------------------------------------------------------------------------
# kernel-input capture over a chunked prefill
# --------------------------------------------------------------------------


class AllLayerTap:
    """Capture k, v, g, beta for every KDA layer across many forward passes.

    KDA layers call ``gated_delta_update`` once per forward in layer order, so
    the call index modulo the number of KDA layers identifies the layer. The
    tap accumulates one chunk per forward while active and stacks them later.
    """

    def __init__(self, n_layers: int):
        self.real = kimi_linear_module.gated_delta_update
        self.n = n_layers
        self.calls = 0
        self.active = False
        self.chunks: Dict[int, Dict[str, List[mx.array]]] = {}
        self.reset()

    def reset(self) -> None:
        self.chunks = {i: {"k": [], "v": [], "g": [], "beta": []} for i in range(self.n)}
        self.calls = 0

    def install(self) -> None:
        def wrapper(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
            idx = self.calls % self.n
            self.calls += 1
            if self.active:
                g = compute_g(A_log, a, dt_bias)
                beta = mx.sigmoid(b)
                mx.eval(k, v, g, beta)
                chunk = self.chunks[idx]
                chunk["k"].append(k)
                chunk["v"].append(v)
                chunk["g"].append(g)
                chunk["beta"].append(beta)
            return self.real(
                q, k, v, a, b, A_log, dt_bias, state=state, mask=mask, use_kernel=use_kernel
            )

        kimi_linear_module.gated_delta_update = wrapper

    def uninstall(self) -> None:
        kimi_linear_module.gated_delta_update = self.real

    def stacked(self) -> Dict[str, mx.array]:
        """``{field: array}`` with shape ``[n_layers, T, ...]`` (batch dim dropped)."""
        out: Dict[str, mx.array] = {}
        for name in ("k", "v", "g", "beta"):
            per_layer = [
                mx.concatenate(self.chunks[i][name], axis=1)[0] for i in range(self.n)
            ]
            out[name] = mx.stack(per_layer, axis=0)
        mx.eval(*out.values())
        return out


# --------------------------------------------------------------------------
# chunked context construction
# --------------------------------------------------------------------------


@dataclass
class RunTrace:
    """One record-present or record-omitted context, with checkpoints at every cut."""

    cache: List[Any]
    boundaries: Dict[str, Boundary] = field(default_factory=dict)
    states: Dict[str, mx.array] = field(default_factory=dict)  # stacked fp32 KDA states
    inputs: Dict[str, mx.array] = field(default_factory=dict)  # suffix kernel inputs
    suffix_seconds: float = 0.0
    suffix_tokens: int = 0


def stacked_state(cache: Sequence[Any], kda: Sequence[int]) -> mx.array:
    arrays = [cache[i][RECURRENT][0].astype(mx.float32) for i in kda]
    out = mx.stack(arrays, axis=0)
    mx.eval(out)
    return out


def chunk_bounds(cuts: Sequence[int]) -> List[Tuple[int, int]]:
    cuts = sorted(set(int(c) for c in cuts))
    if cuts[0] != 0:
        cuts = [0] + cuts
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def run_context(
    model: Any,
    kda: Sequence[int],
    before: Sequence[mx.array],
    suffix: mx.array,
    cuts: Sequence[int],
    tap: AllLayerTap,
    *,
    mark_after: Optional[int] = None,
) -> RunTrace:
    """Ingest ``before`` segments one by one, then the suffix in cut-aligned chunks.

    ``mark_after`` names the index in ``before`` after which the "prefix"
    checkpoint is taken (the record-present run marks the state before the
    victim). The state at the end of ``before`` is stored as ``cut:0``.
    """
    cache = model.make_cache()
    trace = RunTrace(cache=cache)

    def forward(tokens: mx.array) -> None:
        logits = model(tokens, cache=cache)
        mx.eval(logits)

    tap.active = False
    for idx, seg in enumerate(before):
        forward(seg)
        if mark_after is not None and idx == mark_after:
            trace.boundaries["prefix"] = snapshot(model, cache, label="prefix")
            trace.states["prefix"] = stacked_state(cache, kda)
    if "prefix" not in trace.boundaries:
        trace.boundaries["prefix"] = snapshot(model, cache, label="prefix")
        trace.states["prefix"] = stacked_state(cache, kda)
    trace.boundaries["cut:0"] = snapshot(model, cache, label="cut:0")
    trace.states["cut:0"] = stacked_state(cache, kda)

    tap.reset()
    tap.active = True
    t0 = time.perf_counter()
    n_tokens = 0
    for a, b in chunk_bounds(cuts):
        if b > int(suffix.shape[1]):
            break
        forward(suffix[:, a:b])
        n_tokens += b - a
        trace.boundaries[f"cut:{b}"] = snapshot(model, cache, label=f"cut:{b}")
        trace.states[f"cut:{b}"] = stacked_state(cache, kda)
    trace.suffix_seconds = time.perf_counter() - t0
    trace.suffix_tokens = n_tokens
    tap.active = False
    trace.inputs = tap.stacked() if n_tokens else {}
    return trace


# --------------------------------------------------------------------------
# stacked KDA algebra (all layers at once)
# --------------------------------------------------------------------------


@mx.compile
def _step_transition(k, g, beta, state):
    """``state * diag(g) * (I - beta k k^T)`` for ``state[n,H,Dv,Dk]``."""
    decayed = state * g[..., None, :]
    recalled = (decayed * k[..., None, :]).sum(axis=-1)
    correction = k[..., None, :] * (recalled * beta[..., None])[..., None]
    return decayed - correction


@mx.compile
def _step_write(k, v, beta):
    return k[..., None, :] * (v * beta[..., None])[..., None]


def layer_norms(x: mx.array) -> List[float]:
    n = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=(1, 2, 3)))
    mx.eval(n)
    return [float(v) for v in n.tolist()]


def _ratio(num: Sequence[float], den: Sequence[float]) -> List[float]:
    return [a / max(b, 1e-12) for a, b in zip(num, den)]


def _slice(inp: Dict[str, mx.array], t: int) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    return (
        inp["k"][:, t].astype(mx.float32),
        inp["v"][:, t].astype(mx.float32),
        inp["g"][:, t].astype(mx.float32),
        inp["beta"][:, t].astype(mx.float32),
    )


def decompose_from(
    start: int,
    present: RunTrace,
    omitted: RunTrace,
    cuts: Sequence[int],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, mx.array], float]:
    """Propagate the exact difference decomposition from suffix index ``start``.

    Starting from the exact record-present and record-omitted states at
    ``start`` (both known from the chunked runs), carry the transported
    difference, the transition forcing, and the write forcing through the
    record-present transitions, and reconstruct the omitted state through its
    own transitions. At every later cut record per-layer norms, plus the
    residual left by subtracting the transported ``start`` difference from the
    present state (which is the accumulated forcing after ``start``). For
    ``start == 0`` this is the boundary-receipt transport of Theorem 1, and
    the static and diagonal-ledger corrections are recorded as well.

    Returns ``(per_cut_records, corrected_full_states, seconds_per_token)``.
    """
    key0 = f"cut:{start}"
    diff = present.states[key0] - omitted.states[key0]
    tf = mx.zeros_like(diff)
    wf = mx.zeros_like(diff)
    om = omitted.states[key0]
    ledger = mx.ones_like(present.inputs["g"][:, 0].astype(mx.float32)) if start == 0 else None
    n_max = max(c for c in cuts if f"cut:{c}" in present.states)
    records: Dict[int, Dict[str, Any]] = {}
    corrected_full: Dict[int, mx.array] = {}
    cut_set = set(int(c) for c in cuts)
    t0 = time.perf_counter()
    steps = 0
    for t in range(start, n_max):
        kp, vp, gp, bp = _slice(present.inputs, t)
        ko, vo, go, bo = _slice(omitted.inputs, t)
        present_on_omitted = _step_transition(kp, gp, bp, om)
        omitted_transition = _step_transition(ko, go, bo, om)
        diff = _step_transition(kp, gp, bp, diff)
        tf = _step_transition(kp, gp, bp, tf) + (present_on_omitted - omitted_transition)
        wf = _step_transition(kp, gp, bp, wf) + _step_write(kp, vp, bp) - _step_write(ko, vo, bo)
        om = omitted_transition + _step_write(ko, vo, bo)
        if ledger is not None:
            ledger = ledger * gp
        mx.eval(diff, tf, wf, om)
        steps += 1
        c = t + 1
        if c in cut_set:
            key = f"cut:{c}"
            s_plus, s_minus = present.states[key], omitted.states[key]
            direct = s_plus - s_minus
            forcing = tf + wf
            rec: Dict[str, Any] = {
                "start": start,
                "direct_norm": layer_norms(direct),
                "omitted_norm": layer_norms(s_minus),
                "present_norm": layer_norms(s_plus),
                "transported_norm": layer_norms(diff),
                "transition_forcing_norm": layer_norms(tf),
                "write_forcing_norm": layer_norms(wf),
                "forcing_norm": layer_norms(forcing),
                "decomposition_error": _ratio(layer_norms(diff + forcing - direct), layer_norms(direct)),
                "omitted_reconstruction_error": _ratio(layer_norms(om - s_minus), layer_norms(s_minus)),
            }
            # Residual left after subtracting the transported `start` difference
            # from the present state: exactly the accumulated forcing.
            rec["residual_full_rel_direct"] = _ratio(rec["forcing_norm"], rec["direct_norm"])
            rec["residual_full_rel_omitted"] = _ratio(rec["forcing_norm"], rec["omitted_norm"])
            corrected_full[c] = s_plus - diff
            if start == 0:
                receipt = present.states["cut:0"] - omitted.states["cut:0"]
                static_res = (s_plus - receipt) - s_minus
                ledger_res = (s_plus - receipt * ledger[..., None, :]) - s_minus
                rec["residual_static_rel_direct"] = _ratio(layer_norms(static_res), rec["direct_norm"])
                rec["residual_static_rel_omitted"] = _ratio(layer_norms(static_res), rec["omitted_norm"])
                rec["residual_ledger_rel_direct"] = _ratio(layer_norms(ledger_res), rec["direct_norm"])
                rec["residual_ledger_rel_omitted"] = _ratio(layer_norms(ledger_res), rec["omitted_norm"])
                rec["imprint_fraction"] = _ratio(rec["direct_norm"], rec["omitted_norm"])
                rec["corrected_static"] = s_plus - receipt
                rec["corrected_ledger"] = s_plus - receipt * ledger[..., None, :]
            records[c] = rec
    seconds_per_token = (time.perf_counter() - t0) / max(steps, 1)
    return records, corrected_full, seconds_per_token


# --------------------------------------------------------------------------
# behavioral probes on injected states
# --------------------------------------------------------------------------


def with_recurrent(base: Boundary, kda: Sequence[int], stacked: mx.array) -> Boundary:
    """A copy of ``base`` whose recurrent KDA arrays are replaced by ``stacked``."""
    new_kda = dict(base.kda)
    for pos, i in enumerate(kda):
        arrays = list(new_kda[i])
        original = arrays[RECURRENT]
        replacement = stacked[pos][None].astype(original.dtype)
        mx.eval(replacement)
        arrays[RECURRENT] = replacement
        new_kda[i] = tuple(arrays)
    return Boundary(base.token_offset, new_kda, base.mla_offset, label=f"{base.label}|injected")


def _probe_case(memory: RecordMemory, tokenizer: Any, case: Dict[str, Any]) -> ProbeStats:
    return teacher_forced(
        memory,
        encode(tokenizer, "\n" + case["question"] + case.get("body_stem", "")),
        encode(tokenizer, case["secret"]),
    )


def _first_token_logprobs(memory: RecordMemory, tokenizer: Any, case: Dict[str, Any]) -> mx.array:
    logits = memory.probe(encode(tokenizer, "\n" + case["question"] + case.get("body_stem", "")))
    logits = logits[0].astype(mx.float32)
    out = logits - mx.logsumexp(logits)
    mx.eval(out)
    return out


def kl_from_logprobs(reference: mx.array, candidate: mx.array) -> float:
    """KL(reference || candidate) over the full vocabulary, in nats."""
    p = mx.exp(reference)
    val = mx.sum(p * (reference - candidate))
    mx.eval(val)
    return float(val.item())


def probe_condition(
    model: Any,
    cache: List[Any],
    boundary: Boundary,
    drop_pos: Optional[List[int]],
    tokenizer: Any,
    victim: Dict[str, Any],
    retained: Optional[Dict[str, Any]],
    *,
    greedy: bool,
) -> Dict[str, Any]:
    restore(model, cache, boundary)
    set_mla_drop_positions(model, drop_pos)
    try:
        memory = RecordMemory(model, cache=cache)
        out: Dict[str, Any] = {"victim": _probe_case(memory, tokenizer, victim).as_dict()}
        out["victim_first_logprobs"] = _first_token_logprobs(memory, tokenizer, victim)
        if retained is not None:
            out["retained"] = _probe_case(memory, tokenizer, retained).as_dict()
        if greedy:
            out["greedy"] = answer_text(memory, tokenizer, victim["question"])
    finally:
        set_mla_drop_positions(model, None)
    return out


def state_residual(candidate: mx.array, reference: mx.array) -> Dict[str, float]:
    diff = layer_norms(candidate - reference)
    ref = layer_norms(reference)
    rel = _ratio(diff, ref)
    return {"max": max(rel), "mean": sum(rel) / len(rel)}


# --------------------------------------------------------------------------
# one configuration
# --------------------------------------------------------------------------


def evaluate_configuration(
    model: Any,
    tokenizer: Any,
    kda: Sequence[int],
    *,
    preamble: Record,
    prefix_records: Sequence[Record],
    victim_record: Record,
    victim_case: Dict[str, Any],
    retained_case: Optional[Dict[str, Any]],
    suffix: mx.array,
    cuts: Sequence[int],
    behavior_cuts: Sequence[int],
    behavior_trunc: str,
    greedy: bool,
    replay_check: bool,
) -> Dict[str, Any]:
    tap = AllLayerTap(len(kda))
    tap.install()
    try:
        before_present = [preamble.tokens, *[r.tokens for r in prefix_records], victim_record.tokens]
        before_omitted = [preamble.tokens, *[r.tokens for r in prefix_records]]
        present = run_context(
            model, kda, before_present, suffix, cuts, tap, mark_after=len(before_present) - 2
        )
        omitted = run_context(model, kda, before_omitted, suffix, cuts, tap)
    finally:
        tap.uninstall()

    available_cuts = sorted(int(k.split(":")[1]) for k in present.states if k.startswith("cut:"))
    present_log_bytes = None
    if present.inputs:
        k0, g0, b0_ = present.inputs["k"][0, 0], present.inputs["g"][0, 0], present.inputs["beta"][0, 0]
        per_layer = k0.nbytes + g0.nbytes + b0_.nbytes
        present_log_bytes = {
            "per_token_per_layer_native_dtype": int(per_layer),
            "per_token_all_layers_native_dtype": int(per_layer * len(kda)),
            "per_token_all_layers_fp32": int(4 * (k0.size + g0.size + b0_.size) * len(kda)),
            "dtype": str(k0.dtype),
        }
    n_prefix_tokens = int(present.boundaries["prefix"].token_offset)
    n_victim_tokens = victim_record.n_tokens
    victim_positions = list(range(n_prefix_tokens, n_prefix_tokens + n_victim_tokens))

    # --- state-level curves ---------------------------------------------
    # Corrected states are kept only where a behavioral probe will read them:
    # at the probed cuts, and for the truncated-recompute starts the probe
    # policy selects. Everything else is reduced to norms immediately, which
    # keeps a worker near the size of its weights (the September 1 crash came
    # from retaining every corrected state for every cut).
    probed = set(behavior_cuts) & set(available_cuts)

    def chosen_trunc(c: int) -> set:
        keys = [k for k in available_cuts if 0 < k < c]
        if behavior_trunc == "all":
            return set(keys)
        if behavior_trunc == "none":
            return set()
        return {k for target in (c / 2, c / 8) for k in [max([t for t in keys if t <= target], default=None)] if k is not None}

    state_records: Dict[str, Any] = {}
    corrected: Dict[int, Dict[str, mx.array]] = {c: {} for c in available_cuts}
    transport_seconds_per_token = None
    if len(available_cuts) > 1:
        recs, full_states, transport_seconds_per_token = decompose_from(0, present, omitted, available_cuts)
        for c, rec in recs.items():
            static_state = rec.pop("corrected_static")
            ledger_state = rec.pop("corrected_ledger")
            if c in probed:
                corrected[c]["full"] = full_states[c]
                corrected[c]["static"] = static_state
                corrected[c]["ledger"] = ledger_state
            state_records[str(c)] = rec
        del full_states
        truncated: Dict[str, Dict[str, Any]] = {}
        for k in available_cuts:
            if k == 0 or k >= available_cuts[-1]:
                continue
            recs_k, full_k, _ = decompose_from(k, present, omitted, available_cuts)
            truncated[str(k)] = {
                str(c): {
                    "residual_rel_direct": r["residual_full_rel_direct"],
                    "residual_rel_omitted": r["residual_full_rel_omitted"],
                    "forcing_norm": r["forcing_norm"],
                }
                for c, r in recs_k.items()
            }
            for c, s in full_k.items():
                if c in probed and k in chosen_trunc(c):
                    corrected[c][f"trunc:{k}"] = s
            del full_k
    else:
        truncated = {}
    # The captured kernel inputs (up to a few GiB per run) are no longer needed.
    present.inputs = {}
    omitted.inputs = {}
    mx.clear_cache()
    receipt = present.states["cut:0"] - omitted.states["cut:0"]
    state_records["0"] = {
        "start": 0,
        "direct_norm": layer_norms(receipt),
        "omitted_norm": layer_norms(omitted.states["cut:0"]),
        "present_norm": layer_norms(present.states["cut:0"]),
        "imprint_fraction": _ratio(layer_norms(receipt), layer_norms(omitted.states["cut:0"])),
    }

    # --- behavior ---------------------------------------------------------
    behavior: Dict[str, Any] = {}
    for c in sorted(set(behavior_cuts) & set(available_cuts), reverse=True):
        key = f"cut:{c}"
        base_p = present.boundaries[key]
        base_o = omitted.boundaries[key]
        retained_here = retained_case if (retained_case is not None) else None
        conditions: Dict[str, Tuple[List[Any], Boundary, Optional[List[int]], Optional[mx.array]]] = {
            "present": (present.cache, base_p, None, present.states[key]),
            "omitted": (omitted.cache, base_o, None, omitted.states[key]),
            "mask": (present.cache, base_p, victim_positions, present.states[key]),
        }
        for name in ("static", "ledger", "full"):
            if name in corrected[c]:
                conditions[name] = (
                    present.cache,
                    with_recurrent(base_p, kda, corrected[c][name]),
                    None,
                    corrected[c][name],
                )
        if "full" in corrected[c]:
            conditions["mask_full"] = (
                present.cache,
                with_recurrent(base_p, kda, corrected[c]["full"]),
                victim_positions,
                corrected[c]["full"],
            )
        chosen = sorted(int(n.split(":")[1]) for n in corrected[c] if n.startswith("trunc:"))
        for k in chosen:
            conditions[f"trunc:{k}"] = (
                present.cache,
                with_recurrent(base_p, kda, corrected[c][f"trunc:{k}"]),
                None,
                corrected[c][f"trunc:{k}"],
            )
            conditions[f"mask_trunc:{k}"] = (
                present.cache,
                with_recurrent(base_p, kda, corrected[c][f"trunc:{k}"]),
                victim_positions,
                corrected[c][f"trunc:{k}"],
            )

        results: Dict[str, Any] = {}
        raw: Dict[str, Dict[str, Any]] = {}
        for name, (cache, boundary, drop, stacked) in conditions.items():
            raw[name] = probe_condition(
                model, cache, boundary, drop, tokenizer, victim_case, retained_here, greedy=greedy
            )
            raw[name]["_stacked"] = stacked
        ref_lp = raw["omitted"]["victim_first_logprobs"]
        present_stats = ProbeStats(**raw["present"]["victim"])
        omitted_stats = ProbeStats(**raw["omitted"]["victim"])
        for name, r in raw.items():
            stats = ProbeStats(**r["victim"])
            entry: Dict[str, Any] = {
                "victim": r["victim"],
                "lift_vs_omitted_nats": lift_nats(stats, omitted_stats),
                "lift_vs_present_nats": lift_nats(stats, present_stats),
                "first_token_kl_omitted_to_condition": kl_from_logprobs(ref_lp, r["victim_first_logprobs"]),
                "state_residual_vs_omitted": state_residual(r["_stacked"], omitted.states[key]),
            }
            if "retained" in r:
                entry["retained"] = r["retained"]
                entry["retained_lift_vs_omitted_nats"] = lift_nats(
                    ProbeStats(**r["retained"]), ProbeStats(**raw["omitted"]["retained"])
                )
            if "greedy" in r:
                entry["greedy"] = r["greedy"]
                entry["greedy_exact"] = victim_case["secret"].strip().lower() in r["greedy"].lower()
            results[name] = entry
        behavior[str(c)] = results

    # --- replay identity on this context ----------------------------------
    # Runs last: replaying the suffix from the pre-victim checkpoint overwrites
    # the present cache's attention buffer, which the probes above still need.
    replay: Dict[str, Any] = {}
    if replay_check and available_cuts[-1] > 0:
        restore(model, present.cache, present.boundaries["prefix"])
        for a, b in chunk_bounds(available_cuts):
            logits = model(suffix[:, a:b], cache=present.cache)
            mx.eval(logits)
        replayed = stacked_state(present.cache, kda)
        diff = mx.max(mx.abs(replayed - omitted.states[f"cut:{available_cuts[-1]}"]))
        mx.eval(diff)
        replay = {"cut": available_cuts[-1], "max_abs_state_diff_vs_omitted": float(diff.item())}

    # --- admission and bookkeeping --------------------------------------
    b0 = behavior.get("0") or behavior.get(str(min(behavior_cuts))) if behavior else None
    admission: Dict[str, Any] = {}
    if b0 is not None:
        lift0 = b0["present"]["lift_vs_omitted_nats"]
        rank0 = b0["present"]["victim"]["first_token_rank"]
        admission = {
            "lift_nats": lift0,
            "first_token_rank": rank0,
            "qualified": lift0 >= MIN_SIGNAL_NATS and rank0 <= MAX_FIRST_TOKEN_RANK,
            "rule": f"lift >= {MIN_SIGNAL_NATS} nats and first-token rank <= {MAX_FIRST_TOKEN_RANK} at the smallest probed cut",
        }

    log_bytes = present_log_bytes

    return {
        "tokens": {
            "preamble": preamble.n_tokens,
            "prefix_records": len(prefix_records),
            "prefix_total": n_prefix_tokens,
            "victim": n_victim_tokens,
            "suffix_available": int(suffix.shape[1]),
            "cuts": available_cuts,
        },
        "victim_positions": [victim_positions[0], victim_positions[-1]],
        "timing": {
            "present_suffix_seconds": present.suffix_seconds,
            "omitted_suffix_seconds": omitted.suffix_seconds,
            "prefill_seconds_per_token": (present.suffix_seconds + omitted.suffix_seconds)
            / max(present.suffix_tokens + omitted.suffix_tokens, 1),
            "transport_seconds_per_token_all_layers": transport_seconds_per_token,
        },
        "storage": {
            "checkpoint_bytes": present.boundaries["cut:0"].nbytes,
            "transition_log_bytes": log_bytes,
        },
        "admission": admission,
        "state": state_records,
        "truncated": truncated,
        "replay_check": replay,
        "behavior": behavior,
    }


# --------------------------------------------------------------------------
# configuration layout and CLI
# --------------------------------------------------------------------------


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def allocate_cases(cases: List[Dict[str, Any]], n_victims: int, max_prefix: int) -> Tuple[List, List, List]:
    victims = cases[:n_victims]
    prefix_pool = cases[n_victims : n_victims + max_prefix]
    suffix_pool = cases[n_victims + max_prefix :]
    if len(prefix_pool) < max_prefix or not suffix_pool:
        raise SystemExit(
            f"need {n_victims} victims + {max_prefix} prefix records + a suffix pool; have {len(cases)} cases"
        )
    return victims, prefix_pool, suffix_pool


def build_suffix(tokenizer: Any, suffix_pool: List[Dict[str, Any]], n_tokens: int) -> Tuple[mx.array, int]:
    """Concatenate suffix-pool records until at least ``n_tokens`` tokens; returns (tokens, records used)."""
    parts: List[mx.array] = []
    total = 0
    used = 0
    for rec in case_records(tokenizer, suffix_pool):
        parts.append(rec.tokens)
        total += rec.n_tokens
        used += 1
        if total >= n_tokens:
            break
    return mx.concatenate(parts, axis=1), used


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--report-model-label", default=None)
    ap.add_argument("--source", choices=CASE_SOURCES, default="synthetic")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--body-chars", type=int, default=None)
    ap.add_argument("--n-cases", type=int, default=192, help="how many cases to load from the source")
    ap.add_argument("--victims", default="0,1,2,3,4,5", help="case indices (within the loaded cases) used as victims")
    ap.add_argument("--prefix-depths", default="0,4,16,64", help="numbers of prefix records before the victim")
    ap.add_argument("--cuts", default=",".join(str(c) for c in DEFAULT_CUTS))
    ap.add_argument("--behavior-cuts", default=None, help="cuts at which to run behavioral probes (default: all cuts)")
    ap.add_argument("--behavior-trunc", choices=("none", "half", "all"), default="half")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--no-replay-check", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: run configurations i, i+n, i+2n, ...")
    ap.add_argument("--out-dir", default="outputs/kimi_sv/forcing_sweep_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--lazy-load", action="store_true")
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument(
        "--memory-limit-gb",
        type=float,
        default=80.0,
        help="hard MLX memory limit for this process; a runaway fails here instead of exhausting the machine",
    )
    args = ap.parse_args(argv)

    cuts = sorted(set(parse_int_list(args.cuts)) | {0})
    behavior_cuts = cuts if args.behavior_cuts is None else sorted(set(parse_int_list(args.behavior_cuts)) & set(cuts))
    victims = parse_int_list(args.victims)
    depths = sorted(set(parse_int_list(args.prefix_depths)))
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    cases, _stats = load_source_cases(
        args.source, args.n_cases, data_dir=args.data_dir, skip=args.skip, body_chars=args.body_chars
    )
    victim_cases, prefix_pool, suffix_pool = allocate_cases(cases, max(victims) + 1, max(depths))

    mx.clear_cache()
    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    if not args.tiny:
        mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    t0 = time.perf_counter()
    model, tokenizer, label = load_model(args.model, tiny=args.tiny, lazy=args.lazy_load)
    print(f"loaded {label} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    kda, _ = classify_layers(model)
    install_mla_mask(model)

    preamble = preamble_record(tokenizer, source_preamble(args.source))
    prefix_records_all = case_records(tokenizer, prefix_pool)
    suffix, suffix_records_used = build_suffix(tokenizer, suffix_pool, max(cuts))
    if int(suffix.shape[1]) < max(cuts):
        print(
            f"warning: suffix pool gives {int(suffix.shape[1])} tokens, below the largest cut {max(cuts)}; "
            "larger cuts are skipped",
            flush=True,
        )

    out_dir = Path(args.out_dir) / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [(v, p) for v in victims for p in depths]
    mine = [cfg for idx, cfg in enumerate(configs) if idx % shard_n == shard_i]
    print(f"{len(mine)} of {len(configs)} configurations in shard {args.shard}", flush=True)

    for v, p in mine:
        out = out_dir / f"victim{v:03d}-prefix{p:03d}.json"
        if out.exists() and not args.overwrite:
            print(f"exists; skipping {out}", flush=True)
            continue
        victim_case = victim_cases[v]
        victim_record = case_records(tokenizer, [victim_case])[0]
        prefix_records = prefix_records_all[:p]
        retained_case = prefix_pool[p - 1] if p > 0 else None
        t1 = time.perf_counter()
        result = evaluate_configuration(
            model,
            tokenizer,
            kda,
            preamble=preamble,
            prefix_records=prefix_records,
            victim_record=victim_record,
            victim_case=victim_case,
            retained_case=retained_case,
            suffix=suffix,
            cuts=cuts,
            behavior_cuts=behavior_cuts,
            behavior_trunc=args.behavior_trunc,
            greedy=(not args.no_greedy) and args.source in PUBLIC_SOURCES,
            replay_check=not args.no_replay_check,
        )
        report = {
            "schema": "kimi-sv-forcing-sweep-v1",
            "model": args.report_model_label or label,
            "host": {"platform": platform.platform(), "machine": platform.machine()},
            "source": args.source,
            "contains_source_text": False,
            "victim_index": v,
            "victim_key": victim_case["key"] if args.source in PUBLIC_SOURCES else None,
            "prefix_depth": p,
            "suffix_records_used": suffix_records_used,
            "kda_layers": list(kda),
            "protocol": {
                "cuts": cuts,
                "behavior_cuts": behavior_cuts,
                "behavior_trunc": args.behavior_trunc,
                "conditions": (
                    "present, omitted, mask (victim attention rows evicted), static (boundary receipt "
                    "subtracted), ledger (receipt scaled by cumulative decay), full (receipt transported "
                    "through record-present transitions), mask_full, trunc:k (exact first k suffix tokens "
                    "then transport), mask_trunc:k"
                ),
                "metric_direct": "||S_present - S_omitted||_F per layer",
                "metric_residual": "||S_corrected - S_omitted||_F / ||S_omitted||_F and / ||S_present - S_omitted||_F",
            },
            "seconds": time.perf_counter() - t1,
            **result,
        }
        payload = json.dumps(report, indent=1)
        if args.source not in PUBLIC_SOURCES:
            from mimic import assert_no_source_text

            assert_no_source_text(payload, cases)
        out.write_text(payload)
        del result
        mx.clear_cache()
        s = report["state"]
        last = str(max(int(k) for k in s))
        summary_bits = ""
        if "residual_full_rel_direct" in s[last]:
            ff = s[last]["residual_full_rel_direct"]
            imp = s[last]["imprint_fraction"]
            summary_bits = (
                f" forcing/direct at L={last}: {min(ff):.3f}-{max(ff):.3f};"
                f" imprint at L={last}: {min(imp):.2e}-{max(imp):.2e}"
            )
        print(
            f"victim {v} prefix {p}: {report['seconds'] / 60:.1f} min;"
            f" admission={report['admission'].get('qualified')};{summary_bits}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
