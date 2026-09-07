"""SV decrement versus replay deletion: the approximate mechanism against its oracle.

The two papers' deletion mechanisms meet on one model. Replay deletion restores
a pre-victim checkpoint and re-ingests the surviving suffix, producing a state
bitwise identical to never having seen the record -- exact, at O(suffix) cost.
SV decrement zeroes the victim's positions in every grafted gate: the tokens
leave the SVDD fit and carry exactly zero readout weight in the global MLA
layers, at O(1) cost with no state mutation at all.

On a hybrid, the decrement residual has an architectural meaning. All seven
global layers are grafted, so after decrement the victim reaches the readout
only through the twenty untouched KDA layers. The residual lift therefore
measures what the recurrent state remembers that a readout gate cannot delete
-- the quantity that motivates replay for the hybrid's linear path, and an
SV-native linear layer as the third-paper design.

Per victim, four readouts of every record: resident (drops off), never-ingested
reference, under decrement, and after replay. The victim's lift is measured
against the reference in each condition; replay's must be exactly zero and is
checked bitwise on logits and KDA state. Retained-record integrity under
decrement is measured against the resident readout (same context, so the shift
is pure collateral from the refit); after replay it is exact by construction.

The gate runs the mass-preserving readout, so a decrement redistributes the
victim's share of prefix influence over the surviving prefix rather than
handing it to the local window.

Runs on the synthetic case file by default. With ``--source cds`` or
``--source notes`` the same protocol runs on credentialed local MIMIC records
under the ``mimic`` package's disclosure discipline: no source text, value, or
generation is printed or written (greedy readbacks are skipped entirely), and
a substring audit runs before the report is written. When cases carry
field-level probes, each condition is also scored per field from its own stem,
isolating the distinctive values from shared record scaffolding.

``--mechanism`` selects how the victim's attention rows are removed. ``gate``
(the original control) grafts the SV gate onto the MLA layers and drops the
victim's positions from it, so every condition, the present and never-ingested
references included, runs through the gated readout. ``native`` leaves the
released layers in place and evicts the same rows in the fused kernel by
giving them the dtype-minimum score (``mla_mask``), which is the masking the
suffix-length sweep uses; with no positions dropped it reproduces the released
layer bit for bit. ``--sources`` runs several corpora on one loaded model.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from .eval_mimic_deletion import field_stats
from .graft import gate_stats, graft_sv_into_kimi, set_drop_positions
from .mla_mask import install_mla_mask, set_mla_drop_positions
from .probes import lift_nats
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
    score_all,
    source_preamble,
)
from .records import RecordMemory, max_abs_diff, state_max_abs_diff


def build(
    model: Any,
    tokenizer: Any,
    cases: List[Dict[str, str]],
    skip_key: Optional[str] = None,
    preamble: Optional[str] = None,
) -> Tuple[RecordMemory, Dict[str, Tuple[int, int]]]:
    records = [preamble_record(tokenizer, preamble)] + [
        r for r in case_records(tokenizer, cases) if r.key != skip_key
    ]
    memory = RecordMemory(model)
    spans: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for r in records:
        spans[r.key] = (offset, offset + r.n_tokens)
        memory.ingest(r)
        offset += r.n_tokens
    return memory, spans


def run_source(
    model: Any,
    tokenizer: Any,
    label: str,
    source: str,
    args: argparse.Namespace,
    *,
    native: bool,
) -> Dict[str, Any]:
    """Run the oracle protocol for one corpus on an already loaded model."""
    public = source in PUBLIC_SOURCES
    preamble = source_preamble(source)
    cases, source_stats = load_source_cases(
        source,
        args.records,
        data_dir=args.data_dir,
        skip=args.skip,
        body_chars=args.body_chars,
    )
    victims = [v for v in args.victims if v < len(cases)]
    if len(victims) != len(args.victims):
        print(f"{source}: {len(cases)} cases; victims {victims} of {args.victims} are in range", flush=True)

    drop = set_mla_drop_positions if native else set_drop_positions
    name = "mask" if native else "decrement"
    after = f"after_{name}"
    audit_probe = encode(tokenizer, "\nAudit complete.")

    victims_out: Dict[str, Any] = {}
    for v in victims:
        victim = cases[v]
        vkey = victim["key"]
        drop(model, None)

        present, spans = build(model, tokenizer, cases, preamble=preamble)
        reference, _ = build(model, tokenizer, cases, skip_key=vkey, preamble=preamble)

        has_fields = bool(victim.get("fields"))
        present_stats = score_all(present, tokenizer, cases)
        reference_stats = score_all(reference, tokenizer, cases)
        present_fields = field_stats(present, tokenizer, victim) if has_fields else []
        never_fields = field_stats(reference, tokenizer, victim) if has_fields else []
        lift_present = (
            present_stats[vkey].mean_log_probability
            - reference_stats[vkey].mean_log_probability
        )

        def field_lift(condition_fields) -> Optional[float]:
            if not has_fields:
                return None
            return statistics.mean(
                lift_nats(c, n) for c, n in zip(condition_fields, never_fields)
            )

        # -- row removal: the victim's attention rows leave every MLA layer ---
        start, stop = spans[vkey]
        t0 = time.perf_counter()
        drop(model, list(range(start, stop)))
        removal_seconds = time.perf_counter() - t0
        removed_stats = score_all(present, tokenizer, cases)
        removed_fields = field_stats(present, tokenizer, victim) if has_fields else []
        removed_answer = (
            answer_text(
                present, tokenizer, victim["question"], max_tokens=args.greedy_tokens
            )
            if public
            else None
        )
        lift_removed = (
            removed_stats[vkey].mean_log_probability
            - reference_stats[vkey].mean_log_probability
        )
        retained_keys = [c["key"] for c in cases if c["key"] != vkey]
        removed_collateral = [
            abs(
                removed_stats[k].mean_log_probability
                - present_stats[k].mean_log_probability
            )
            for k in retained_keys
        ]
        drop(model, None)

        # -- replay: the oracle ----------------------------------------------
        deletion = present.delete(vkey)
        replay_stats = score_all(present, tokenizer, cases)
        replay_fields = field_stats(present, tokenizer, victim) if has_fields else []
        replay_answer = (
            answer_text(
                present, tokenizer, victim["question"], max_tokens=args.greedy_tokens
            )
            if public
            else None
        )
        never_answer = (
            answer_text(
                reference, tokenizer, victim["question"], max_tokens=args.greedy_tokens
            )
            if public
            else None
        )
        lift_replay = (
            replay_stats[vkey].mean_log_probability
            - reference_stats[vkey].mean_log_probability
        )
        logit_delta = max_abs_diff(
            present.probe(audit_probe), reference.probe(audit_probe)
        )
        state_delta = state_max_abs_diff(present, reference)

        removal = 1.0 - lift_removed / lift_present if lift_present > 0 else None
        f_present = field_lift(present_fields)
        f_removed = field_lift(removed_fields)
        f_replay = field_lift(replay_fields)
        f_removal = (
            1.0 - f_removed / f_present
            if f_present is not None and f_present > 0
            else None
        )
        victims_out[vkey] = {
            "victim_index": v,
            "span": [start, stop],
            "n_tokens": stop - start,
            "lift_nats": {
                "resident": lift_present,
                after: lift_removed,
                "after_replay": lift_replay,
            },
            f"{name}_removal_fraction": removal,
            "field_lift_nats": (
                {
                    "resident": f_present,
                    after: f_removed,
                    "after_replay": f_replay,
                }
                if has_fields
                else None
            ),
            "field_removal_fraction": f_removal,
            "victim_rank": {
                "resident": present_stats[vkey].first_token_rank,
                after: removed_stats[vkey].first_token_rank,
                "after_replay": replay_stats[vkey].first_token_rank,
                "never_ingested": reference_stats[vkey].first_token_rank,
            },
            "greedy_answers": (
                {
                    after: removed_answer,
                    "after_replay": replay_answer,
                    "never_ingested": never_answer,
                }
                if public
                else None
            ),
            "retained_collateral_nats": {
                f"{name}_vs_resident_mean": statistics.mean(removed_collateral),
                f"{name}_vs_resident_max": max(removed_collateral),
            },
            "cost": {
                f"{name}_seconds": removal_seconds,
                f"{name}_tokens_touched": 0,
                "replay_seconds": deletion.seconds,
                "replay_tokens": deletion.replayed_tokens,
                "checkpoint_bytes": deletion.checkpoint_bytes,
            },
            "replay_equivalence": {
                "max_abs_logit_delta": logit_delta,
                "max_abs_kda_state_delta": state_delta,
            },
        }
        out = victims_out[vkey]
        field_note = (
            f", field lift {f_present:+.3f} -> {f_removed:+.3f}" if has_fields else ""
        )
        print(
            f"{source} victim idx {v}: lift {lift_present:+.3f} -> "
            f"{name} {lift_removed:+.3f} "
            f"({(removal or 0)*100:.0f}% removed, rank "
            f"{out['victim_rank'][after]}){field_note}, "
            f"replay {lift_replay:+.3f} (residual {logit_delta:.3e}); "
            f"{name} {removal_seconds*1000:.1f} ms vs replay "
            f"{deletion.seconds:.1f} s / {deletion.replayed_tokens} tokens",
            flush=True,
        )

    residuals = [o["lift_nats"][after] for o in victims_out.values()]
    removals = [
        o[f"{name}_removal_fraction"]
        for o in victims_out.values()
        if o[f"{name}_removal_fraction"] is not None
    ]
    field_removals = [
        o["field_removal_fraction"]
        for o in victims_out.values()
        if o["field_removal_fraction"] is not None
    ]
    report = {
        "model": label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "source": source,
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "generations_produced": public,
        "source_rows": source_stats,
        "protocol": {
            "records": len(cases),
            "skip": args.skip,
            "row_removal": "native-mla-mask" if native else "sv-gate-decrement",
            "chunk": None if native else args.chunk,
            "nu": None if native else args.nu,
            "preserve_prefix_mass": None if native else True,
            "victims": victims,
            "greedy_tokens": args.greedy_tokens if public else None,
            "memory_limit_gb": args.memory_limit_gb,
        },
        "gate_stats": None if native else gate_stats(model),
        "aggregate": {
            f"{name}_residual_lift_mean": statistics.mean(residuals),
            f"{name}_residual_lift_max": max(residuals),
            f"{name}_removal_fraction_mean": (
                statistics.mean(removals) if removals else None
            ),
            "field_removal_fraction_mean": (
                statistics.mean(field_removals) if field_removals else None
            ),
            "replay_exact_all": all(
                o["replay_equivalence"]["max_abs_logit_delta"] == 0.0
                and o["replay_equivalence"]["max_abs_kda_state_delta"] == 0.0
                for o in victims_out.values()
            ),
        },
        "victims": victims_out,
    }
    payload = json.dumps(report, indent=2)
    if not public:
        from mimic import assert_no_source_text

        assert_no_source_text(payload, cases)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", choices=CASE_SOURCES, default="synthetic")
    ap.add_argument(
        "--sources",
        choices=CASE_SOURCES,
        nargs="+",
        default=None,
        help="run several corpora on one loaded model; overrides --source",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--body-chars", type=int, default=None)
    ap.add_argument("--records", type=int, default=16)
    ap.add_argument(
        "--victims",
        type=int,
        nargs="+",
        default=[1, 8, 14],
        help="early, middle, and late positions by default, for the cost curve",
    )
    ap.add_argument("--mechanism", choices=("gate", "native"), default="gate")
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--greedy-tokens", type=int, default=12)
    ap.add_argument("--memory-limit-gb", type=float, default=None)
    ap.add_argument("--out", default="outputs/kimi_sv/oracle_decrement_vs_replay.json")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="with --sources: write oracle_<mechanism>_<source>.json here",
    )
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)

    native = args.mechanism == "native"
    sources = args.sources or [args.source]
    if len(sources) > 1 and not args.out_dir:
        ap.error("--sources needs --out-dir")

    if args.memory_limit_gb is not None and not args.tiny:
        mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    t0 = time.perf_counter()
    model, tokenizer, label = load_model(args.model, tiny=args.tiny)
    print(f"loaded {label} in {(time.perf_counter() - t0)/60:.1f} min", flush=True)

    if native:
        wrapped = install_mla_mask(model)
        print(f"native MLA masking installed on layers {wrapped}", flush=True)
    else:
        grafted = graft_sv_into_kimi(
            model,
            mode="latent",
            nu=args.nu,
            chunk=args.chunk,
            preserve_prefix_mass=True,
            collect_stats=True,
        )
        print(f"grafted SV gate (mass-preserving) onto layers {grafted}", flush=True)

    for source in sources:
        report = run_source(model, tokenizer, label, source, args, native=native)
        if args.out_dir:
            out_path = Path(args.out_dir) / f"oracle_{args.mechanism}_{source}.json"
        else:
            out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"report written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
