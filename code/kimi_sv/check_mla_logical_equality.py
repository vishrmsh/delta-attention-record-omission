"""Compare the logical MLA cache contents of replay and the never-stored run.

The replay certificate declares final logits and the 80 KDA arrays. The MLA
caches are allocated in 256-token blocks, so their raw buffers carry unused
capacity and cannot be compared byte for byte. This check compares what the
buffers logically hold: for every MLA layer, the keys and values up to the
current offset, after (a) restoring the pre-record checkpoint of the
record-present run and replaying the suffix and (b) running the never-stored
context directly. It reports the maximum absolute difference per layer.

Usage:
    python -m kimi_sv.check_mla_logical_equality --source synthetic --victims 0,3 \
        --prefix-depths 0,16 --cuts 0,4096 --out gemma_sv/benchmarks/kda_mla_logical_equality_v1.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

from kimi_sv.eval_forcing_sweep import allocate_cases, build_suffix, chunk_bounds, parse_int_list, run_context
from kimi_sv.eval_masked_sampling import NullTap
from kimi_sv.protocol import PUBLIC_SOURCES, case_records, load_model, load_source_cases, preamble_record, source_preamble
from kimi_sv.state import classify_layers, restore


def mla_logical(cache: List[Any], mla: List[int]) -> Dict[int, Dict[str, mx.array]]:
    out = {}
    for i in mla:
        c = cache[i]
        n = int(c.offset)
        out[i] = {"keys": c.keys[..., :n, :], "values": c.values[..., :n, :], "offset": n}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit")
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-cases", type=int, default=192)
    ap.add_argument("--victims", default="0,3")
    ap.add_argument("--prefix-depths", default="0,16")
    ap.add_argument("--cuts", default="0,4096")
    ap.add_argument("--out", default="gemma_sv/benchmarks/kda_mla_logical_equality_v1.json")
    ap.add_argument("--memory-limit-gb", type=float, default=75.0)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)

    cuts = sorted(set(parse_int_list(args.cuts)) | {0})
    victims = parse_int_list(args.victims)
    depths = sorted(set(parse_int_list(args.prefix_depths)))
    cases, _ = load_source_cases(args.source, args.n_cases, data_dir=args.data_dir)
    victim_cases, prefix_pool, suffix_pool = allocate_cases(cases, max(victims) + 1, max(depths))
    mx.clear_cache()
    if not args.tiny:
        mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    model, tokenizer, label = load_model(args.model, tiny=args.tiny)
    kda, mla = classify_layers(model)
    preamble = preamble_record(tokenizer, source_preamble(args.source))
    prefix_all = case_records(tokenizer, prefix_pool)
    suffix, _ = build_suffix(tokenizer, suffix_pool, max(cuts))
    tap = NullTap()

    results = []
    for v in victims:
        for p in depths:
            t0 = time.perf_counter()
            victim_record = case_records(tokenizer, [victim_cases[v]])[0]
            before_p = [preamble.tokens, *[r.tokens for r in prefix_all[:p]], victim_record.tokens]
            before_o = [preamble.tokens, *[r.tokens for r in prefix_all[:p]]]
            present = run_context(model, kda, before_p, suffix, cuts, tap, mark_after=len(before_p) - 2)
            omitted = run_context(model, kda, before_o, suffix, cuts, tap)
            # Replay: restore the pre-record checkpoint of the record-present run and re-ingest the suffix.
            restore(model, present.cache, present.boundaries["prefix"])
            for a, b in chunk_bounds(cuts):
                if b > int(suffix.shape[1]):
                    break
                mx.eval(model(suffix[:, a:b], cache=present.cache))
            rep = mla_logical(present.cache, mla)
            ref = mla_logical(omitted.cache, mla)
            per_layer = {}
            for i in mla:
                if rep[i]["offset"] != ref[i]["offset"]:
                    per_layer[i] = {"offset_replay": rep[i]["offset"], "offset_never_stored": ref[i]["offset"], "equal": False}
                    continue
                dk = float(mx.max(mx.abs(rep[i]["keys"].astype(mx.float32) - ref[i]["keys"].astype(mx.float32))).item())
                dv = float(mx.max(mx.abs(rep[i]["values"].astype(mx.float32) - ref[i]["values"].astype(mx.float32))).item())
                per_layer[i] = {"offset": rep[i]["offset"], "max_abs_diff_keys": dk, "max_abs_diff_values": dv, "equal": dk == 0.0 and dv == 0.0}
            results.append(
                {
                    "victim_index": v,
                    "prefix_depth": p,
                    "suffix_tokens": int(present.suffix_tokens),
                    "mla_layers": per_layer,
                    "all_equal": all(x["equal"] for x in per_layer.values()),
                    "seconds": time.perf_counter() - t0,
                }
            )
            print(
                f"victim {v} prefix {p}: {results[-1]['seconds']:.0f}s; MLA logical contents equal in "
                f"{sum(x['equal'] for x in per_layer.values())}/{len(per_layer)} layers; offset {next(iter(per_layer.values())).get('offset')}",
                flush=True,
            )
            del present, omitted, rep, ref
            mx.clear_cache()
    report = {
        "schema": "kda-mla-logical-equality-v1",
        "model": label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "source": args.source,
        "contains_source_text": False,
        "protocol": {
            "cuts": cuts,
            "comparison": "keys[..., :offset, :] and values[..., :offset, :] of every MLA layer, replay vs never-stored run, float32 max abs diff",
        },
        "configurations": results,
        "all_equal": all(r["all_equal"] for r in results),
        "n_configurations": len(results),
        "n_mla_layers": len(mla),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else None
    if existing and existing.get("schema") == report["schema"] and existing.get("source") != args.source:
        report = {"schema": report["schema"], "model": label, "contains_source_text": False, "sources": {existing["source"]: existing, args.source: report}}
        report["all_equal"] = all(s["all_equal"] for s in report["sources"].values())
        report["n_configurations"] = sum(s["n_configurations"] for s in report["sources"].values())
        report["n_mla_layers"] = len(mla)
    elif existing and "sources" in existing:
        existing["sources"][args.source] = report
        existing["all_equal"] = all(s["all_equal"] for s in existing["sources"].values())
        existing["n_configurations"] = sum(s["n_configurations"] for s in existing["sources"].values())
        report = existing
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}; all_equal={report['all_equal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
