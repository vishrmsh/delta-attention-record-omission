"""Sampling attack against masked and receipt-corrected Kimi states.

The suffix sweep scores each corrected state by teacher forcing and greedy
decoding. This driver asks the complementary question a reviewer will ask: if
an attacker simply samples answers from the corrected state, how often does the
deleted record's secret come out? For every admitted configuration of the sweep
it rebuilds the record-present and record-omitted contexts, and at each chosen
cut it draws ``--samples`` continuations of the victim's question at the
attack-suite settings (temperature 1.0, top-p 1.0) under five conditions:

* present        -- the record was ingested and nothing was done (ceiling);
* omitted        -- the record was never ingested (floor);
* mask           -- the record's attention rows are evicted from every MLA
                    layer; the recurrent and convolution state is untouched;
* static         -- the boundary receipt is subtracted from the recurrent state;
* mask_static    -- both.

Reports hold hit counts and hit positions only; no sampled text is stored.

Usage (one shard of the synthetic configurations):
    python -m kimi_sv.eval_masked_sampling --source synthetic --victims 0,1,2,3,4,5 \
        --prefix-depths 0,4,16,64 --cuts 0,4096 --samples 100 --shard 0/2
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx

from kimi_sv.eval_forcing_sweep import allocate_cases, build_suffix, parse_int_list, run_context, with_recurrent
from kimi_sv.mla_mask import install_mla_mask, set_mla_drop_positions
from kimi_sv.protocol import (
    PUBLIC_SOURCES,
    case_records,
    encode,
    load_model,
    load_source_cases,
    preamble_record,
    source_preamble,
    stop_ids,
)
from kimi_sv.records import RecordMemory
from kimi_sv.state import Boundary, classify_layers, restore, snapshot


class NullTap:
    """Stands in for the kernel tap when no transition inputs are needed."""

    active = False

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def stacked(self) -> Dict[str, mx.array]:
        return {}


_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.casefold()).split())


def is_hit(sample: str, secret: str) -> bool:
    s = normalize(secret)
    return bool(s) and s in normalize(sample)


def sample_answer(
    memory: RecordMemory,
    prompt: mx.array,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: set,
    key: mx.array,
) -> List[int]:
    """Draw one continuation without disturbing the memory."""
    mark_model, mark_cache = memory.model, memory.cache
    mark = snapshot(mark_model, mark_cache, label="sample")
    try:
        out: List[int] = []
        logits = memory._forward(prompt)
        for _ in range(max_tokens):
            logp = logits[0].astype(mx.float32) / temperature
            if top_p < 1.0:
                probs = mx.softmax(logp, axis=-1)
                order = mx.argsort(-probs)
                sorted_probs = probs[order]
                cum = mx.cumsum(sorted_probs)
                keep = cum - sorted_probs < top_p
                masked = mx.where(keep, sorted_probs, mx.zeros_like(sorted_probs))
                logp_sorted = mx.log(masked / mx.sum(masked))
                key, sub = mx.random.split(key)
                pick = int(mx.random.categorical(logp_sorted, key=sub).item())
                tok = int(order[pick].item())
            else:
                key, sub = mx.random.split(key)
                tok = int(mx.random.categorical(logp, key=sub).item())
            if tok in stop:
                break
            out.append(tok)
            logits = memory._forward(mx.array([[tok]]))
        return out
    finally:
        restore(mark_model, mark_cache, mark)


def sample_condition(
    model: Any,
    cache: List[Any],
    boundary: Boundary,
    drop_pos: Optional[List[int]],
    tokenizer: Any,
    victim: Dict[str, Any],
    *,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> Dict[str, Any]:
    restore(model, cache, boundary)
    set_mla_drop_positions(model, drop_pos)
    try:
        memory = RecordMemory(model, cache=cache)
        prompt = encode(tokenizer, "\n" + victim["question"] + victim.get("body_stem", ""))
        stop = set(stop_ids(tokenizer))
        hits: List[bool] = []
        lengths: List[int] = []
        t0 = time.perf_counter()
        for i in range(n):
            toks = sample_answer(
                memory, prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop,
                key=mx.random.key(seed * 100_003 + i),
            )
            text = tokenizer.decode(toks)
            hits.append(is_hit(text, victim["secret"]))
            lengths.append(len(toks))
        seconds = time.perf_counter() - t0
    finally:
        set_mla_drop_positions(model, None)
    first = next((i for i, h in enumerate(hits) if h), None)
    return {
        "n_samples": n,
        "hits": int(sum(hits)),
        "hit_rate": sum(hits) / n,
        "any_hit": bool(any(hits)),
        "first_hit_index": first,
        "mean_tokens": sum(lengths) / n,
        "seconds": seconds,
    }


def evaluate_configuration(
    model: Any,
    tokenizer: Any,
    kda: Sequence[int],
    *,
    preamble: Any,
    prefix_records: Sequence[Any],
    victim_record: Any,
    victim_case: Dict[str, Any],
    suffix: mx.array,
    cuts: Sequence[int],
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> Dict[str, Any]:
    tap = NullTap()
    before_present = [preamble.tokens, *[r.tokens for r in prefix_records], victim_record.tokens]
    before_omitted = [preamble.tokens, *[r.tokens for r in prefix_records]]
    present = run_context(model, kda, before_present, suffix, cuts, tap, mark_after=len(before_present) - 2)
    omitted = run_context(model, kda, before_omitted, suffix, cuts, tap)
    n_prefix_tokens = int(present.boundaries["prefix"].token_offset)
    victim_positions = list(range(n_prefix_tokens, n_prefix_tokens + victim_record.n_tokens))
    receipt = present.states["cut:0"] - omitted.states["cut:0"]
    mx.eval(receipt)

    available = sorted(int(k.split(":")[1]) for k in present.states if k.startswith("cut:"))
    out: Dict[str, Any] = {}
    # Cuts run longest first and each condition restores onto the cache of its
    # own run, so a restore never has to roll a cache forward.
    for c in sorted(set(cuts) & set(available), reverse=True):
        bp, bo = present.boundaries[f"cut:{c}"], omitted.boundaries[f"cut:{c}"]
        static_state = present.states[f"cut:{c}"] - receipt
        mx.eval(static_state)
        b_static = with_recurrent(bp, kda, static_state)
        conditions = {
            "present": (present.cache, bp, None),
            "omitted": (omitted.cache, bo, None),
            "mask": (present.cache, bp, victim_positions),
            "static": (present.cache, b_static, None),
            "mask_static": (present.cache, b_static, victim_positions),
        }
        out[str(c)] = {}
        for name, (cache, boundary, drop) in conditions.items():
            out[str(c)][name] = sample_condition(
                model, cache, boundary, drop, tokenizer, victim_case,
                n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed,
            )
        del b_static, static_state
        mx.clear_cache()
    return {"sampling": out, "victim_positions": [victim_positions[0], victim_positions[-1]], "cuts_available": available}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit")
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-cases", type=int, default=192)
    ap.add_argument("--victims", default="0,1,2,3,4,5")
    ap.add_argument("--prefix-depths", default="0,4,16,64")
    ap.add_argument("--cuts", default="0,4096")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out-dir", default="outputs/kimi_sv/masked_sampling_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--lazy-load", action="store_true")
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument("--memory-limit-gb", type=float, default=75.0)
    args = ap.parse_args(argv)

    cuts = sorted(set(parse_int_list(args.cuts)) | {0})
    victims = parse_int_list(args.victims)
    depths = sorted(set(parse_int_list(args.prefix_depths)))
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    cases, _ = load_source_cases(args.source, args.n_cases, data_dir=args.data_dir)
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
        t1 = time.perf_counter()
        result = evaluate_configuration(
            model, tokenizer, kda,
            preamble=preamble, prefix_records=prefix_records_all[:p], victim_record=victim_record,
            victim_case=victim_case, suffix=suffix, cuts=cuts,
            n=args.samples, max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            seed=args.seed,
        )
        report = {
            "schema": "kimi-sv-masked-sampling-v1",
            "model": label,
            "host": {"platform": platform.platform(), "machine": platform.machine()},
            "source": args.source,
            "contains_source_text": False,
            "victim_index": v,
            "victim_key": victim_case["key"] if args.source in PUBLIC_SOURCES else None,
            "prefix_depth": p,
            "suffix_records_used": suffix_records_used,
            "protocol": {
                "cuts": cuts,
                "samples_per_condition": args.samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "hit_rule": "casefolded, punctuation-stripped, whitespace-collapsed substring match of the secret",
                "conditions": "present, omitted, mask (victim attention rows evicted), static (boundary receipt subtracted), mask_static",
            },
            "seconds": time.perf_counter() - t1,
            **result,
        }
        payload = json.dumps(report, indent=1)
        if args.source not in PUBLIC_SOURCES:
            from mimic import assert_no_source_text

            assert_no_source_text(payload, cases)
        out.write_text(payload)
        mx.clear_cache()
        bits = "; ".join(
            f"L={c}: " + " ".join(f"{k}={r['hits']}/{r['n_samples']}" for k, r in conds.items())
            for c, conds in report["sampling"].items()
        )
        print(f"victim {v} prefix {p}: {report['seconds'] / 60:.1f} min; {bits}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
