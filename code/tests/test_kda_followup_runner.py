"""Intervention isolation and full-state checks on native tiny Kimi kernels."""
import copy

import mlx.core as mx

from kda_followup.manifest import build_manifest
from kda_followup.run import compare_cache, evaluate_record, token_plan
from kimi_sv.mla_mask import install_mla_mask, set_mla_drop_positions
from kimi_sv.protocol import ByteTokenizer, encode
from kimi_sv.tiny import build_tiny


def test_all_conditions_preserve_replay_and_instruction_budget():
    model, _ = build_tiny(seed=7)
    install_mla_mask(model)
    record = build_manifest()["records"][0]
    result = evaluate_record(model, ByteTokenizer(), record, tiny=True, max_new_tokens=2)
    assert set(result["conditions"]) == set(build_manifest()["protocol"]["conditions"])
    assert result["state_checks"]["replay_vs_never"]["all_equal"]
    assert result["state_checks"]["repeat_never_vs_never"]["all_equal"]
    assert result["tokens"]["forget_instruction"] == result["tokens"]["neutral_instruction"]
    assert result["tokens"]["forget_instruction"] > 0
    conditions = result["conditions"]
    for key in ("target_probes", "retained_probes", "continuations"):
        for a, b in zip(conditions["replay"][key], conditions["never_stored"][key]):
            assert a["mean_log_probability"] == b["mean_log_probability"]
    assert all("greedy_exact" in p for p in conditions["present"]["retained_probes"])


def test_full_cache_check_detects_conv_and_mla_errors():
    model, _ = build_tiny(seed=2)
    a, b = model.make_cache(), model.make_cache()
    tokens = encode(ByteTokenizer(), "A short context.")
    mx.eval(model(tokens, cache=a), model(tokens, cache=b))
    assert compare_cache(model, a, b)["all_equal"]
    linear = next(i for i, l in enumerate(model.layers) if l.is_linear)
    b[linear].state[0] = b[linear].state[0] + .125
    assert not compare_cache(model, a, b)["all_equal"]
    b[linear].state = list(a[linear].state)
    global_idx = next(i for i, l in enumerate(model.layers) if not l.is_linear)
    b[global_idx].keys = b[global_idx].keys + .125
    assert not compare_cache(model, a, b)["all_equal"]


def test_neutral_padding_and_deleted_target_exclusion():
    record = copy.deepcopy(build_manifest()["records"][0])
    record["neutral_instruction"] = "OK"
    *_, forget, neutral = token_plan(ByteTokenizer(), record, tiny=True)
    assert forget.shape == neutral.shape
    for probe in record["continuations"]:
        assert record["target"]["secret"].strip() not in probe["target"] + probe["prompt"]
