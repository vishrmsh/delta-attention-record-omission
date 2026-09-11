import json
from pathlib import Path
import struct

import pytest

from kda_followup.precision import inspect_snapshot, official_metadata, require_precision, freeze_subset
from kda_followup.amendments import make_manifest


def snapshot(tmp_path, dtype, quant=None):
    config = {"model_type": "kimi_linear", "dtype": "bfloat16"}
    if quant:
        config["quantization"] = quant
    (tmp_path / "config.json").write_text(json.dumps(config))
    hdr = json.dumps({"model.embed_tokens.weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 4]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 4)
    return tmp_path


def test_bf16_config_does_not_disguise_packed_quantization(tmp_path):
    audit = inspect_snapshot(snapshot(tmp_path, "U32", {"bits": 8}))
    assert not audit["bf16_tensor_format"]
    require_precision(audit, "8bit")
    with pytest.raises(ValueError, match="original hash-verified"):
        require_precision(audit, "bf16")


def test_float_tensor_headers_do_not_prove_unquantized_origin(tmp_path):
    audit = inspect_snapshot(snapshot(tmp_path, "BF16"))
    assert audit["bf16_tensor_format"]
    with pytest.raises(ValueError, match="original hash-verified"):
        require_precision(audit, "bf16")
    with pytest.raises(ValueError, match="shard inventory"):
        inspect_snapshot(tmp_path, verify_official=True)


def test_missing_shards_rejected(tmp_path):
    snapshot(tmp_path, "BF16")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="missing"):
        inspect_snapshot(tmp_path)


def test_pinned_metadata_inventory_is_exact():
    assert official_metadata()["sha"] == "e1df551a447157d4658b573f9a695d57658590e9"


def test_amendment_manifest_unique_and_prefix_stable():
    manifest = make_manifest()
    assert len(manifest["scenarios"]) == 12
    assert make_manifest(2)["scenarios"] == manifest["scenarios"][:2]
    codes = []
    keys = []
    for scenario in manifest["scenarios"]:
        assert scenario["old"]["key"] == scenario["new"]["key"]
        assert scenario["old"]["secret"] != scenario["new"]["secret"]
        records = [scenario["old"], *scenario["retained"], scenario["admission"]]
        keys.extend(r["key"] for r in records)
        codes.extend(r["secret"] for r in records)
        codes.append(scenario["new"]["secret"])
        for retained in scenario["retained"]:
            assert scenario["old"]["secret"].strip() not in retained["text"]
            assert scenario["new"]["secret"].strip() not in retained["text"]
    assert len(keys) == len(set(keys))
    assert len(codes) == len(set(codes))


def test_precision_subset_preselected_across_suffix_strata(tmp_path):
    from kda_followup.manifest import write_manifest
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest)
    subset = freeze_subset(manifest, tmp_path / "subset.json")
    assert subset["suffix_strata"] == {"short": 4, "long": 4}
    assert len(subset["record_ids"]) == 8
    assert subset["record_ids"][-1] == "kda-fu-evaluation-009"
    assert freeze_subset(manifest, tmp_path / "subset.json") == subset


def test_complete_cache_detects_conv_and_logical_attention_corruption():
    import mlx.core as mx
    from kimi_sv.protocol import build_memory
    from kimi_sv.records import Record
    from kimi_sv.state import classify_layers
    from kimi_sv.tiny import build_tiny, token_block
    from kda_followup.amendments import compare_cache
    model, _ = build_tiny(seed=1)
    records = [Record(key="a", tokens=token_block(12))]
    left, right = build_memory(model, records), build_memory(model, records)
    assert compare_cache(model, left, right)["all_equal"]
    kda, mla = classify_layers(model)
    saved = right.cache[kda[0]].state[0]
    right.cache[kda[0]].state[0] = saved + 1
    assert not compare_cache(model, left, right)["all_equal"]
    right.cache[kda[0]].state[0] = saved
    # Unused allocated MLA capacity is deliberately outside the comparison.
    c = right.cache[mla[0]]
    offset = int(c.offset)
    original = c.keys
    c.keys = mx.concatenate([c.keys[..., :offset, :], c.keys[..., offset:, :] + 1], axis=-2)
    assert compare_cache(model, left, right)["all_equal"]
    c.keys = original + 1
    assert not compare_cache(model, left, right)["all_equal"]
