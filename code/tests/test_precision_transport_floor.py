"""Validate the independent transport arithmetic diagnostic on a live tiny KDA."""
import math

from kda_followup.transport_floor import diagnostic_record


def test_transport_floor_tracks_native_arithmetic_and_nonzero_context_effect():
    from kimi_sv.protocol import ByteTokenizer
    from kimi_sv.tiny import build_tiny

    model, _ = build_tiny(seed=2)
    record = {"record_id": "tiny-nonvacuity", "preamble": "abc", "prefix_records": [],
              "retained_records": [], "target": {"text": "Secret-D"},
              "suffix_records": [{"text": "A divergent retained suffix."}],
              "suffix_tokens": 4, "suffix_class": "short", "forget_instruction": "forget D", "neutral_instruction": "continue"}
    result = diagnostic_record(model, ByteTokenizer(), record, tiny=True)
    global_ = result["global"]
    assert result["status"] == "complete"
    assert global_["native_present_minus_omitted_frobenius"] > 0
    assert global_["native_transport_residual_frobenius"] > 0
    assert global_["max_empirical_floor"] >= global_["reconstruction_error_triangle_sum"]
    assert global_["max_empirical_floor"] >= global_["frozen_input_identity_error_frobenius"]
    assert math.isfinite(global_["residual_to_max_empirical_floor_ratio"])
    # Reconstruction follows the same recorded native branch and should be
    # close even though deleting the target changes the branch's actual state.
    assert global_["present_reconstruction_relative_error"] < 1e-5
    assert global_["omitted_reconstruction_relative_error"] < 1e-5


def test_wilson_intervals_respect_independent_group_size():
    from kda_followup.maintenance_report import wilson
    all_twelve = wilson(12, 12)
    all_six = wilson(6, 6)
    assert all_twelve["ci_low"] > all_six["ci_low"]
    assert wilson(0, 12)["ci_high"] > 0.2
