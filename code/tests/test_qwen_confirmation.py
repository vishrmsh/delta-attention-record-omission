import copy
import torch

from gemma_sv import qwen_replay_audit as audit
from recurrent_families.qwen_confirmation import receipt_analysis, summarize, suffix_case, wilson


def test_receipt_controls_bind_to_model_states_and_changed_inputs():
    torch.manual_seed(391)
    initial_p = torch.randn(1, 2, 4, 3)
    initial_o = torch.randn(1, 2, 4, 3)
    common = {"k":torch.nn.functional.normalize(torch.randn(1,9,2,4),dim=-1),"v":torch.randn(1,9,2,3),"g":-torch.rand(1,9,2)*.2,"beta":torch.rand(1,9,2)}
    p = {**common,"initial":initial_p}
    o = {**copy.deepcopy(common),"initial":initial_o}
    o["v"] += .15
    p["final"] = audit._run_recurrence(initial_p,p)
    o["final"] = audit._run_recurrence(initial_o,o)
    result = receipt_analysis(initial_o,initial_p,p,o)
    assert result["frozen_transport_state_scaled_error"] < 1e-6
    assert result["forcing_decomposition_relative_error"] < 1e-6
    assert result["live_recurrence_state_scaled_error"] == 0
    assert result["native_transport_state_scaled_error"] > .01
    assert result["native_mismatch_to_numerical_floor"] > 5
    corrupted = copy.deepcopy(p)
    corrupted["final"] += .1
    assert receipt_analysis(initial_o,initial_p,corrupted,o)["live_recurrence_state_scaled_error"] > .01


def test_whole_record_failures_and_suffixes_remain_paired():
    rows = [{"record_id":"a","status":"completed","cases":[{"suffix":s,"checks":{"exact":True},"receipt_analysis":{"0":{k:1. for k in ("native_transport_relative_residual","native_transport_state_scaled_error","frozen_transport_state_scaled_error","frozen_transport_relative_residual","frozen_static_state_scaled_error","frozen_ledger_state_scaled_error","live_recurrence_state_scaled_error","forcing_decomposition_relative_error","native_mismatch_to_numerical_floor")}}} for s in ("short","long")]},{"record_id":"b","status":"failed","cases":[]}]
    manifest = {"records":[{"record_id":"a"},{"record_id":"b"}],"analysis":{"bootstrap_seed":7,"bootstrap_replicates":100}}
    result = summarize(rows,manifest)
    assert result["whole_record_checks"]["exact"]["mean"] == .5
    assert result["whole_record_checks"]["exact"]["denominator_including_failures"] == 2
    assert result["paired_long_minus_short"]["native_transport_relative_residual"]["mean"] == 0
    assert result["failed_records"] == ["b"]
    assert wilson(32,32)[0] < 1


def test_complete_cache_replay_and_long_suffix_capture():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(981)
    config = Qwen3_5TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=1,linear_num_value_heads=2,layer_types=["linear_attention"]*3+["full_attention"],rope_parameters={"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":.25,"mrope_section":[1,1,0],"mrope_interleaved":True},mtp_num_hidden_layers=0)
    model = Qwen3_5ForCausalLM(config).eval()
    record = {"token_ids":{"prefix":list(range(10,26)),"victim":list(range(30,38)),"long":list(range(40,120))}}
    with torch.inference_mode():
        result = suffix_case(model,record,"long",[0,2],"cpu")
    assert result["replay_vs_fresh_omission"]["array_count"] == 9
    assert result["replay_vs_fresh_omission"]["flag_count"] == 12
    for label in ("replay","repeat"):
        assert result[f"{label}_vs_fresh_omission"]["all_arrays_exact"]
        assert result[f"{label}_vs_fresh_omission"]["all_flags_equal"]
    assert all(r["frozen_transport_state_scaled_error"] < 5e-3 for r in result["receipt_analysis"].values())
