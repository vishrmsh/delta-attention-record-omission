"""Standard-library verification of the completed anonymized follow-up evidence."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read(relative):
    return json.loads((ROOT/relative).read_text())

def provenance():
    files=read('followup/PROVENANCE.json')['files']
    entries={item['released_path']:item for item in files}
    assert len(entries)==len(files), 'duplicate follow-up provenance path'
    return entries

def original_hash(relative, canonical_hash=False):
    return provenance()[relative]['original_canonical_sha256' if canonical_hash else 'original_sha256']

def verify():
    entries=provenance()
    scientific={path.relative_to(ROOT).as_posix()
                for path in (ROOT/'followup').rglob('*')
                if path.is_file() and path.suffix in ('.json','.jsonl')
                and path.name!='PROVENANCE.json'}
    assert scientific <= set(entries), 'follow-up scientific provenance coverage differs'
    for name,item in entries.items():
        path=ROOT/name
        assert path.is_file() and not path.is_symlink(), f'follow-up file missing: {name}'
        data=path.read_bytes()
        assert digest(data)==item['released_sha256'], f'follow-up release checksum: {name}'
        assert len(data)==item['released_bytes'], f'follow-up size: {name}'
        if path.suffix in ('.json','.jsonl'):
            value=json.loads(data) if path.suffix=='.json' else [json.loads(line) for line in data.decode().splitlines()]
            assert digest(canonical(value))==item['released_canonical_sha256'], f'follow-up canonical checksum: {name}'
            masked=copy.deepcopy(value)
            for change in item['path_string_changes']:
                assert change['path'][-1] in {'model_id','path','resolved_path','model','directory','8bit','bf16','manifest_path','source_model_path','precision_provenance_path','model_8bit','model_bf16','output','loaded_model_label','local_model_directory'}, 'unexpected anonymization field'
                cursor=masked
                for key in change['path'][:-1]: cursor=cursor[key]
                key=change['path'][-1]
                assert isinstance(cursor[key],str), 'anonymization must only change strings'
                assert digest(cursor[key].encode())==change['released_string_sha256'], 'anonymization pointer/hash'
                cursor[key]={'anonymized_path_metadata':True}
            assert digest(canonical(masked))==item['unchanged_content_sha256'], f'preserved content checksum: {name}'
        if item['transformation']=='verbatim':
            assert item['original_sha256']==item['released_sha256'], f'verbatim provenance: {name}'
    from kda_followup.manifest import load_manifest
    manifest=load_manifest(ROOT/'followup/manifests/main_v1.json')
    assert len(manifest['records'])==80
    expected={r['record_id']:r for r in manifest['records']}
    assert sum(r['split']=='evaluation' for r in expected.values())==40
    main=read('followup/analysis/kimi8_analysis_v2.json')
    assert main['analysis_status']=='complete' and main['completed_records']==80
    assert main['analysis_provenance']['analyzer_sha256']==digest((ROOT/'code/kda_followup/analyze.py').read_bytes())
    plan=read('followup/manifests/precision_subset_v1.json')
    ids=plan['record_ids']; assert len(ids)==len(set(ids))==8
    assert plan['manifest_sha256']==manifest['manifest_sha256']
    token_check=read('followup/provenance/precision_token_equality_v1.json')
    assert token_check['all_equal'] and token_check['plan_sha256']==plan['plan_sha256']
    arrays=0
    for folder,wanted in (('kimi8_v2',list(expected)),('kimi_bf16_v1',ids),('kimi_derived8_v1',ids)):
        directory=ROOT/'followup/raw'/folder
        paths={p.stem:p for p in directory.glob('kda-fu-*.json')}
        assert set(paths)==set(wanted), f'raw context inventory: {folder}'
        config=read(f'followup/raw/{folder}/run_config.json')
        assert not config['tiny'] and config['max_new_tokens']==24
        assert config.get('evaluator_sha256',config['runner_sha256'])==digest((ROOT/'code/kda_followup/run.py').read_bytes())
        for rid,path in paths.items():
            row=json.loads(path.read_text())
            assert row['record_id']==rid and row['status']=='complete' and not row['tiny']
            assert row['model_id']==config['model_id'] and row['manifest_sha256']==manifest['manifest_sha256']
            assert set(row['conditions'])==set(manifest['protocol']['conditions'])
            for condition in row['conditions'].values():
                for group in ('target_probes','retained_probes','continuations'):
                    assert [p['probe_id'] for p in condition[group]]==[p['probe_id'] for p in expected[rid][group]]
                for group in ('target_probes','retained_probes'):
                    assert all(isinstance(p['greedy_text'],str) for p in condition[group] if 'greedy_exact' in p)
            for audit in ('replay_vs_never','repeat_never_vs_never'):
                value=row['state_checks'][audit]
                assert value['all_equal'] and value['max_abs_diff']==0 and value['logits']['max_abs_diff']==0
            arrays+=1
            if folder=='kimi8_v2':
                assert main['analysis_provenance']['raw_result_canonical_sha256'][rid]==original_hash(path.relative_to(ROOT).as_posix(),True)
    derived=read('followup/provenance/kimi_derived8_provenance_v1.json')
    bf=read('followup/provenance/kimi_bf16_provenance.json')
    assert derived['source_original_bf16_audit']['official_weight_checks']==bf['official_weight_checks']
    assert len(bf['official_weight_checks'])==20 and not derived['weight_files_saved']
    from kda_followup.precision import require_precision, official_metadata
    require_precision(bf, 'bf16'); require_precision(derived, '8bit')
    pinned = {item['rfilename']:item for item in official_metadata()['siblings'] if item['rfilename'].endswith('.safetensors')}
    assert {item['file'] for item in bf['official_weight_checks']}==set(pinned)
    for item in bf['official_weight_checks']:
        assert item['sha256']==pinned[item['file']]['lfs']['sha256']
        assert item['bytes']==pinned[item['file']]['size']
    assert derived['packed_weight_tensor_count']>0 and derived['quantization']=={'bits':8,'group_size':64,'mode':'affine'}
    dconfig=read('followup/raw/kimi_derived8_v1/run_config.json')
    assert dconfig['runner_sha256']==digest((ROOT/'code/kda_followup/derived_precision.py').read_bytes())
    for rid in ids:
        assert read(f'followup/raw/kimi_derived8_v1/{rid}.json')['tokens']==read(f'followup/raw/kimi_bf16_v1/{rid}.json')['tokens']
    for folder in ('transport_floor_kimi8_v1','transport_floor_bf16_v1','transport_floor_derived8_v1'):
        config=read(f'followup/raw/{folder}/run_config.json')
        summary=read(f'followup/raw/{folder}/summary.json')
        assert not config['tiny_plumbing_only'] and config['plan_sha256']==plan['plan_sha256']
        assert summary['status']=='complete' and summary['record_ids']==ids and summary['n_records']==8
        rows=[read(f'followup/raw/{folder}/{rid}.json') for rid in ids]
        assert all(r['status']=='complete' and r['record_id']==rid for r,rid in zip(rows,ids))
        for key,rawkey in (('global_residual_to_reconstruction_error_ratios','residual_to_reconstruction_error_ratio'),('global_residual_to_max_empirical_floor_ratios','residual_to_max_empirical_floor_ratio')):
            assert summary[key]==[r['global'][rawkey] for r in rows]
        for key,rawkey in (('minimum_global_ratio','residual_to_reconstruction_error_ratio'),('maximum_global_ratio','residual_to_reconstruction_error_ratio'),('minimum_global_combined_floor_ratio','residual_to_max_empirical_floor_ratio'),('maximum_global_combined_floor_ratio','residual_to_max_empirical_floor_ratio'),('max_present_reconstruction_relative_error','present_reconstruction_relative_error'),('max_omitted_reconstruction_relative_error','omitted_reconstruction_relative_error')):
            values=[r['global'][rawkey] for r in rows if r['global'][rawkey] is not None]
            if key in summary:
                assert summary[key]==(min(values) if key.startswith('minimum') else max(values))
            else:
                assert folder=='transport_floor_derived8_v1' and key in ('minimum_global_ratio','maximum_global_ratio')
        for row in rows:
            g=row['global']
            for key in ('native_transport_residual_frobenius','serial_transport_residual_frobenius','present_reconstruction_error_frobenius','omitted_reconstruction_error_frobenius','frozen_input_identity_error_frobenius','residual_change_from_reconstruction_frobenius','native_present_minus_omitted_frobenius','native_present_frobenius','native_omitted_frobenius'):
                assert g[key]==math.sqrt(sum(layer[key]**2 for layer in row['per_layer']))
            for value in [g,*row['per_layer']]:
                assert value['reconstruction_error_triangle_sum']==value['present_reconstruction_error_frobenius']+value['omitted_reconstruction_error_frobenius']
                assert value['max_empirical_floor']==max(value['reconstruction_error_triangle_sum'],value['frozen_input_identity_error_frobenius'])
                assert value['residual_to_max_empirical_floor_ratio']==value['native_transport_residual_frobenius']/value['max_empirical_floor']
    qmanifest=read('followup/manifests/qwen_confirmation_20260911.json')
    qwen=read('followup/raw/qwen_confirmation/results.json')
    rows=qwen['records']
    assert qwen['manifest_sha256']==original_hash('followup/manifests/qwen_confirmation_20260911.json')
    assert qwen['implementation_sha256']==digest((ROOT/'code/recurrent_families/qwen_confirmation.py').read_bytes())
    assert qwen['base_audit_implementation_sha256']==digest((ROOT/'code/gemma_sv/qwen_replay_audit.py').read_bytes())
    assert len(rows)==32 and [r['record_id'] for r in rows]==[r['record_id'] for r in qmanifest['records']]
    assert qwen['summary']==read('followup/analysis/qwen_summary.json')
    assert all(r['status']=='completed' and [c['suffix'] for c in r['cases']]==['short','long'] for r in rows)
    assert all(all(c['checks'].values()) for r in rows for c in r['cases'])
    maintenance=read('followup/raw/maintenance_kimi8_v1.json')
    assert maintenance['status']=='complete' and not maintenance['tiny_plumbing_only']
    assert len(maintenance['amendments'])==12 and len(maintenance['sequences'])==12
    assert sum(len(s['steps']) for s in maintenance['sequences'])==36
    msummary=read('followup/analysis/maintenance_summary_v1.json')
    assert msummary['raw_output_sha256']==original_hash('followup/raw/maintenance_kimi8_v1.json')
    assert msummary['sequential']['independent_record_pairs']==6
    print(f'PASS follow-up provenance; {arrays} Kimi contexts, 24 arithmetic diagnostics, 32 paired Qwen contexts, 12 amendments and 36 sequential transitions')
    return {'status':'pass','kimi_contexts':arrays,'arithmetic_diagnostics':24,'qwen_records':32,'qwen_suffix_evaluations':64,'amendment_records':12,'sequential_independent_pairs':6,'sequential_transitions':36}

if __name__=='__main__': verify()
