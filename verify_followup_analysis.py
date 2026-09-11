"""Recompute all follow-up statistics offline using NumPy; never load a model.

Recorded external snapshot audits are checked as provenance, not re-executed.
Only path fields and original-vs-released file hashes differ during regeneration.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from verify_followup import read, provenance, original_hash, verify


def main():
    verify()
    import numpy as np
    if np.__version__!='2.4.6':
        raise RuntimeError('Exact stored bootstrap reproduction requires numpy==2.4.6; see requirements-analysis.txt')
    from kda_followup.analyze import analyze, load_results
    from kda_followup.manifest import load_manifest
    from kda_followup.maintenance_report import summarize
    from kda_followup import precision
    from kda_followup.precision_report import attach_floors
    from recurrent_families.qwen_confirmation import summarize as qwen_summarize
    from kda_followup.auc_sensitivity import run as auc_sensitivity
    from kda_followup.export_tables import render_tables
    mf=ROOT/'followup/manifests/main_v1.json'
    result=analyze(load_manifest(mf),load_results([ROOT/'followup/raw/kimi8_v2']))
    expected=read('followup/analysis/kimi8_analysis_v2.json')
    comparison=copy.deepcopy(expected)
    # Embedded hashes refer to original pre-anonymization raw records. The release
    # map independently verifies both values; the current analyzer hashes released records.
    mapped={rid:provenance()[f'followup/raw/kimi8_v2/{rid}.json']['released_canonical_sha256'] for rid in expected['analysis_provenance']['raw_result_canonical_sha256']}
    comparison['analysis_provenance']['raw_result_canonical_sha256']=mapped
    assert result==comparison, 'complete main analysis regeneration differs'
    print('PASS main analysis: every statistic, interval, attack and state aggregate')
    m=summarize(read('followup/raw/maintenance_kimi8_v1.json'))
    m_expected=read('followup/analysis/maintenance_summary_v1.json')
    m_expected.pop('raw_output_sha256')  # Original raw hash already checked by verify().
    assert m==m_expected, 'maintenance statistics differ'
    print('PASS maintenance: exact/containment, record bootstrap, six independent sequence groups')
    q=qwen_summarize(read('followup/raw/qwen_confirmation/results.json')['records'],read('followup/manifests/qwen_confirmation_20260911.json'))
    assert q==read('followup/analysis/qwen_summary.json'), 'Qwen statistics differ'
    print('PASS Qwen: all paired suffix summaries, record bootstrap and Wilson intervals')
    with tempfile.TemporaryDirectory(prefix='kda-followup-offline-') as temporary:
        temporary=Path(temporary)
        plan=read('followup/manifests/precision_subset_v1.json')
        plan['manifest_path']=str(mf)
        planpath=temporary/'plan.json';planpath.write_text(json.dumps(plan))
        audit_bf=read('followup/provenance/kimi_bf16_provenance.json')
        audit_cached=read('followup/provenance/kimi_8bit_provenance.json')
        # This explicitly substitutes *recorded* source metadata in the original
        # analysis function's preflight. It performs no filesystem weight check.
        def recorded_snapshot(path):
            for item in (audit_bf,audit_cached):
                if str(path)==item['path']: return item
            raise ValueError(f'Unknown recorded model path: {path}')
        precision.inspect_snapshot=recorded_snapshot
        for variant,folder,audit,floor,name in (
            ('primary','kimi_derived8_v1','kimi_derived8_provenance_v1.json','transport_floor_derived8_v1','precision_comparison_v1.json'),
            ('secondary','kimi8_v2','kimi_8bit_provenance.json','transport_floor_kimi8_v1','precision_cached8_comparison_v1.json')):
            got=precision.summarize_pair(planpath,ROOT/'followup/raw'/folder,ROOT/'followup/raw/kimi_bf16_v1',ROOT/'followup/provenance/precision_token_equality_v1.json',ROOT/'followup/provenance'/audit,ROOT/'followup/provenance/kimi_bf16_provenance.json',temporary/name)
            got=attach_floors(got,ROOT/'followup/raw'/floor,ROOT/'followup/raw/transport_floor_bf16_v1')
            want=read('followup/analysis/'+name)
            for key in ('source_directories','provenance_audits'):
                assert {k:str(Path(v).relative_to(ROOT)) for k,v in got[key].items()}==want[key], key
                got[key]=want[key]
            for label in ('8bit','bf16'):
                assert str(Path(got['arithmetic_diagnostics'][label]['directory']).relative_to(ROOT))==want['arithmetic_diagnostics'][label]['directory']
                got['arithmetic_diagnostics'][label]['directory']=want['arithmetic_diagnostics'][label]['directory']
            assert got==want, variant+' precision statistics differ'
            print('PASS '+variant+' precision: every own-reference metric, paired difference and own arithmetic diagnostic')
        released_analysis=temporary/'released-analysis.json';released_analysis.write_text(json.dumps(result))
        a=auc_sensitivity(mf,ROOT/'followup/raw/kimi8_v2',released_analysis)
        assert a==read('followup/analysis/auc_pairing_sensitivity_v2.json'), 'AUC sensitivity differs'
        print('PASS complete exploratory off-diagonal AUC sensitivity')
    tex=render_tables(expected,'followup/analysis/kimi8_analysis_v2.json',original_hash('followup/analysis/kimi8_analysis_v2.json'))
    tex=tex.replace('% Source SHA-256:', '% Original source SHA-256 before path anonymization:')
    assert tex==(ROOT/'followup/figures/cohort_results_table.tex').read_text(), 'table regeneration differs'
    print('PASS byte-identical follow-up numerical tables')
    print('All comparisons used released data, recorded source provenance and CPU statistics only. External weight SHA claims were not independently repeated.')

if __name__=='__main__': main()
