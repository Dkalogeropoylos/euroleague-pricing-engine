"""Fail closed if data or any shadow report is stale or inconsistent."""
from __future__ import annotations
import json
from pathlib import Path

def main(base:Path=Path('data/euroleague')):
    mf=json.loads((base/'exports/manifest.json').read_text())
    c=json.loads((base/'exports/league_coefficients.json').read_text())
    v06=json.loads((base/'shadow_v06/structural_v06_report.json').read_text())
    v07=json.loads((base/'shadow_v07/efficiency_benchmark_v07.json').read_text())
    total=json.loads((base/'shadow_v07/pregame_totals_audit_v07.json').read_text())
    dist=json.loads((base/'shadow_v07/distribution_audit_v07.json').read_text())
    if dist.get('production_pricing_gate') is not False:
        raise RuntimeError('Distribution pricing authorization wrongly enabled')
    if total.get('status')!='SHADOW_PREGAME_TOTALS_AUDIT' or total.get('games',0)<100:
        raise RuntimeError('Missing independent pregame TOTAL audit')
    for obj,key in ((mf,'market_schema_version'),(c,'market_schema_version'),
                    (v06,'market_definition'),(v07,'data_definition')):
        if obj.get(key)!='official_inclusive_v2':
            raise RuntimeError(f'Market settlement mismatch in {key}')
    if v06.get('status')!='SHADOW_NOT_PRICING_READY' or v07.get('status')!='SHADOW_BENCHMARK_ONLY':
        raise RuntimeError('Models incorrectly promoted to production')
    if v07.get('outer_games',0)<100 or not v07.get('future_validation_required'):
        raise RuntimeError('Independent model validation status or count bad')
    print('V07_GATE_PASS: official-inclusive v2 + independent efficiency shadow; odds UNCALIBRATED')
if __name__=='__main__':main()
