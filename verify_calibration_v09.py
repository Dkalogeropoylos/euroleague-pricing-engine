"""Checks output of an OFFLINE research-only team-count benchmark."""
from pathlib import Path
import json, math
from calibrate_shrinkage_v09 import MARKETS

P=Path('data/euroleague/shadow_v09/calibration_v09_report.json')
if not P.exists():raise SystemExit('Missing v0.9 calibrated report')
r=json.loads(P.read_text())
assert r['status']=='SHADOW_NOT_PRICING_READY'
assert r['market_definition']=='official_inclusive_v2'
assert r['dataset']['games']>=500
assert set(r['markets'])==set(MARKETS)
for k,v in r['markets'].items():
    assert math.isfinite(v['blocked_cv']['rmse_candidate'])
    assert 2 <=v['selected']['prior_games']<=32
    assert v['watch_2026']['n']>0
print('PASS: count benchmark is shadow-only, schema correct, validation present')
