import json
from pathlib import Path
from core.matchup_v11 import COUNTS,RATES
p=Path('data/euroleague/shadow_v11/matchup_v11_report.json')
a=Path('data/euroleague/shadow_v11/distribution_v11_report.json')
if not (p.exists() and a.exists()):raise SystemExit('Missing required v1.1 calibration reports')
x=json.loads(p.read_text());y=json.loads(a.read_text())
if x.get('live_contract')!='SAME_SEASON_ONLY' or x.get('market_definition')!='official_inclusive_v2':raise SystemExit('Wrong source contract')
if x.get('training_seasons')!=[2023,2024,2025]:raise SystemExit('Wrong train seasons')
if y.get('production_pricing_gate') is not False or y.get('status')!='SHADOW_NOT_PRICING_READY':raise SystemExit('Must block production odds')
if 'margin_model' not in x or x['margin_model'].get('net_prior_games') is None:
 raise SystemExit('Untrained current-season margin component')
if abs(sum(x['margin_model']['score_component_weights'].values())-1)>1e-6:
 raise SystemExit('Margin allocation weights do not conserve total points')
if x.get('test_2025_sides',0)<300 or x.get('watch_2026_sides',0)<60:raise SystemExit('Insufficient check data')
if len(x.get('matched_holdout_2025_residuals',[]))<120:raise SystemExit('Missing paired empirical errors')
print('V1.1 safe release: verified sample chronology, explicit current-season contract, production price HOLD')
