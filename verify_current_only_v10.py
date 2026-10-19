"""Hard gate that historical team priors are absent from live v1.0 forecasts."""
import json
from pathlib import Path
import pandas as pd
from core.structural_v06 import verified_games,event_rate_rows
from core.current_only_v10 import current_only_upcoming,FACTORS

report=json.loads(Path('data/euroleague/shadow_v10/current_only_v10_report.json').read_text())
assert report['status']=='SHADOW_NOT_PRICING_READY'
assert report['live_input_contract']=='CURRENT_SEASON_ONLY_OFFENSE_ALLOWED_AND_LEAGUE'
assert report['training_seasons']==[2023,2024,2025]
assert report['market_definition']=='official_inclusive_v2'
assert set(report['factors'])==set(FACTORS)
assert report['watch_2026']['games_evaluated']>=20
base=Path('data/euroleague/exports')
x=verified_games(pd.read_csv(base/'team_games.csv'),pd.read_csv(base/'player_games.csv'))
ev=event_rate_rows(x)
last=ev[ev.SEASON.eq(2026)]
assert last.GAME_ID.nunique()>=40
teams=last[['TEAM_ID']].drop_duplicates().TEAM_ID.astype(str).tolist()
h,a=current_only_upcoming(ev,season=2026,home_id=teams[0],away_id=teams[1],asof_date='2026-10-31')
# Whole earlier seasons can change, but unchanged current rows must imply identical live inputs.
ev2=ev[ev.SEASON.eq(2026)]
h2,a2=current_only_upcoming(ev2,season=2026,home_id=teams[0],away_id=teams[1],asof_date='2026-10-31')
for t,r in [(h,h2),(a,a2)]:
 for k in FACTORS:
  for v in ('BASE','OFF','DEF'):
   assert abs(float(t[f'{k}_{v}'])-float(r[f'{k}_{v}']))<1.e-12,(k,v)
print('Verified CURRENT-ONLY live input contract, historical-only coefficient lineage, report coverage')
