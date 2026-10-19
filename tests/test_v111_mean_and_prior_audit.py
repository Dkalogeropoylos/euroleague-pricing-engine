import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import json
from core.matchup_v11 import tidy_data, full_current_only_forecast, bootstrap_simulation, ALL
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/euroleague/exports/team_games.csv'
REPORT=ROOT/'data/euroleague/shadow_v11/matchup_v11_report.json'

@pytest.mark.parametrize('home,away',[('HTA','VIR'),('PAN','BAS')])
def test_mc_centers_historical_residuals_and_preserves_means(home,away):
 if not DATA.exists() or not REPORT.exists():pytest.skip('Full verified 2026 dataset not bundled')
 d=tidy_data(pd.read_csv(DATA));r=json.loads(REPORT.read_text())
 pred,hh,aa=full_current_only_forecast(d,r,2026,home,away,'2026-10-20')
 sim=bootstrap_simulation(pred,r['matched_holdout_2025_residuals'],n=6000,seed=202611,
                          sd=r['chosen_reconciliation_sd_2024'])
 for side in ('home','away'):
  for k,tolerance in [('FG3M',.30),('PTS',.40),('MARKET_OREB',.28),('MARKET_REB',.40)]:
   assert np.mean(sim[side][k])==pytest.approx(pred[side][k],abs=tolerance)

@pytest.mark.parametrize('home,away',[('HTA','VIR'),('PAN','BAS')])
def test_every_stat_provenance_and_prior_share_are_exposed(home,away):
 if not DATA.exists() or not REPORT.exists():pytest.skip('Full verified 2026 dataset not bundled')
 d=tidy_data(pd.read_csv(DATA));r=json.loads(REPORT.read_text())
 _,h,a=full_current_only_forecast(d,r,2026,home,away,'2026-10-20')
 for raw in (h,a):
  for k in ALL:
   assert np.isfinite(raw[k+'_RAW_OFF'])
   assert np.isfinite(raw[k+'_RAW_DEF'])
   assert 0 <= raw[k+'_PRIOR_SHARE_OFF'] <= 1
   assert 0 <= raw[k+'_PRIOR_SHARE_DEF'] <= 1
  assert raw['MARKET_DREB_PRIOR_SHARE_OFF']==pytest.approx(16/(6+16))
