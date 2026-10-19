"""v1.1 anti-leakage, joint market conservation, uncertainty and bookmaker arithmetic tests."""
import numpy as np,pandas as pd,pytest
from core.matchup_v11 import live_features,projection,bootstrap_simulation,quality_gate,tidy_data,COUNTS,ALL,full_current_only_forecast
from core.trader_markets_v07 import build_trader_board
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/euroleague/exports/team_games.csv'
REPORT=ROOT/'data/euroleague/shadow_v11/matchup_v11_report.json'

@pytest.fixture(scope='module')
def data_and_report():
 if not DATA.exists() or not REPORT.exists():pytest.skip('Real verified snapshot not available')
 return tidy_data(pd.read_csv(DATA)),json.loads(REPORT.read_text())

def test_current_only_provenance(data_and_report):
 d,r=data_and_report
 h1,a1=live_features(d,r,2026,'PAN','BAS','2026-10-21')
 h2,a2=live_features(d[d.SEASON.eq(2026)],r,2026,'PAN','BAS','2026-10-21')
 for k in ALL:
  for suff in ('OFF','DEF','BASE','PRED'):
   assert h1[k+'_'+suff]==pytest.approx(h2[k+'_'+suff],rel=0,abs=1e-12)
   assert a1[k+'_'+suff]==pytest.approx(a2[k+'_'+suff],rel=0,abs=1e-12)


def test_future_matches_never_leak(data_and_report):
 d,r=data_and_report
 early='2026-10-15'
 h1,a1=live_features(d,r,2026,'PAN','BAS',early)
 after=d[d.DAY<pd.Timestamp(early,tz='UTC')]
 h2,a2=live_features(after,r,2026,'PAN','BAS',early)
 for k in ALL:assert h1[k+'_PRED']==pytest.approx(h2[k+'_PRED'],abs=1e-10)


def test_joint_market_identical_accounts(data_and_report):
 d,r=data_and_report
 for h,a in [('PAN','BAS'),('HTA','VIR'),('PRS','VIR'),('OLY','ULK')]:
  hh,aa=live_features(d,r,2026,h,a,'2026-10-21')
  pred=projection(hh,aa,r['chosen_reconciliation_sd_2024'],r.get('home_adjustment_from_2024_made_baskets'))
  p=pred['pace']
  for side in ('home','away'):
   x=pred[side]
   assert x['PTS']==pytest.approx(2*x['FG2M']+3*x['FG3M']+x['FTM'])
   assert x['MARKET_REB']==pytest.approx(x['MARKET_OREB']+x['MARKET_DREB'])
   assert x['AST'] <= x['FGM']+1e-8
   assert x['FG3M'] <= x['FG3A']
   assert x['STL'] <= pred['away' if side=='home' else 'home']['PLAYER_CREDITED_TO']+1e-8
   assert x['FGA']+.44*x['FTA']+x['MARKET_TO']-x['MARKET_OREB']==pytest.approx(p,abs=1e-7)


def test_paired_mc_conservation_and_three_way(data_and_report):
 d,r=data_and_report
 hh,aa=live_features(d,r,2026,'PAN','BAS','2026-10-21')
 pr=projection(hh,aa,r['chosen_reconciliation_sd_2024'],r.get('home_adjustment_from_2024_made_baskets'))
 sim=bootstrap_simulation(pr,r['matched_holdout_2025_residuals'],n=1600,seed=37,sd=r['chosen_reconciliation_sd_2024'])
 for side in ('home','away'):
  x=sim[side]
  np.testing.assert_array_equal(x['PTS'],2*x['FG2M']+3*x['FG3M']+x['FTM'])
  np.testing.assert_array_equal(x['MARKET_REB'],x['MARKET_OREB']+x['MARKET_DREB'])
  assert np.all(x['FG3M']<=x['FG3A']);assert np.all(x['AST']<=x['FG2M']+x['FG3M'])
 board=build_trader_board(sim,'PAN','BAS')
 assert len(board['teams'])==56
 assert len(board['match'])==28
 assert len(board['most'])==42
 for key,vigor,group_size in [('teams',1.09,2),('match',1.09,2),('most',1.15,3)]:
  frame=board[key]
  for i in range(0,len(frame),group_size):
   q=frame.iloc[i:i+group_size]
   assert q.P.sum()==pytest.approx(1.,abs=1e-7)
   bk=q['Book_109'] if vigor==1.09 else q['Book_115']
   if np.isfinite(bk).all():assert (1/bk).sum()==pytest.approx(vigor,abs=1e-8)


def test_release_gate_never_passes_without_distribution_calibration(data_and_report):
 d,r=data_and_report
 hh,aa=live_features(d,r,2026,'PAN','BAS','2026-10-21')
 pr=projection(hh,aa,r['chosen_reconciliation_sd_2024'],r.get('home_adjustment_from_2024_made_baskets'))
 result=quality_gate(pr,hh,aa,calibration={'production_pricing_gate':False})
 assert result['status']=='HOLD'
 assert any('not independently verified' in t for t in result['flags'])


def test_market_definition_and_train_seasons(data_and_report):
 d,r=data_and_report
 assert r['training_seasons']==[2023,2024,2025]
 assert r['live_contract']=='SAME_SEASON_ONLY'
 assert r['market_definition']=='official_inclusive_v2'
 assert all(z['beta'] in (0,.25,.5,.75,1) for z in r['params'].values())


def test_all_380_current_teams_no_internal_accounting_breaks(data_and_report):
 d,r=data_and_report
 ids=sorted(d[d.SEASON.eq(2026)].TEAM_ID.unique())
 count=0
 for home in ids:
  for away in ids:
   if home==away:continue
   h,a=live_features(d,r,2026,home,away,'2026-10-21')
   pr=projection(h,a,r['chosen_reconciliation_sd_2024'],r['home_adjustment_from_2024_made_baskets'])
   count+=1
   for side in ('home','away'):
    q=pr[side];opp=pr['away' if side=='home' else 'home']
    assert q['PTS']==pytest.approx(2*q['FG2M']+3*q['FG3M']+q['FTM'],abs=1e-7)
    assert q['MARKET_REB']==pytest.approx(q['MARKET_OREB']+q['MARKET_DREB'],abs=1e-7)
    assert q['STL']<=opp['PLAYER_CREDITED_TO']+1e-7
    assert q['FG2M']<=q['FG2A'] and q['FG3M']<=q['FG3A'] and q['FTM']<=q['FTA']
    assert q['FG2A']+q['FG3A']+.44*q['FTA']+q['MARKET_TO']-q['MARKET_OREB']==pytest.approx(pr['pace'],abs=1e-6)
 assert count==len(ids)*(len(ids)-1)==380


def test_calibrated_margin_model_current_only_and_reconciles_counts(data_and_report):
 d,r=data_and_report
 assert r['margin_model']['blend_weight'] in (0,.25,.5,.75,1)
 for home,away in [('PAN','BAS'),('HTA','VIR'),('PRS','VIR')]:
  pred,h,a=full_current_only_forecast(d,r,2026,home,away,'2026-10-21')
  assert pred['home']['PTS']-pred['away']['PTS']==pytest.approx(pred['audit']['margin_target'],abs=.8)
  for side in ('home','away'):
   p=pred[side]
   assert p['PTS']==pytest.approx(2*p['FG2M']+3*p['FG3M']+p['FTM'])
   assert p['MARKET_REB']==pytest.approx(p['MARKET_OREB']+p['MARKET_DREB'])


def test_margin_live_features_ignore_all_other_seasons(data_and_report):
 from core.margin_v11 import upcoming_margin
 d,r=data_and_report
 cfg=r['margin_model']
 full=upcoming_margin(d,2026,'PAN','BAS','2026-10-21',cfg)
 current=upcoming_margin(d[d.SEASON==2026],2026,'PAN','BAS','2026-10-21',cfg)
 assert full==pytest.approx(current,rel=0,abs=1e-12)
 assert abs(sum(cfg['score_component_weights'].values())-1)<1e-10


def test_full_forecast_mc_market_accounting_and_hold(data_and_report):
 d,r=data_and_report
 pr,hh,aa=full_current_only_forecast(d,r,2026,'PAN','BAS','2026-10-21')
 sm=bootstrap_simulation(pr,r['matched_holdout_2025_residuals'],n=1800,seed=88,
       sd=r['chosen_reconciliation_sd_2024'])
 for side in ('home','away'):
  a=sm[side]
  np.testing.assert_array_equal(a['PTS'],2*a['FG2M']+3*a['FG3M']+a['FTM'])
  np.testing.assert_array_equal(a['MARKET_REB'],a['MARKET_OREB']+a['MARKET_DREB'])
 assert quality_gate(pr,hh,aa,None,{'production_pricing_gate':False})['status']=='HOLD'
