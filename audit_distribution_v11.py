"""2026 sequential watch of historical-2025 residual bootstrap. Research-only."""
import json,argparse
import numpy as np,pandas as pd
from pathlib import Path
from train_matchup_v11 import feature_data
from core.margin_v11 import strength_snapshots,margin_prediction
from core.matchup_v11 import tidy_data,projection,bootstrap_simulation

FIELDS=('PTS','FG2A','FG3A','FG2M','FG3M','FTA','FTM','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK')

def run(team_path,model_path,out_path,n=1000):
 d=tidy_data(pd.read_csv(team_path));r=json.loads(Path(model_path).read_text())
 ft=feature_data(d,r,2026)
 obs={};seen=[]
 mlookup={str(z['GAME_ID']):z for z in strength_snapshots(d,2026,r['margin_model']['net_prior_games']).to_dict('records')}
 for i,(gid,g) in enumerate(ft.groupby('GAME_ID')):
  if len(g)!=2:continue
  h=g[g.HOME_AWAY=='H'].iloc[0].to_dict();a=g[g.HOME_AWAY=='A'].iloc[0].to_dict()
  pr=projection(h,a,r['chosen_reconciliation_sd_2024'],r.get('home_adjustment_from_2024_made_baskets'))
  if str(gid) in mlookup:
   ind=margin_prediction(pd.Series(mlookup[str(gid)]),r['margin_model'])
   z=r['margin_model']['blend_weight']
   tar=(1-z)*(pr['home']['PTS']-pr['away']['PTS'])+z*ind
   pr=projection(h,a,r['chosen_reconciliation_sd_2024'],r['home_adjustment_from_2024_made_baskets'],tar,r['margin_model']['score_component_weights'])
  sim=bootstrap_simulation(pr,r['matched_holdout_2025_residuals'],n=n,seed=121+i,sd=r['chosen_reconciliation_sd_2024'])
  actual=d[d.GAME_ID==gid]
  for side,s in (('home','H'),('away','A')):
   row=actual[actual.HOME_AWAY==s].iloc[0]
   for k in FIELDS:
    val=float(row[k]);pred=sim[side][k]
    q10,q90=np.quantile(pred,[.05,.95])
    obs.setdefault(k,[]).append({'covered':float(q10<=val<=q90),'pred_mean':float(pred.mean()),'actual':val,
      'p10':float(q10),'p90':float(q90)})
  seen.append(gid)
 report={'status':'SHADOW_NOT_PRICING_READY','production_pricing_gate':False,
   'market_definition':'official_inclusive_v2','test_season':2026,
   'distinct_games':len(seen),'side_observations':len(seen)*2,'draws_each':n,
   'source':'2025 paired game residual bootstrap with 2026 chronological watch; 2026 repeatedly examined, not independent',
   'markets':{k:{'coverage_90':round(float(np.mean([z['covered'] for z in rows])),3),
           'RMSE':round(float(np.mean([(z['pred_mean']-z['actual'])**2 for z in rows])**.5),3),
           'mean_predictive_sd':None,'n':len(rows)} for k,rows in obs.items()}}
 Path(out_path).parent.mkdir(parents=True,exist_ok=True)
 Path(out_path).write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(pd.DataFrame(report['markets']).T.to_string())
 print('Wrote',out_path)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--team',default='data/euroleague/exports/team_games.csv');p.add_argument('--model',default='data/euroleague/shadow_v11/matchup_v11_report.json');p.add_argument('--out',default='data/euroleague/shadow_v11/distribution_v11_report.json');p.add_argument('--draws',type=int,default=1200);z=p.parse_args()
 run(z.team,z.model,z.out,z.draws)
