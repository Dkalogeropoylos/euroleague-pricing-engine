"""Train v1.1 on 2024 (historical validation), evaluate 2025, watch 2026.
Reports contain paired out-of-year residual samples for exploratory MC ONLY.
"""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from core.matchup_v11 import (tidy_data,snapshots,select_params,projection,COUNTS,RATES,ORDER,ALL)
from core.margin_v11 import select_margin_config,strength_snapshots,margin_prediction

METRICS=('POSS','PTS','FG2A','FG3A','FG2M','FG3M','FTA','FTM','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK')

def feature_data(d,config,season):
 caches={p:snapshots(d,season,p) for p in set(z['prior'] for z in config['params'].values())}
 key_cols=('GAME_ID','TEAM_ID','HOME_AWAY')
 first=caches[next(iter(caches))][list(key_cols)+['DAY','SEASON','OPP_ID','N_OWN','N_OPP']].copy()
 for k,z in config['params'].items():
  p=z['prior'];col=[k+'_OFF',k+'_DEF',k+'_BASE',k+'_ACTUAL']
  first=first.merge(caches[p][list(key_cols)+col],on=list(key_cols),validate='one_to_one')
  first[k+'_PRED']=np.clip(first[k+'_OFF']+z['beta']*(first[k+'_DEF']-first[k+'_BASE']),0,150.)
 return first

def audit_year(d,config,year,sd=None,with_residuals=False,home_adjustment=None,margin_config=None):
 ft=feature_data(d,config,year)
 rec=[];resid=[];mean_errors=[]
 margin_lookup={}
 if margin_config:
  ms=strength_snapshots(d,year,margin_config['net_prior_games'])
  margin_lookup={str(z['GAME_ID']):z for z in ms.to_dict('records')}
 for game,g in ft.groupby('GAME_ID',sort=False):
  if len(g)!=2:continue
  h=g[g.HOME_AWAY=='H'].iloc[0].to_dict();a=g[g.HOME_AWAY=='A'].iloc[0].to_dict()
  pred=projection(h,a,sd,home_adjustment)
  if margin_config and str(game) in margin_lookup:
   marg=margin_prediction(pd.Series(margin_lookup[str(game)]),margin_config)
   z=margin_config['blend_weight']
   target=(1-z)*(pred['home']['PTS']-pred['away']['PTS'])+z*marg
   pred=projection(h,a,sd,home_adjustment,target,margin_config['score_component_weights'])
  realg=d[d.GAME_ID==game]; actual={r.HOME_AWAY:r for _,r in realg.iterrows()}
  if len(actual)!=2:continue
  for name,s,rate in [('home','H',h),('away','A',a)]:
   obs=actual[s]; p=pred[name]
   row={'SEASON':year,'GAME_ID':game,'SIDE':s,'TEAM_ID':obs.TEAM_ID,'OPP_ID':obs.OPP_ID,
    'DAY':str(obs.DAY),'POSS':float(obs.SHARED_PACE_LABEL),'POSS_pred':float(pred['pace'])}
   for k in METRICS[1:]:
    row[k]=float(obs[k]);row[k+'_pred']=float(p[k])
   row['raw_FG3A_pred']=rate['FG3A_PRED'];row['base_FG3A']=rate['FG3A_OFF'];
   rec.append(row)
  if with_residuals:
   rrow={'GAME_ID':game}
   for name,s in [('home','H'),('away','A')]:
    obs=actual[s];pr=pred[name]
    for k in METRICS[1:]:rrow[name+'_'+k]=float(obs[k]-pr[k])
    rrow[name+'_P2']=float(obs.FG2M/max(obs.FG2A,1)-pr['FG2M']/max(pr['FG2A'],1))
    rrow[name+'_P3']=float(obs.FG3M/max(obs.FG3A,1)-pr['FG3M']/max(pr['FG3A'],1))
    rrow[name+'_PF']=float(obs.FTM/max(obs.FTA,1)-pr['FTM']/max(pr['FTA'],1))
    rrow[name+'_AST_RATE']=float(obs.AST/max(obs.FGM,1)-pr['AST']/max(pr['FGM'],1))
   rrow['POSS']=float(actual['H'].SHARED_PACE_LABEL-pred['pace'])
   resid.append(rrow)
 return pd.DataFrame(rec),pd.DataFrame(resid)

def score(rows):
 if rows.empty:return {}
 return {k:{'rmse':round(float(np.mean((rows[k+'_pred']-rows[k])**2)**.5),3),
            'bias':round(float(np.mean(rows[k+'_pred']-rows[k])),3),
            'offense_only_baseline_rmse': round(float(np.mean((rows['base_FG3A']-rows['FG3A'])**2)**.5),3) if k=='FG3A' else None}
         for k in METRICS}

def run(team_path,save):
 d=tidy_data(pd.read_csv(team_path));config=select_params(d)
 rows24,_=audit_year(d,config,2024)
 # Estimate symmetric home-court mark-up on realized *makes* from 2024 only.
 # Never fit this adjustment on 2025 historical test or 2026 watch outcomes.
 home_effect={}
 for k in ('FG2M','FG3M','FTM'):
  delta=rows24[k]-rows24[k+'_pred']
  home_effect[k]=round(float((delta[rows24.SIDE.eq('H')].mean()-delta[rows24.SIDE.eq('A')].mean())/2),5)
 sd={'POSS':max(2,float(np.std(rows24.POSS_pred-rows24.POSS)))}
 for k in ORDER[1:]:sd[k]=max(1,float(np.std(rows24[k+'_pred']-rows24[k])))
 rows24_updated,_=audit_year(d,config,2024,sd,False,home_effect)
 margin_cfg=select_margin_config(d,rows24_updated)
 rows25,err25=audit_year(d,config,2025,sd,True,home_effect,margin_cfg)
 rows26,err26=audit_year(d,config,2026,sd,True,home_effect,margin_cfg)
 report={**config,'chosen_reconciliation_sd_2024':sd,'home_adjustment_from_2024_made_baskets':home_effect,'margin_model':margin_cfg,
  'test_2025':score(rows25),'watch_2026':score(rows26),
  'test_2025_sides':len(rows25),'watch_2026_sides':len(rows26),
  'distribution_status':'UNVERIFIED_DO_NOT_QUOTE','matched_holdout_2025_residuals':err25.to_dict('records')}
 path=Path(save);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
 # Human-readable chronological checks
 rows25.to_csv(path.with_name('chronological_test_2025_v11.csv'),index=False)
 rows26.to_csv(path.with_name('watch_2026_v11.csv'),index=False)
 print('SCORE 2025\n',pd.DataFrame(report['test_2025']).T.to_string())
 print('WATCH 2026\n',pd.DataFrame(report['watch_2026']).T.to_string())
 print('SAVED',path)
 return report

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--team',default='data/euroleague/exports/team_games.csv');ap.add_argument('--output',default='data/euroleague/shadow_v11/matchup_v11_report.json');args=ap.parse_args()
 run(args.team,args.output)
