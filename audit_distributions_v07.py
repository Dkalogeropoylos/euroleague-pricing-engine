"""Distribution calibration diagnostics on previously examined 194-game holdout.

DOES NOT calibrate the MC, DOES NOT authorize production prices.
Coverage uncertainty is computed from binomial sampling theory, not
custom acceptance tolerances. The holdout is known/previously inspected.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from core.structural_v06 import verified_games,event_rate_rows,chronological_design,simulate_two_sides

LABELS=('PTS','FG3A','FTA','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST')

def run(d:pd.DataFrame,report:dict,n:int=1500,seed:int=117)->dict:
    x=chronological_design(event_rate_rows(d))
    cutoff=pd.Timestamp(report['cutoff_utc'])
    hold=x.loc[x.DAY>=cutoff]
    models=report['coefficients']  # outer-fitted coefficients, never after-audit refit
    records={k:[] for k in LABELS+('GAME_TOTAL','GAME_MARGIN')}
    for index,(gid,g) in enumerate(hold.groupby('GAME_ID')):
        if len(g)!=2:continue
        h=g[g.HOME_AWAY.eq('H')].iloc[0];a=g[g.HOME_AWAY.eq('A')].iloc[0]
        z=simulate_two_sides(h,a,models,n=n,seed=seed+index)
        for r,side in ((h,'home'),(a,'away')):
            for k in LABELS:
                truth=float(r['ACTUAL_'+k]);arr=np.asarray(z[side][k]);
                records[k].append((truth,arr.quantile(.05) if isinstance(arr,pd.Series) else np.quantile(arr,.05),np.quantile(arr,.95),np.mean(arr)))
        for k,arr,truth in (
            ('GAME_TOTAL',z['home']['PTS']+z['away']['PTS'],float(h.ACTUAL_PTS+a.ACTUAL_PTS)),
            ('GAME_MARGIN',z['home']['PTS']-z['away']['PTS'],float(h.ACTUAL_PTS-a.ACTUAL_PTS))):
            records[k].append((truth,np.quantile(arr,.05),np.quantile(arr,.95),np.mean(arr)))
    result={'status':'SHADOW_PROBABILITY_CALIBRATION_UNVERIFIED',
            'data_definition':'official_inclusive_v2','nominal_coverage':.90,
            'draws_per_game':n,'multiple_prior_analyses_on_this_holdout':True,
            'production_pricing_gate':False,'metrics':{}}
    for k,v in records.items():
        mat=np.asarray(v,dtype=float)
        ok=(mat[:,0]>=mat[:,1])&(mat[:,0]<=mat[:,2])
        N=len(ok);coverage=float(ok.mean())
        tol=float(1.96*np.sqrt(.9*.1/N))
        result['metrics'][k]={'n':N,'observed_90pct_coverage':round(coverage,4),
            'binomial_sampling_95pct_lower':round(.9-tol,4),
            'binomial_sampling_95pct_upper':round(.9+tol,4),
            'nominal_interval_coverage_flag':bool(abs(coverage-.9)>tol),
            'mean_error':round(float((mat[:,3]-mat[:,0]).mean()),3)}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('data/euroleague/exports'))
    ap.add_argument('--shadow',type=Path,default=Path('data/euroleague'))
    ap.add_argument('--draws',type=int,default=1500)
    z=ap.parse_args()
    d=verified_games(pd.read_csv(z.data/'team_games.csv'),pd.read_csv(z.data/'player_games.csv'))
    v06=json.loads((z.shadow/'shadow_v06/structural_v06_report.json').read_text())
    out=run(d,v06,n=z.draws)
    path=z.shadow/'shadow_v07/distribution_audit_v07.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('Uncalibrated distribution audit stored at',path)
    print('Coverage flags:',[k for k,v in out['metrics'].items() if v['nominal_interval_coverage_flag']])
if __name__=='__main__':main()
