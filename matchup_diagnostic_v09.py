"""Display independent SHADOW team-count predictions for selected fixtures as of day."""
import argparse,json,sys
from pathlib import Path
import pandas as pd
import numpy as np
from calibrate_shrinkage_v09 import prepare_games, build_snapshots,feature_array,fit_predict,MARKETS


def run(team_csv,report_json,home,away,date):
    raw=prepare_games(team_csv)
    d=pd.Timestamp(date)
    d=d.tz_localize('UTC') if d.tzinfo is None else d.tz_convert('UTC')
    if (raw.GAME_DATE>=d).any():
        # Forecast historical snapshot only if no game from that day or after
        raw=raw[raw.DAY<d].copy()
    future=[]
    for tid,opp,side in [(home,away,'H'),(away,home,'A')]:
        src=raw[raw.TEAM_ID.eq(tid)].iloc[-1].copy()
        src['GAME_ID']='SHADOW_FUTURE';src['GAME_DATE']=d;src['DAY']=d
        src['TEAM_ID']=tid;src['OPP_ID']=opp;src['HOME_AWAY']=side;src['SEASON']=2026
        for m in MARKETS:
            src[m]=0
        src['POSS_RAW']=0.;src['OT_FLAG']=False
        future.append(src)
    snap=build_snapshots(pd.concat([raw,pd.DataFrame(future)],ignore_index=True).sort_values('DAY'))
    info=json.load(open(report_json))
    if info.get('market_definition')!='official_inclusive_v2' or info.get('status')!='SHADOW_NOT_PRICING_READY':
        raise ValueError('Shadow calibration report schema/status mismatch')
    cfg=info['markets']
    pred={}
    for market in MARKETS:
        rows=snap[snap.HOME_AWAY.eq('H')].copy() if market=='POSS' else snap.copy()
        rows=rows.reset_index(drop=True)
        train=np.flatnonzero(rows.DAY.lt(d).to_numpy())
        valid=np.flatnonzero(rows.GAME_ID.eq('SHADOW_FUTURE').to_numpy())
        model=cfg[market]['selected']
        x=feature_array(rows,market,model['prior_games'],model['previous_season_weight_in_prior'],model['feature_mode'])
        estimates,_,_=fit_predict(x,rows['B_'+market].to_numpy(float),rows['Y_'+market].to_numpy(float),train,valid,model['ridge_alpha'],market)
        for ix,est in zip(valid,estimates):
            key=rows.iloc[ix].TEAM_ID
            pred.setdefault(key,{})[market]=round(float(est),2)
    return pred

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--report',required=True)
    p.add_argument('--home',required=True);p.add_argument('--away',required=True)
    p.add_argument('--date',default='2026-10-13');a=p.parse_args()
    print(json.dumps(run(a.csv,a.report,a.home,a.away,a.date),indent=2))
