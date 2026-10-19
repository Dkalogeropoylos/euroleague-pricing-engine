"""Research-only, prequential ridge shrinkage audit for EuroLeague team counts.

Does not update live bookmaker prices, the production app, or Monte Carlo distributions.
All predictors are frozen before the UTC calendar day of the target game.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

MARKETS=('POSS','FG3A','STL','TO','OREB','DREB','REB','PTS')
FEATURES=('off','def','recent_off','recent_def','home')
PRIOR_GRID=(2.,4.,8.,16.,32.)
PREV_GRID=(0.,0.5,1.)
ALPHA_GRID=(1.,10.,100.)
MODE_GRID=('team','opp','both','multiscale')
FOLDS=(("2025-01-01","2025-06-01"),("2026-01-01","2026-06-01"))


def prepare_games(csv_path):
    x=pd.read_csv(csv_path,low_memory=False)
    x['GAME_DATE']=pd.to_datetime(x.GAME_DATE,utc=True)
    if 'MARKET_SCHEMA_VERSION' in x.columns:
        if not x.MARKET_SCHEMA_VERSION.eq('official_inclusive_v2').all():
            raise ValueError('Settlement schema mismatch: expected official_inclusive_v2')
        for field in ('OREB','DREB','REB','TO'):
            if not np.allclose(x['MARKET_'+field],x[field],equal_nan=False):
                raise ValueError('Official-inclusive target mismatch for '+field)
    if not np.allclose(x.REB, x.OREB + x.DREB):
        raise ValueError('OREB+DREB must equal total team rebounds')
    x=x[~x.OT_FLAG.fillna(False).astype(bool)].copy()
    if not (x.groupby('GAME_ID').size()==2).all():raise ValueError('Invalid matched games')
    x['POSS']=x.groupby('GAME_ID').POSS_RAW.transform('mean')
    x['DAY']=x.GAME_DATE.dt.floor('D')
    return x.sort_values(['DAY','GAME_ID','TEAM_ID']).reset_index(drop=True)


def build_snapshots(raw):
    """Full historical raw count sums held in histories; no game-day leakage."""
    hist_off=defaultdict(list)
    hist_def=defaultdict(list)
    hist_lg=deque()
    out=[]
    for day,g in raw.groupby('DAY',sort=True):
        while hist_lg and day-hist_lg[0]['DAY']>pd.Timedelta(days=550):hist_lg.popleft()
        lbase={m:float(np.mean([z[m] for z in hist_lg])) if hist_lg else np.nan for m in MARKETS}
        if len(hist_lg)>=70:
            for row in g.to_dict('records'):
                season=int(row['SEASON']); t=str(row['TEAM_ID']);opp=str(row['OPP_ID'])
                # Team offensive raw outputs, opponent defensive allowed raw outputs.
                feat={'GAME_ID':str(row['GAME_ID']),'DAY':day,'SEASON':season,
                      'TEAM_ID':t,'OPP_ID':opp,'HOME_AWAY':str(row['HOME_AWAY']),
                      'HOME':int(row['HOME_AWAY']=='H')}
                for m in MARKETS:
                    feat['Y_'+m]=float(row[m]);feat['B_'+m]=lbase[m]
                    for prefix,series in [('off',hist_off[(season,t)]),('def',hist_def[(season,opp)]),
                                         ('prev_off',hist_off[(season-1,t)]),('prev_def',hist_def[(season-1,opp)])]:
                        seq=series[-22:] if not prefix.startswith('prev') else series[-38:]
                        feat[f'{prefix}_n']=len(seq)
                        feat[f'{prefix}_{m}']=float(np.mean([z[m] for z in seq])) if seq else np.nan
                    for prefix,series in [('recent_off',hist_off[(season,t)]),('recent_def',hist_def[(season,opp)])]:
                        seq=series[-5:]
                        feat[f'{prefix}_n']=len(seq)
                        feat[f'{prefix}_{m}']=float(np.mean([z[m] for z in seq])) if seq else np.nan
                out.append(feat)
        # Commit all of a day AFTER generating the day's previews.
        for z in g.to_dict('records'):
            hist_off[(int(z['SEASON']),str(z['TEAM_ID']))].append(z)
            hist_def[(int(z['SEASON']),str(z['OPP_ID']))].append(z)
            hist_lg.append(z)
    return pd.DataFrame(out)


def feature_array(snap, market, prior, prev_weight, mode):
    base=snap['B_'+market].to_numpy(float)
    n=len(snap)
    val={}
    for sig in ('off','def','recent_off','recent_def'):
        pfx='off' if 'off' in sig else 'def'
        current=snap[f'{sig}_{market}'].to_numpy(float)
        count=snap[f'{sig}_n'].to_numpy(float)
        prev=snap[f'prev_{pfx}_{market}'].to_numpy(float)
        prev=np.where(np.isfinite(prev),prev,base)
        pa=(1-prev_weight)*base+prev_weight*prev
        # Independent n-game effective prior; no arbitrary 3PA / steals overrides.
        adjusted=np.where(count>0,(np.nan_to_num(current)*count+prior*pa)/(count+prior),pa)
        val[sig]=np.log(np.maximum(adjusted,0.1)/np.maximum(base,0.1))
    cols={'team':('off','home'),'opp':('def','home'),
          'both':('off','def','home'),'multiscale':FEATURES}[mode]
    val['home']=snap.HOME.to_numpy(float)
    return np.column_stack([val[c] for c in cols])


def fit_predict(x,base,actual,train,valid,alpha,market):
    # Positive-shift log target and fixed historical offset.
    y=np.log((np.maximum(0,actual[train])+.5)/(base[train]+.5))
    model=Ridge(alpha=alpha).fit(x[train],y)
    trpred=np.maximum(0.,(base[train]+.5)*np.exp(np.clip(model.predict(x[train]),-.8,.8))-.5)
    # Multiplicative bias calibration is fitted only from the training period.
    correction=float(np.sum(actual[train])/max(np.sum(trpred),1e-7))
    correction=float(np.clip(correction,.75,1.25))
    pred=np.maximum(0.,((base[valid]+.5)*np.exp(np.clip(model.predict(x[valid]),-.8,.8))-.5)*correction)
    if market=='POSS':pred=np.clip(pred,53,105)
    return pred,model,correction


def loss_rmse(actual,pred):return float(np.sqrt(np.mean((actual-pred)**2)))


def calibrate(snap):
    result={}
    # score pace once per game, on HOME records only; other markets both teams.
    for metric in MARKETS:
        sub=snap[snap.HOME_AWAY.eq('H')].reset_index(drop=True) if metric=='POSS' else snap.reset_index(drop=True)
        base=sub['B_'+metric].to_numpy(float);actual=sub['Y_'+metric].to_numpy(float)
        d=sub.DAY.dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')
        folds=[]
        for begin,end in FOLDS:
            tr=np.where(d<np.datetime64(begin))[0]
            va=np.where((d>=np.datetime64(begin))&(d<np.datetime64(end)))[0]
            if len(tr)<100 or len(va)<50:raise ValueError((metric,begin,len(tr),len(va)))
            folds.append((tr,va,begin,end))
        watch=np.where(d>=np.datetime64('2026-09-01'))[0]
        pre_watch=np.where(d<np.datetime64('2026-09-01'))[0]
        rows=[]
        for prior in PRIOR_GRID:
            for prev in PREV_GRID:
                for mode in MODE_GRID:
                    x=feature_array(sub,metric,prior,prev,mode)
                    for alpha in ALPHA_GRID:
                        err=[];pred_bases=[]; actual_vals=[]
                        for train,valid,_,_ in folds:
                            yp,_,_=fit_predict(x,base,actual,train,valid,alpha,metric)
                            err.extend((actual[valid]-yp).tolist())
                            pred_bases.extend(base[valid].tolist())
                            actual_vals.extend(actual[valid].tolist())
                        score=float(np.sqrt(np.mean(np.square(err))))
                        rows.append((score,prior,prev,mode,alpha))
        rows.sort()
        best=rows[0]
        x=feature_array(sub,metric,best[1],best[2],best[3])
        pred_watch,mdl,corr=fit_predict(x,base,actual,pre_watch,watch,best[4],metric)
        watch_actual=actual[watch]; watch_base=base[watch]
        all_train_preds=np.zeros(0); all_y=np.zeros(0); all_b=np.zeros(0)
        for train,valid,_,_ in folds:
            yp,_,_=fit_predict(x,base,actual,train,valid,best[4],metric)
            all_train_preds=np.concatenate([all_train_preds,yp]);all_y=np.concatenate([all_y,actual[valid]])
            all_b=np.concatenate([all_b,base[valid]])
        # simple count/variance and fixed-year diagnostics, not causal effects
        result[metric]={
           'selected':{'prior_games':best[1],'previous_season_weight_in_prior':best[2],
                       'feature_mode':best[3],'ridge_alpha':best[4],
                       'trained_coefficients':mdl.coef_.round(6).tolist(),
                       'intercept':round(float(mdl.intercept_),6),'mean_bias_correction':round(corr,6)},
           'blocked_cv':{'n':len(all_y),'rmse_candidate':round(loss_rmse(all_y,all_train_preds),4),
                         'rmse_simple_prior':round(loss_rmse(all_y,all_b),4),
                         'mean_error':round(float(np.mean(all_train_preds-all_y)),4)},
           'watch_2026':{'n':len(watch),'rmse_candidate':round(loss_rmse(watch_actual,pred_watch),4),
                          'rmse_league':round(loss_rmse(watch_actual,watch_base),4),
                          'bias_candidate':round(float(np.mean(pred_watch-watch_actual)),4),
                          'prediction_std':round(float(np.std(pred_watch)),4),
                          'actual_std':round(float(np.std(watch_actual)),4)},
           'top_candidates':[{'cv_rmse':round(z[0],4),'prior':z[1],'prev_weight':z[2],
                              'mode':z[3],'alpha':z[4]} for z in rows[:5]]}
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('--team-csv',default='data/euroleague/exports/team_games.csv');a.add_argument('--out',default='data/euroleague/shadow_v09/calibration_v09_report.json')
    args=a.parse_args()
    raw=prepare_games(args.team_csv);snap=build_snapshots(raw)
    report={'study':'SHADOW_PARAMETER_CALIBRATION_NO_PRICE_DEPLOY',
       'model_version':'0.9-count-only-calibration-audit',
       'status':'SHADOW_NOT_PRICING_READY',
       'market_definition':'official_inclusive_v2',
       'dataset':{'games':int(raw.GAME_ID.nunique()),'snapshots':len(snap)},
       'validation':'Two blocked rolling-year folds with each model fitted strictly before its validation range. 2026 used as watch diagnostics (previously inspected, not pristine holdout).',
       'notes':['A dynamic team-count benchmark, not replacement for joint structural simulation.',
                'Pregame history prior to UTC calendar day; 550-day league history.',
                'All scored outputs are observed final counts, not realized exposure-adjusted rate targets.',
                'The winning hyperparameter search and its CV score are NOT unbiased post-selection estimates.',
                'Market settlement: official-inclusive team OREB, DREB, REB, TO.'],
       'markets':calibrate(snap)}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False))
    print('WROTE',args.out)
    for k,v in report['markets'].items():
       print(k,'CV',v['blocked_cv'],'watch',v['watch_2026'],'cfg',v['selected']['prior_games'],v['selected']['previous_season_weight_in_prior'],v['selected']['feature_mode'])
if __name__=='__main__':main()
