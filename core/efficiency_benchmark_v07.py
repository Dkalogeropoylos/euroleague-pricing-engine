"""Pregame, chronologically trained, independent total-points benchmark.

PPP from points / *shared box-score estimated* possessions (regulation only).
A paired game has one common pace label; a past game affects predictions only
on later DATES. Rate priors are learned by inner chronological validation,
not visually adjusted to the current matchup. This is an audit against v0.6,
not an automatic blend nor a replacement for a possession-structured model.
"""
from __future__ import annotations
from collections import defaultdict,deque
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PRIORS=(2.,4.,8.,16.)
ALPHAS=(2.,15.,80.)
EPS=1e-6


def _rates(off:list,defense:list,league:float,prior:float,league_pace:float):
    eq=prior*max(league_pace,1.)
    offp=(sum(p for p,_ in off)+league*eq)/(sum(pace for _,pace in off)+eq)
    dep=(sum(p for p,_ in defense)+league*eq)/(sum(pace for _,pace in defense)+eq)
    return offp,dep


def chrono_ppp_design(d:pd.DataFrame,prior_games:float,history_days=550)->pd.DataFrame:
    need=['GAME_ID','GAME_DATE','SEASON','TEAM_ID','OPP_ID','HOME_AWAY','PTS','OPP_PTS','SHARED_PACE_LABEL']
    if not set(need).issubset(d.columns):raise ValueError('Missing verified historical fields')
    if not len(d) or d.groupby('GAME_ID').size().ne(2).any():raise ValueError('Invalid game pairs')
    s=d.copy()
    s['DAY']=pd.to_datetime(s.GAME_DATE,utc=True).dt.floor('D')
    s=s.sort_values(['DAY','GAME_ID','TEAM_ID'])
    league=deque();off=defaultdict(list);allow=defaultdict(list)
    rows=[]
    for day,b in s.groupby('DAY',sort=True):
        cutoff=day-pd.Timedelta(days=history_days)
        while league and league[0][0]<cutoff:league.popleft()
        past_games=list(league)
        if len(past_games)>=50:
            overall_pts=sum(z[1] for z in past_games)
            overall_poss=sum(z[2] for z in past_games)
            baseline=overall_pts/max(overall_poss,EPS)
            avg_pace=overall_poss/len(past_games)
            for _,r in b.iterrows():
                season=int(r.SEASON);team=str(r.TEAM_ID);opp=str(r.OPP_ID)
                xo=[(p,pace) for date,p,pace in off[season,team] if date>=cutoff]
                xd=[(p,pace) for date,p,pace in allow[season,opp] if date>=cutoff]
                oo,dd=_rates(xo,xd,baseline,prior_games,avg_pace)
                rows.append({'GAME_ID':str(r.GAME_ID),'DAY':day,'SEASON':season,
                  'TEAM_ID':team,'OPP_ID':opp,'HOME':int(r.HOME_AWAY=='H'),
                  'HOME_AWAY':r.HOME_AWAY,'BASE':baseline,
                  'X_OFF':np.log(max(oo,EPS)/baseline),'X_DEF':np.log(max(dd,EPS)/baseline),
                  'TARGET':float(r.PTS)/float(r.SHARED_PACE_LABEL),
                  'ACTUAL_PTS':float(r.PTS),'ACTUAL_PACE':float(r.SHARED_PACE_LABEL),
                  'N_OFF':len(xo),'N_DEF':len(xd)})
        # Both teams on a calendar day are committed only AFTER feature freeze
        for _,r in b.iterrows():
            dt=day;season=int(r.SEASON);team=str(r.TEAM_ID);opp=str(r.OPP_ID)
            pace=float(r.SHARED_PACE_LABEL)
            league.append((dt,float(r.PTS),pace))
            off[season,team].append((dt,float(r.PTS),pace))
            allow[season,opp].append((dt,float(r.PTS),pace))
    if not rows: raise ValueError('Insufficient past games')
    return pd.DataFrame(rows).sort_values(['DAY','GAME_ID','TEAM_ID']).reset_index(drop=True)


def _pred(d,model):
    X=d[['X_OFF','X_DEF','HOME']].to_numpy(float)
    c=np.asarray(model['coefficients'],float)
    ppp=d.BASE.to_numpy(float)*np.exp(np.clip(X@c+model['intercept'],-.30,.30))
    return ppp


def fit_efficiency_audit(d:pd.DataFrame, holdout_fraction=.20)->dict:
    """Inner temporal validation selects prior and ridge penalty; one outer audit.

    Rerunning this repeatedly on the same holdout does not qualify as new OOS evidence.
    """
    cache={p:chrono_ppp_design(d,p) for p in PRIORS}
    base=cache[PRIORS[0]]
    unique_days=sorted(base.DAY.unique())
    cut=unique_days[int((1.-holdout_fraction)*len(unique_days))]
    td=base[base.DAY<cut];outer=base[base.DAY>=cut]
    inner_days=sorted(td.DAY.unique());innercut=inner_days[int(.75*len(inner_days))]
    results=[]
    for prior,features in cache.items():
        train=features[features.DAY<innercut]
        inner=features[(features.DAY>=innercut)&(features.DAY<cut)]
        if len(train)<350 or len(inner)<100:raise ValueError('Need more chronological data')
        for alpha in ALPHAS:
            X=train[['X_OFF','X_DEF','HOME']].to_numpy(float)
            y=np.log(np.maximum(train.TARGET.to_numpy(float),EPS)/train.BASE.to_numpy(float))
            m=Ridge(alpha=alpha,fit_intercept=True).fit(X,y)
            mdl={'coefficients':m.coef_.tolist(),'intercept':float(m.intercept_)}
            pred=_pred(inner,mdl)*inner.ACTUAL_PACE.to_numpy(float)
            actual=inner.ACTUAL_PTS.to_numpy(float)
            err=np.sqrt(np.mean((actual-pred)**2))
            results.append((float(err),prior,alpha))
    # Include fully static 'league PPP' benchmark as selectable, avoiding forced sophistication
    inner=base[(base.DAY>=innercut)&(base.DAY<cut)]
    baseline_err=float(np.sqrt(np.mean((inner.ACTUAL_PTS-inner.BASE*inner.ACTUAL_PACE)**2)))
    best=min(results)
    if baseline_err<=best[0]:
        prior=PRIORS[0];alpha=None;model={'coefficients':[0.,0.,0.],'intercept':0.};mode='league_baseline'
    else:
        _,prior,alpha=best;chosen=cache[prior];train=chosen[chosen.DAY<cut]
        X=train[['X_OFF','X_DEF','HOME']].to_numpy(float)
        y=np.log(np.maximum(train.TARGET.to_numpy(float),EPS)/train.BASE.to_numpy(float))
        m=Ridge(alpha=alpha).fit(X,y)
        model={'coefficients':m.coef_.tolist(),'intercept':float(m.intercept_)};mode='ridge_log_ppp'
    # NEXT-GAME deployment refit: preserve the selected specification; now use ALL
    # observed history (including outer holdout) only for FUTURE predictions.
    whole=cache[prior]
    if mode=='ridge_log_ppp':
        Xall=whole[['X_OFF','X_DEF','HOME']].to_numpy(float)
        yall=np.log(np.maximum(whole.TARGET.to_numpy(float),EPS)/whole.BASE.to_numpy(float))
        next_fit=Ridge(alpha=alpha).fit(Xall,yall)
        next_model={'coefficients':next_fit.coef_.tolist(),'intercept':float(next_fit.intercept_)}
    else:
        next_model={'coefficients':[0.,0.,0.],'intercept':0.}
    out=cache[prior];out=out[out.DAY>=cut]
    predicted=_pred(out,model)*out.ACTUAL_PACE.to_numpy(float)
    actual=out.ACTUAL_PTS.to_numpy(float)
    league_pred=out.BASE.to_numpy(float)*out.ACTUAL_PACE.to_numpy(float)
    rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    model_rmse=rmse(predicted,actual);league_rmse=rmse(league_pred,actual)
    # Independent TOTAL audit using observed pace only measures scoring rate correctness.
    by_game=out[['GAME_ID']].copy();by_game['act']=actual;by_game['pred']=predicted;by_game['base']=league_pred
    agg=by_game.groupby('GAME_ID')[['act','pred','base']].sum()
    return {'status':'SHADOW_BENCHMARK_ONLY','data_definition':'official_inclusive_v2',
       'selection':mode,'prior_games':prior,'alpha':alpha,'model':model,
       'next_game_model':next_model,'next_game_training_rows':int(len(whole)),
       'notes':'Outer scoring audit uses observed pace only to isolate PPP error. Never substitute observed pace in pregame projections.',
       'inner_baseline_rmse':baseline_err,'inner_best_candidate_rmse':best[0],
       'outer_team_ppp_model_rmse_pts':model_rmse,'outer_team_league_baseline_rmse_pts':league_rmse,
       'outer_total_ppp_model_rmse_pts':rmse(agg.pred,agg.act),'outer_total_league_rmse_pts':rmse(agg.base,agg.act),
       'outer_games':int(len(agg)),'outer_team_rows':len(out),
       'outer_cutoff_day_utc':str(cut),'future_validation_required':True}


def upcoming_efficiency_benchmark(d:pd.DataFrame,report:dict,season:int,home_id:str,away_id:str,asof_date,projected_possessions:float):
    if report.get('data_definition')!='official_inclusive_v2':raise ValueError('Wrong market definition')
    when=pd.Timestamp(asof_date)
    when=when.tz_localize('UTC') if when.tzinfo is None else when.tz_convert('UTC')
    snap=d.copy()
    snap['DAY']=pd.to_datetime(snap.GAME_DATE,utc=True).dt.floor('D')
    snap=snap[(snap.DAY<when)&(snap.DAY>=when-pd.Timedelta(days=550))]
    if snap.empty:raise ValueError('No historical games before as-of date')
    pp=snap.PTS.sum()/snap.SHARED_PACE_LABEL.sum();lgpace=float(snap.SHARED_PACE_LABEL.mean())
    def calc(t,opp,home):
        ts=snap[(snap.SEASON==season)&(snap.TEAM_ID==t)]
        allow=snap[(snap.SEASON==season)&(snap.OPP_ID==opp)]
        oo,dd=_rates(list(zip(ts.PTS,ts.SHARED_PACE_LABEL)),list(zip(allow.PTS,allow.SHARED_PACE_LABEL)),pp,report['prior_games'],lgpace)
        x=np.array([np.log(oo/pp),np.log(dd/pp),float(home)])
        m=report.get('next_game_model') or report['model'];ppp=pp*np.exp(np.clip(x@np.asarray(m['coefficients'])+m['intercept'],-.30,.30))
        return {'expected_pts':float(ppp*projected_possessions), 'expected_ppp':float(ppp),
                'offensive_ppp':float(oo),'opponent_allowed_ppp':float(dd),
                'team_games':int(len(ts)),'opp_def_games':int(len(allow))}
    h=calc(home_id,away_id,True);a=calc(away_id,home_id,False)
    return {'home':h,'away':a,'projected_total_points':h['expected_pts']+a['expected_pts'],
            'league_ppp':float(pp),'league_pace':lgpace,'shared_projected_possessions':projected_possessions,
            'status':'INDEPENDENT_SHADOW_BENCHMARK_NO_BLEND'}
