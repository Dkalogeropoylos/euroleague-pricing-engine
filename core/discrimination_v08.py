"""Chronological team-state challenger; NO artificial range expansion.
Uses only information with DAY < forecast day. Competitor to v0.6 regression,
never adjusts slopes merely to make team prices visually different.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from core.structural_v06 import (FACTOR_NAMES, BOUNDS, EPS, _predict_factor,
                                  chronological_design, project_two_sides, simulate_two_sides,
                                  upcoming_pair, chronological_split)
TARGETS=("PACE","THREE_SHARE","TWO_PCT","THREE_PCT","FT_PCT","FTA_RATE","OREB_RATE","DREB_RATE")
FEATURES=("recent_off","season_off","prev_off","recent_def","season_def","prev_def","home")
NFEATURES=len(FEATURES)

def _rate(rows,k,base,prior_games=0.):
    if not rows:return float(base)
    n=sum(float(z[f"N_{k}"]) for z in rows)
    e=sum(float(z[f"E_{k}"]) for z in rows)
    # Exposure-based prior, not count-based, to handle pace, proportions, DREB.
    ave_exp=e/len(rows) if e>0 else 1.
    prior=prior_games*ave_exp
    return float((n+base*prior)/max(e+prior,1e-9))

def _features(cur_off, prev_off, cur_def, prev_def, baseline,k,is_home,prior_games):
    # Defensive histories are stored as the offensive rows of that club's opponents.
    groups=[cur_off[-5:],cur_off[-22:],prev_off[-22:],
            cur_def[-5:],cur_def[-22:],prev_def[-22:]]
    vals=[]
    for g in groups:
        reg=_rate(g,k,baseline,prior_games=prior_games)
        vals.append(float(np.clip(np.log(max(reg,EPS)/max(baseline,EPS)),-0.7,0.7)))
    vals.append(float(is_home))
    return vals

def add_state_features(events:pd.DataFrame, design:pd.DataFrame, *, prior_games=4.):
    """Restore team-state features with 24h embargo, plus predecessor season.
    Feature names are frozen after grouping a day. Played opponents for each
    team go into that team's defensive allowed-history, never own offense.
    """
    base=design.copy()
    history_off=defaultdict(list)
    history_def=defaultdict(list)
    ev=events.sort_values(['GAME_DATE','GAME_ID','TEAM_ID']).copy()
    ev['DAY']=ev.GAME_DATE.dt.floor('D')
    feat_map={}
    dmap={(str(z.GAME_ID),str(z.TEAM_ID)):z for _,z in base.iterrows()}
    for day,g in ev.groupby('DAY',sort=True):
        for z in g.to_dict('records'):
            key=(str(z['GAME_ID']),str(z['TEAM_ID']))
            if key not in dmap:continue
            b=dmap[key]
            s,t,o=int(z['SEASON']),str(z['TEAM_ID']),str(z['OPP_ID'])
            feat={}
            for k in TARGETS:
                x=_features(history_off[(s,t)],history_off[(s-1,t)],
                            history_def[(s,o)],history_def[(s-1,o)],
                            float(b[f'{k}_BASE']),k,
                            z['HOME_AWAY']=='H',prior_games)
                feat.update({f'V8_{k}_{nm}':v for nm,v in zip(FEATURES,x)})
            feat_map[key]=feat
        for z in g.to_dict('records'):
            history_off[(int(z['SEASON']),str(z['TEAM_ID']))].append(z)
            history_def[(int(z['SEASON']),str(z['OPP_ID']))].append(z)
    if len(feat_map)!=len(base):
        raise ValueError(f'v0.8 feature completeness {len(feat_map)} != {len(base)}')
    out=base.copy()
    for k in TARGETS:
        for nm in FEATURES:
            col=f'V8_{k}_{nm}'
            out[col]=[feat_map[(str(r.GAME_ID),str(r.TEAM_ID))][col] for _,r in base.iterrows()]
    return out

def upcoming_with_state(events, *, season, home_id,away_id,asof_date,prior_games=4.):
    h,a=upcoming_pair(events,season=season,home_id=home_id,away_id=away_id,asof_date=asof_date)
    cutoff=pd.Timestamp(asof_date)
    if cutoff.tzinfo is None:cutoff=cutoff.tz_localize('UTC')
    else:cutoff=cutoff.tz_convert('UTC')
    cutoff=cutoff.floor('D')
    past=events[events.GAME_DATE.dt.floor('D')<cutoff]
    def add(row):
        s=int(row.SEASON);t=str(row.TEAM_ID);o=str(row.OPP_ID)
        def src(ss,tt,mode):
            d=past[past.SEASON.astype(int).eq(ss)]
            d=d[d.TEAM_ID.astype(str).eq(tt) if mode=='off' else d.OPP_ID.astype(str).eq(tt)]
            return d.sort_values('GAME_DATE').to_dict('records')
        out=row.copy()
        for k in TARGETS:
            vals=_features(src(s,t,'off'),src(s-1,t,'off'),
                           src(s,o,'def'),src(s-1,o,'def'),
                           float(row[f'{k}_BASE']),k,row.HOME_AWAY=='H',prior_games)
            for nm,v in zip(FEATURES,vals):out[f'V8_{k}_{nm}']=v
        return out
    return add(h),add(a)

def _x(df,k,selected):
    names={
        'recent':['recent_off','recent_def','home'],
        'season':['season_off','season_def','home'],
        'carryover':['season_off','season_def','prev_off','prev_def','home'],
        'multiscale':list(FEATURES)
    }[selected]
    return df[[f'V8_{k}_{nm}' for nm in names]].to_numpy(float),names

def _candidate_pred(df,k,candidate):
    if candidate['kind']=='v06':
        m=candidate['v06_model']
        return np.array([_predict_factor(r,k,m) for _,r in df.iterrows()])
    if candidate['kind']=='league':
        return df[f'{k}_BASE'].to_numpy(float)
    x,_=_x(df,k,candidate['kind'])
    co=np.array(candidate['coefficients'])
    yhat=candidate['intercept']+x@co
    base=df[f'{k}_BASE'].to_numpy(float)
    return np.clip(base*np.exp(np.clip(yhat,-0.5,0.5)),*BOUNDS.get(k,(45.,95.)))

def _fit(train,k,kind,alpha):
    x,names=_x(train,k,kind)
    base=train[f'{k}_BASE'].to_numpy(float)
    yt=train[f'{k}_TARGET'].to_numpy(float)
    y=np.log(np.clip(yt,EPS,None)/np.maximum(base,EPS))
    y=np.clip(y,-0.85,0.85)
    den=train[f'{k}_DEN'].to_numpy(float)
    w=np.clip(den/np.maximum(np.mean(den),1e-5),0.1,5.)
    model=Ridge(alpha=alpha).fit(x,y,sample_weight=w)
    return {'kind':kind,'names':names,'coefficients':model.coef_.tolist(),'intercept':float(model.intercept_),
            'alpha':alpha,'n':len(train)}

def _loss(df,k,pred):
    # Opportunity-weighted count error. Realized exposure used ONLY as the
    # scoring label, never as a feature to forecast a future game.
    exposure=df[f'{k}_DEN'].to_numpy(float)
    actual=df[f'{k}_TARGET'].to_numpy(float)*exposure
    return float(np.mean((actual-pred*exposure)**2))

def _cv_folds(data,train_ids):
    d=data[data.GAME_ID.isin(train_ids)]
    days=sorted(d.DAY.unique())
    # two growing-window folds, one model-fitting interval per fold; no future rows
    return [(d[d.DAY<days[int(len(days)*q)]],
             d[(d.DAY>=days[int(len(days)*q)]) & (d.DAY<(days[int(len(days)*r)] if r<1. else days[-1]+pd.Timedelta(days=1)))])
            for q,r in [(0.50,0.72),(0.72,1.0)]]

def fit_challenger(data,base_models,train_ids,*,prior_games=4.):
    folds=_cv_folds(data,train_ids)
    all_train=data[data.GAME_ID.isin(train_ids)]
    config={}
    for k in TARGETS:
        candidates=[('league',None),('v06',None)]
        candidates += [(mode,alpha) for mode in ('recent','season','carryover','multiscale') for alpha in (0.5,5.,25.)]
        rows=[]
        for mode,alpha in candidates:
            scores=[]
            for tr,va in folds:
                valid=va[va.HOME_AWAY.eq('H')] if k=='PACE' else va
                train=tr[tr.HOME_AWAY.eq('H')] if k=='PACE' else tr
                if len(train)<40 or len(valid)<20:continue
                if mode=='league':
                    spec={'kind':'league'}
                elif mode=='v06':
                    spec={'kind':'v06','v06_model':base_models[k]}
                else:
                    spec=_fit(train,k,mode,alpha)
                scores.append(_loss(valid,k,_candidate_pred(valid,k,spec)))
            if scores:rows.append((sum(scores)/len(scores),mode,alpha,scores))
        rows.sort(key=lambda t:t[0])
        best=rows[0]
        kind,alpha=best[1],best[2]
        if kind in ('v06','league'): final={'kind':kind}
        else:
            tr=all_train[all_train.HOME_AWAY.eq('H')] if k=='PACE' else all_train
            final=_fit(tr,k,kind,alpha)
        config[k]={**final,'inner_cv_rmse':float(np.sqrt(best[0])),
                   'candidates':[{'kind':r[1],'alpha':r[2], 'rmse':round(np.sqrt(r[0]),4)}
                                  for r in rows[:10]]}
    return {'status':'SHADOW_NOT_PRICING_READY','market_definition':'official_inclusive_v2',
            'model_version':'0.8-team-state-challenger','prior_games':prior_games,
            'targets':config,'limitations':['No injuries or lineup changes','No PBP rebound opportunities',
            'Repeated historical samples are NOT fresh independent evidence','Probability distributions remain uncalibrated']}

def apply_challenger(h,a,base_models,challenger):
    m=copy.deepcopy(base_models)
    h=h.copy();a=a.copy()
    for k,c in challenger['targets'].items():
        if c['kind']=='v06':continue
        for row in (h,a):
            new=float(_candidate_pred(pd.DataFrame([row]),k,
                              c if c['kind']!='league' else {'kind':'league'})[0])
            row[f'{k}_BASE']=new
        m[k]={'selected':'league'}
    return h,a,m

def calculate(h,a,base_models,challenger,*,draws=0,seed=2026):
    hh,aa,mm=apply_challenger(h,a,base_models,challenger)
    return (project_two_sides(hh,aa,mm),
            simulate_two_sides(hh,aa,mm,n=draws,seed=seed) if draws else None)

def compare_holdout(design,baseline_model,challenger,test_ids):
    frame=design[design.GAME_ID.isin(test_ids)]
    tracked=('POSS','PTS','FG2A','FG3A','FTA','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST')
    val={k:{'actual':[],'v06':[],'v08':[]} for k in tracked}
    for gid,g in frame.groupby('GAME_ID',sort=False):
        h=g[g.HOME_AWAY.eq('H')].iloc[0];a=g[g.HOME_AWAY.eq('A')].iloc[0]
        old=project_two_sides(h,a,baseline_model)
        new,_=calculate(h,a,baseline_model,challenger)
        for row,side in ((h,'home'),(a,'away')):
            for k in tracked:
                val[k]['actual'].append(float(row.PACE_TARGET if k=='POSS' else row['ACTUAL_'+k]))
                val[k]['v06'].append(float(old['pace'] if k=='POSS' else old[side][k]))
                val[k]['v08'].append(float(new['pace'] if k=='POSS' else new[side][k]))
    report={}
    for k,v in val.items():
        act=np.array(v['actual']);old=np.array(v['v06']);new=np.array(v['v08'])
        rmse=lambda p:float(np.sqrt(np.mean((p-act)**2)))
        report[k]={'rmse_v06':round(rmse(old),4),'rmse_v08':round(rmse(new),4),
                   'change_pct':round(100*(1-rmse(new)/rmse(old)),2),
                   'bias_v08':round(float(np.mean(new-act)),4),
                   'std_v06':round(float(np.std(old)),4),
                   'std_v08':round(float(np.std(new)),4),
                   'std_actual':round(float(np.std(act)),4)}
    return report

def refit_challenger(design, selected):
    """Refit selected ONLY on available full past after historical score is sealed."""
    out=copy.deepcopy(selected)
    for k,v in out['targets'].items():
        if v['kind'] in ('league','v06'):
            continue
        d=design.loc[design.HOME_AWAY.eq('H')] if k=='PACE' else design
        refreshed=_fit(d,k,v['kind'],v['alpha'])
        out['targets'][k]={**v,'coefficients':refreshed['coefficients'],
                           'intercept':refreshed['intercept'],
                           'n':refreshed['n']}
    return out
