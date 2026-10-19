"""EUROLEAGUE V1.0 RESEARCH CHALLENGER.
Historical seasons fit league-wide coefficients ONLY. Every live team/opponent/league
input is built from the requested SEASON prior to forecast UTC calendar DAY.
No previous-season team averages, carryover features or future match data.
Not production odds: variance, PBP, lineup-state remain unvalidated.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from core.structural_v06 import (FACTOR_NAMES,EPS,BOUNDS,project_two_sides,verified_games,event_rate_rows)

FACTORS=('PACE',)+FACTOR_NAMES
PRIOR_GRID=(0.,2.,4.,8.,16.)
ALPHA_GRID=(1.,10.,100.)
FEATURE_MODES=('off','off_def','off_def_home')


def _sum_ratio(hist,k):
    if not hist:return (0.,0.)
    return (sum(float(x['N_'+k]) for x in hist),sum(float(x['E_'+k]) for x in hist))


def _make_row(s,t,o,side,base,off,allowed,nteam,nopp,prior,game_id,day):
    out={'SEASON':int(s),'TEAM_ID':str(t),'OPP_ID':str(o),'HOME_AWAY':side,
         'HOME':float(side=='H'),'GAME_ID':str(game_id),'DAY':day,
         'CURRENT_TEAM_GAMES':nteam,'CURRENT_OPP_ALLOWED_GAMES':nopp}
    for k in FACTORS:
        b=base[k]; avg_expo=b[1]/max(base['_rows'],1)
        bval=b[0]/max(b[1],EPS)
        for label,h in [('OFF',off),('DEF',allowed)]:
            nn,ee=_sum_ratio(h,k)
            pe=prior*avg_expo
            out[k+'_'+label]=(nn+bval*pe)/max(ee+pe,EPS)
        out[k+'_BASE']=bval
    return out


def build_current_only_snapshots(events, *,prior_games,season_filter=None,min_team_games=2,min_league_rows=40):
    """Pre-game row embeddings. Strict same-season histories and full-date embargo."""
    if events.empty:return pd.DataFrame()
    ev=events.copy();ev['DAY']=ev.GAME_DATE.dt.floor('D')
    results=[]
    for season, season_data in ev.groupby('SEASON',sort=True):
        if season_filter is not None and int(season) not in season_filter:continue
        league=[];own=defaultdict(list);allowed=defaultdict(list)
        for day,grp in season_data.groupby('DAY',sort=True):
            if len(league)>=min_league_rows:
                base={k:_sum_ratio(league,k) for k in FACTORS};base['_rows']=len(league)
                for r in grp.to_dict('records'):
                    t=str(r['TEAM_ID']);o=str(r['OPP_ID']); ht=own[t];od=allowed[o]
                    if min(len(ht),len(od))<min_team_games:continue
                    z=_make_row(season,t,o,r['HOME_AWAY'],base,ht,od,len(ht),len(od),prior_games,r['GAME_ID'],day)
                    z['GAME_DATE']=r['GAME_DATE']
                    for k in FACTORS:
                        z[k+'_ACTUAL_RATE']=float(r['N_'+k])/max(float(r['E_'+k]),EPS)
                        z[k+'_ACTUAL_EXPOSURE']=float(r['E_'+k])
                    for field in ('PTS','FG2A','FG3A','FG2M','FG3M','FTA','FTM',
                                  'MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK'):
                        z['ACTUAL_'+field]=float(r[field])
                    z['ACTUAL_POSS']=float(r['SHARED_PACE_LABEL'])
                    results.append(z)
            # after all predictions for the UTC calendar day
            for r in grp.to_dict('records'):
                t=str(r['TEAM_ID']);o=str(r['OPP_ID'])
                own[t].append(r);allowed[o].append(r);league.append(r)
    return pd.DataFrame(results)


def current_only_upcoming(events,*,season,home_id,away_id,asof_date,prior_games=4.,min_games=2):
    """Strictly the current season (< day). Changing any other season cannot affect forecasts."""
    cutoff=pd.Timestamp(asof_date)
    cutoff=cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    cutoff=cutoff.floor('D')
    if home_id==away_id:raise ValueError('Home and away identical')
    current=events[(events.SEASON.astype(int)==int(season)) &
                   (events.GAME_DATE.dt.floor('D')<cutoff)].sort_values('GAME_DATE').copy()
    if current.empty:raise ValueError('No current-season data before as-of')
    if int(current.GAME_ID.nunique())<20:raise ValueError('Not enough league games CURRENT season')
    league=current.to_dict('records')
    base={k:_sum_ratio(league,k) for k in FACTORS};base['_rows']=len(league)
    def make(t,o,side):
        own=current[current.TEAM_ID.astype(str)==str(t)].to_dict('records')
        allowed=current[current.OPP_ID.astype(str)==str(o)].to_dict('records')
        if min(len(own),len(allowed))<min_games:raise ValueError('Too few current-season team/opponent games')
        return pd.Series(_make_row(season,t,o,side,base,own,allowed,len(own),len(allowed),prior_games,
                                   'FUTURE_CURRENT_ONLY',cutoff))
    return make(home_id,away_id,'H'),make(away_id,home_id,'A')


def _xy(df,k,mode):
    b=np.maximum(df[k+'_BASE'].to_numpy(float),EPS)
    x=np.column_stack([np.log(np.maximum(df[k+'_OFF'].to_numpy(float),EPS)/b),
                       np.log(np.maximum(df[k+'_DEF'].to_numpy(float),EPS)/b),
                       df.HOME.to_numpy(float)])
    inds={'off':[0],'off_def':[0,1],'off_def_home':[0,1,2]}[mode]
    y=np.log(np.maximum(df[k+'_ACTUAL_RATE'].to_numpy(float),EPS)/b).clip(-1.5,1.5)
    return x[:,inds],y,inds


def _fit(train,k,mode,alpha):
    x,y,inds=_xy(train,k,mode)
    w=np.clip(train[k+'_ACTUAL_EXPOSURE'].to_numpy(float),0.1,300.)
    w=w/max(w.mean(),EPS)
    m=Ridge(alpha=alpha).fit(x,y,sample_weight=w)
    co=np.zeros(3)
    for idx,c in zip(inds,m.coef_):co[idx]=c
    return {'selected':mode,'coefficients':co.tolist(),'intercept':float(m.intercept_),
            'alpha':alpha,'train_rows':int(len(train))}


def _pred_rate(df,k,m):
    b=df[k+'_BASE'].to_numpy(float)
    if m.get('selected')=='league':return b
    x,y,inds=_xy(df,k,m['selected'])
    beta=np.asarray(m['coefficients'])[inds]
    z=np.clip(m['intercept']+x@beta,-0.8,0.8)
    lo,hi=BOUNDS.get(k,(45.,95.))
    return np.clip(b*np.exp(z),lo,hi)


def train_historical_only(events):
    """Fit coefficients using <=2025 SEASONS only. Hyperparams tuned on 2024,
    2025 checked separately, and all historical seasons refitted for live 2026.
    Tuning is rate-space; final product requires structural-count validation.
    """
    all_rows={p:build_current_only_snapshots(events,prior_games=p,
                                    season_filter={2023,2024,2025}) for p in PRIOR_GRID}
    for p,d in all_rows.items():
        if d.empty or set(d.SEASON.astype(int))!={2023,2024,2025}:
            raise ValueError(f'Incomplete historical snapshot for prior {p}')
    specs={};stats={}
    for k in FACTORS:
        scores=[]
        for prior,d in all_rows.items():
            a=d[d.SEASON.eq(2023)];b=d[d.SEASON.eq(2024)]
            if k=='PACE':a=a[a.HOME_AWAY.eq('H')];b=b[b.HOME_AWAY.eq('H')]
            for mode in FEATURE_MODES:
                for alpha in ALPHA_GRID:
                    m=_fit(a,k,mode,alpha)
                    pred=_pred_rate(b,k,m);actual=b[k+'_ACTUAL_RATE'].to_numpy(float)
                    # Use rate rather than realized exposures for model selection.
                    err=float(np.sqrt(np.mean((pred-actual)**2)))
                    scores.append((err,prior,mode,alpha))
            base=b[k+'_BASE'].to_numpy(float);actual=b[k+'_ACTUAL_RATE'].to_numpy(float)
            scores.append((float(np.sqrt(np.mean((base-actual)**2))),prior,'league',None))
        scores.sort(key=lambda x:x[0]);best=scores[0]
        prior,mode,alpha=best[1:]
        d=all_rows[prior]
        tr=d[d.SEASON.isin([2023,2024])];va=d[d.SEASON.eq(2025)]
        if k=='PACE':tr=tr[tr.HOME_AWAY.eq('H')];va=va[va.HOME_AWAY.eq('H')]
        m={'selected':'league','coefficients':[0.,0.,0.],'intercept':0.,'alpha':None} if mode=='league' else _fit(tr,k,mode,alpha)
        pred=_pred_rate(va,k,m); actual=va[k+'_ACTUAL_RATE'].to_numpy(float)
        rmse=float(np.sqrt(np.mean((pred-actual)**2)))
        base_rmse=float(np.sqrt(np.mean((va[k+'_BASE'].to_numpy(float)-actual)**2)))
        full=d[d.SEASON.isin([2023,2024,2025])]
        if k=='PACE':full=full[full.HOME_AWAY.eq('H')]
        final={'selected':'league','coefficients':[0.,0.,0.],'intercept':0.,'alpha':None} if mode=='league' else _fit(full,k,mode,alpha)
        specs[k]={'prior_games':prior,'model':final,'validation_2025_rate_rmse':rmse,'league_2025_rate_rmse':base_rmse,
                  'selected_2024_rmse':best[0], 'validation_rows':len(va)}
        stats[k]={'2025_rate_rmse':rmse,'league_rate_rmse':base_rmse,'better_than_league':rmse<base_rmse}
    return {'version':'1.0-current-only-historical-coefficients','status':'SHADOW_NOT_PRICING_READY',
            'market_definition':'official_inclusive_v2','training_seasons':[2023,2024,2025],
            'live_input_contract':'CURRENT_SEASON_ONLY_OFFENSE_ALLOWED_AND_LEAGUE',
            'factors':specs,'validation_2025':stats}


def current_only_project(events,config,*,season,home_id,away_id,asof_date):
    p={k:float(v['prior_games']) for k,v in config['factors'].items()}
    # Each factor can use a different prior; no prior-year team profile is admitted.
    chosen=[]
    for team,opp,side in ((home_id,away_id,'H'),(away_id,home_id,'A')):
        result=None
        for prior in sorted(set(p.values())):
            h,a=current_only_upcoming(events,season=season,home_id=home_id,away_id=away_id,
                                       asof_date=asof_date,prior_games=prior)
            row=h if side=='H' else a
            if result is None:result=row.copy()
            for k in FACTORS:
                if p[k]==prior:
                    for suf in ('BASE','OFF','DEF'):result[k+'_'+suf]=row[k+'_'+suf]
        chosen.append(result)
    models={k:v['model'] for k,v in config['factors'].items()}
    return project_two_sides(chosen[0],chosen[1],models),chosen,models
