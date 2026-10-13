"""EuroLeague structural team-game shadow model, V0.5.

IMPORTANT: observational/experimental only. No claims of bookmaker pricing or
calibrated probabilities. Requires the verified players_only_v1 data contract.

- Same-date history embargo (no same-day game can leak into any prediction).
- Team/opponent empirical-Bayes rates trained on EuroLeague games only.
- Nested chronological coefficient selection; no tuning on outer holdout.
- Common possessions, linked attempts/makes, credited-only rebound/TO markets.
- Scenario/event-level Monte Carlo with conserved scoring & shot attempts.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from market_truth import attach_market_truth, MARKET_SCHEMA_VERSION

# Reboundable fraction of missed free throws is approximate absent free-throw PBP;
# it must not be interpreted as a learned fact or exact rebound opportunity count.
FT_REBOUND_PROXY = 0.50
PRIOR_TEAM_GAMES = 8.0
HISTORY_DAYS = 550
MAX_TEAM_HISTORY = 22
FACTOR_NAMES = (
    'TO_RATE','TO_CREDIT','FTA_RATE','THREE_SHARE',
    'TWO_PCT','THREE_PCT','FT_PCT','OREB_RATE','OREB_CREDIT',
    'DREB_RATE','AST_RATE','STL_RATE','BLK_RATE',
)
BOUNDS = {
    'TO_RATE':(.02,.38), 'TO_CREDIT':(.10,1.), 'FTA_RATE':(.02,.80),
    'THREE_SHARE':(.12,.76), 'TWO_PCT':(.24,.83), 'THREE_PCT':(.16,.62),
    'FT_PCT':(.38,.94), 'OREB_RATE':(.05,.70), 'OREB_CREDIT':(.40,1.),
    'DREB_RATE':(.35,1.), 'AST_RATE':(.18,.94), 'STL_RATE':(.10,.97),
    'BLK_RATE':(.01,.55),
}
EPS=1.e-9


def verified_games(team_raw:pd.DataFrame, players:pd.DataFrame|None=None, *, exclude_ot:bool=True)->pd.DataFrame:
    """Load either reconciled v0.4 export or validate legacy team & player CSV."""
    if players is not None:
        d=attach_market_truth(team_raw,players)
    else:
        d=team_raw.copy()
        if 'MARKET_SCHEMA_VERSION' not in d or not d['MARKET_SCHEMA_VERSION'].eq(MARKET_SCHEMA_VERSION).all():
            raise ValueError('Unverified settlement definition: attach player rows before training')
        for s in ('OREB','DREB','REB','TO'):
            if any(c not in d.columns for c in (f'MARKET_{s}',f'OFFICIAL_{s}',f'TEAM_ONLY_{s}')):
                raise ValueError(f'Missing settlement fields for {s}')
    if d.duplicated(['GAME_ID','TEAM_ID']).any() or not (d.groupby('GAME_ID').size()==2).all():
        raise ValueError('Unpaired or duplicate games')
    d['GAME_DATE']=pd.to_datetime(d.GAME_DATE,utc=True,errors='raise')
    if exclude_ot and 'OT_FLAG' in d.columns:
        d=d.loc[~d.GAME_ID.isin(d.loc[d.OT_FLAG.fillna(False).astype(bool),'GAME_ID'])].copy()
    # No per-team pace asymmetry used as the forecast target; both sides share a game proxy.
    d['SHARED_PACE_LABEL']=d.groupby('GAME_ID').POSS_RAW.transform('mean')
    if (d.SHARED_PACE_LABEL < 53).any() or (d.SHARED_PACE_LABEL>105).any():
        raise ValueError('Pace proxy out of plausible regulation-game bounds')
    d=d.sort_values(['GAME_DATE','GAME_ID','TEAM_ID']).reset_index(drop=True)
    if not np.array_equal(d.MARKET_REB.to_numpy(), (d.MARKET_OREB+d.MARKET_DREB).to_numpy()):
        raise ValueError('Player-credit rebounds identity violated')
    return d


def event_rate_rows(d:pd.DataFrame)->pd.DataFrame:
    """Game sides transformed to historical event / *opportunity* exposures."""
    o=d[['GAME_ID','TEAM_ID','FG2A','FG2M','FG3A','FG3M','FTA','FTM','OFFICIAL_OREB','MARKET_TO','SHARED_PACE_LABEL']].copy()
    o.columns=['GAME_ID','OPP_ID']+['OPP_'+k for k in o.columns[2:]]
    # Drop legacy opponent box-score fields; rebuild from the actual paired row.
    left=d.drop(columns=[c for c in d.columns if c.startswith('OPP_') and c not in ('OPP_ID','OPP_NAME')])
    x=left.merge(o,on=['GAME_ID','OPP_ID'],how='left',validate='one_to_one')
    if x.OPP_FG2A.isna().any(): raise ValueError('Unmatched opponents')
    miss=(x.FG2A-x.FG2M+x.FG3A-x.FG3M).astype(float)
    opp_miss=(x.OPP_FG2A-x.OPP_FG2M+x.OPP_FG3A-x.OPP_FG3M).astype(float)
    opp2miss=(x.OPP_FG2A-x.OPP_FG2M).astype(float)
    def factor(name,numer,denom):
        x[f'N_{name}']=pd.to_numeric(numer).to_numpy(float)
        x[f'E_{name}']=np.maximum(np.asarray(denom,dtype=float),EPS)
    factor('PACE',x.SHARED_PACE_LABEL,np.ones(len(x)))
    factor('TO_RATE',x.OFFICIAL_TO,x.SHARED_PACE_LABEL)
    factor('TO_CREDIT',x.MARKET_TO,x.OFFICIAL_TO)
    factor('FTA_RATE',x.FTA,x.SHARED_PACE_LABEL)
    factor('THREE_SHARE',x.FG3A,x.FG2A+x.FG3A)
    factor('TWO_PCT',x.FG2M,x.FG2A)
    factor('THREE_PCT',x.FG3M,x.FG3A)
    factor('FT_PCT',x.FTM,x.FTA)
    factor('OREB_RATE',x.OFFICIAL_OREB,miss)
    factor('OREB_CREDIT',x.MARKET_OREB,x.OFFICIAL_OREB)
    # Box-score data cannot identify reboundable FT misses. This proxy is approximate.
    defensive_chances=(opp_miss-x.OPP_OFFICIAL_OREB+FT_REBOUND_PROXY*(x.OPP_FTA-x.OPP_FTM))
    factor('DREB_RATE',x.MARKET_DREB,defensive_chances)
    factor('AST_RATE',x.AST,x.FG2M+x.FG3M)
    factor('STL_RATE',x.STL,x.OPP_MARKET_TO)
    factor('BLK_RATE',x.BLK,opp2miss)
    if any((x[f'N_{key}'] < 0).any() for key in FACTOR_NAMES): raise ValueError('Negative event rates')
    return x


def _sumratio(records:list[dict],k:str)->tuple[float,float]:
    return sum(float(v[f'N_{k}']) for v in records),sum(float(v[f'E_{k}']) for v in records)


def chronological_design(event_rows:pd.DataFrame, min_prior_games:int=35)->pd.DataFrame:
    """Frozen pregame features; day embargo applies to BOTH league and team history.

    Team offensive/allowed profiles are same-SEASON only, so transfers across
    seasons do not masquerade as persistent roster performance.
    """
    league_hist:deque[dict]=deque()
    team_off:dict[tuple[int,str],deque[dict]]=defaultdict(deque)
    team_def:dict[tuple[int,str],deque[dict]]=defaultdict(deque)
    rows=[]
    d=event_rows.copy()
    d['DAY']=d.GAME_DATE.dt.floor('D')
    for day,group in d.groupby('DAY',sort=True):
        # Preserve only prior dates, NEVER update between games on this same day.
        while league_hist and (day - league_hist[0]['DAY']).days > HISTORY_DAYS:
            league_hist.popleft()
        league=list(league_hist)
        if len(league)>=min_prior_games:
            league_opp_avg={k:_sumratio(league,k) for k in ('PACE',)+FACTOR_NAMES}
            for _,r in group.iterrows():
                season=int(r.SEASON);team=str(r.TEAM_ID);opp=str(r.OPP_ID)
                tlist=list(team_off[(season,team)])[-MAX_TEAM_HISTORY:]
                dlist=list(team_def[(season,opp)])[-MAX_TEAM_HISTORY:]
                row={'GAME_ID':str(r.GAME_ID),'DAY':day,'GAME_DATE':r.GAME_DATE,
                     'SEASON':season,'TEAM_ID':team,'OPP_ID':opp,
                     'HOME_AWAY':r.HOME_AWAY,'HOME':float(r.HOME_AWAY=='H'),
                     'HOME_TEAM_GAMES':len(tlist),'OPP_DEF_GAMES':len(dlist)}
                for k in ('PACE',)+FACTOR_NAMES:
                    ln,le=league_opp_avg[k]
                    base=ln/max(le,EPS)
                    # No fully invented startup league constants; base comes only from past games.
                    exposure_prior=PRIOR_TEAM_GAMES * le / max(len(league),1)
                    on,oe=_sumratio(tlist,k)
                    dn,de=_sumratio(dlist,'ALLOW_'+k)
                    off=(on+base*exposure_prior)/(oe+exposure_prior)
                    deff=(dn+base*exposure_prior)/(de+exposure_prior)
                    row[f'{k}_BASE']=base
                    row[f'{k}_OFF']=off
                    row[f'{k}_DEF']=deff
                    row[f'{k}_TARGET']=float(r[f'N_{k}'])/max(float(r[f'E_{k}']),EPS)
                    row[f'{k}_DEN']=float(r[f'E_{k}'])
                for k in ['PTS','FG2A','FG3A','FTA','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK','FG2M','FG3M','FTM']:
                    row['ACTUAL_'+k]=float(r[k])
                rows.append(row)
        # Commit to history only after all day's features have been frozen.
        for _,r in group.iterrows():
            obj=r.to_dict();obj['DAY']=day
            league_hist.append(obj)
            team_off[(int(r.SEASON),str(r.TEAM_ID))].append(obj)
            allow={'DAY':day}
            for k in ('PACE',)+FACTOR_NAMES:
                allow[f'N_ALLOW_{k}']=obj[f'N_{k}']
                allow[f'E_ALLOW_{k}']=obj[f'E_{k}']
            # In opponent-defense log, opponent's allowed distribution equals THIS team's output.
            team_def[(int(r.SEASON),str(r.OPP_ID))].append(allow)
    return pd.DataFrame(rows).sort_values(['DAY','GAME_ID','TEAM_ID']).reset_index(drop=True)


def _predict_factor(row:pd.Series,k:str,mdl:dict|None)->float:
    base=float(row[f'{k}_BASE'])
    if mdl is None or mdl.get('selected')=='league': return float(np.clip(base,*BOUNDS.get(k,(45.,95.))))
    def ratio(col): return np.log(max(float(row[f'{k}_{col}']),EPS)/max(base,EPS))
    vals=[ratio('OFF'),ratio('DEF'),float(row.HOME)]
    correction=float(mdl['intercept'])+float(np.dot(vals,mdl['coefficients']))
    if k=='PACE': return float(np.clip(base*np.exp(np.clip(correction,-.3,.3)),45,95))
    lo,hi=BOUNDS[k]
    return float(np.clip(base*np.exp(np.clip(correction,-0.8,0.8)),lo,hi))


def _fitted_arrays(df:pd.DataFrame,k:str,cols:list[int]):
    base=np.maximum(df[f'{k}_BASE'].to_numpy(float),EPS)
    a=np.log(np.maximum(df[f'{k}_OFF'].to_numpy(float),EPS)/base)
    b=np.log(np.maximum(df[f'{k}_DEF'].to_numpy(float),EPS)/base)
    x=np.column_stack([a,b,df.HOME.to_numpy(float)])
    y=np.log(np.maximum(df[f'{k}_TARGET'].to_numpy(float),EPS)/base)
    if k=='PACE': y=np.clip(y,-.55,.55)
    else:y=np.clip(y,-1.5,1.5)
    w=np.minimum(np.maximum(df[f'{k}_DEN'].to_numpy(float),1.),300.)
    return x[:,cols],y,w


def fit_factors(design:pd.DataFrame,train_ids:set[str],test_ids:set[str])->dict:
    """Inner chronological cutoff selects features/penalty; outer period untouched."""
    tr=design[design.GAME_ID.isin(train_ids)].copy()
    te=design[design.GAME_ID.isin(test_ids)].copy()
    days=sorted(tr.DAY.unique())
    if len(days)<25:raise ValueError('Not enough train dates')
    split=days[int(len(days)*.75)]
    it=tr.loc[tr.DAY<split];iv=tr.loc[tr.DAY>=split]
    fitted={}
    for k in ('PACE',)+FACTOR_NAMES:
        # Pace is one target per game; duplicate sides would double count games.
        a=tr.loc[tr.HOME_AWAY.eq('H')] if k=='PACE' else tr
        b=te.loc[te.HOME_AWAY.eq('H')] if k=='PACE' else te
        aa=it.loc[it.HOME_AWAY.eq('H')] if k=='PACE' else it
        bb=iv.loc[iv.HOME_AWAY.eq('H')] if k=='PACE' else iv
        if len(aa)<30 or len(bb)<15: raise ValueError(f'Too few historic rows for {k}')
        candidates=[('league',[],None),('off_only',[0],5.),('off_def',[0,1],5.),('off_def_home',[0,1,2],5.),
                    ('off_def_25',[0,1],25.),('off_def_100',[0,1],100.)]
        def pred_for(candidate,source,target):
            label,cols,alpha=candidate
            base=target[f'{k}_BASE'].to_numpy(float)
            if not cols: return base
            x,y,w=_fitted_arrays(source,k,cols)
            z,_,_=_fitted_arrays(target,k,cols)
            model=Ridge(alpha=alpha).fit(x,y,sample_weight=w/np.mean(w))
            raw=base*np.exp(np.clip(model.predict(z),-.8,.8))
            return np.clip(raw,*BOUNDS.get(k,(45.,95.)))
        def loss(pred,target):
            # Evaluate predicted ACTUAL event counts, not fitted rate residuals.
            y=target[f'{k}_TARGET'].to_numpy(float)*target[f'{k}_DEN'].to_numpy(float)
            expected=pred*target[f'{k}_DEN'].to_numpy(float)
            return float(np.mean((expected-y)**2))
        scores=[]
        for candidate in candidates:
            score=loss(pred_for(candidate,aa,bb),bb)
            scores.append((score,candidate))
        scores.sort(key=lambda x:x[0])
        best_score,best=scores[0]
        label,cols,alpha=best
        if cols:
            x,y,w=_fitted_arrays(a,k,cols)
            model=Ridge(alpha=alpha).fit(x,y,sample_weight=w/np.mean(w))
            coef=[0.,0.,0.]
            for c,v in zip(cols,model.coef_):coef[c]=float(v)
            intercept=float(model.intercept_)
        else:coef=[0.,0.,0.];intercept=0.
        fitted[k]={'selected':label,'coefficients':coef,'intercept':intercept,
                   'alpha':alpha,'inner_valid_rmse':np.sqrt(best_score),'train_rows':len(a),'holdout_rows':len(b)}
    return fitted


def _model_rates(row:pd.Series,models:dict|None)->dict:
    models=models or {}
    return {k:_predict_factor(row,k,models.get(k)) for k in ('PACE',)+FACTOR_NAMES}


def project_two_sides(home:pd.Series,away:pd.Series,models:dict|None)->dict:
    """Expected coherent 40-min event forecasts; no future outcome used."""
    if str(home.GAME_ID)!=str(away.GAME_ID) or home.HOME_AWAY!='H' or away.HOME_AWAY!='A':
        raise ValueError('Expected paired home/away frozen pregame observations')
    rh,ra=_model_rates(home,models),_model_rates(away,models)
    # A game has one set of possessions, not two inconsistent team possessions.
    pace=float((rh['PACE']+ra['PACE'])/2)
    def side(r:dict):
        to_off=pace*r['TO_RATE'];ftas=pace*r['FTA_RATE']
        base_fga=max(1.,pace-to_off-.44*ftas)
        s=r['THREE_SHARE'];missshare=(1-s)*(1-r['TWO_PCT'])+s*(1-r['THREE_PCT'])
        # Each official OREB creates another FG attempt on average (renewed chance).
        fga=base_fga/max(1.-r['OREB_RATE']*missshare,.25)
        pa3=fga*s;pa2=fga-pa3
        pm2=pa2*r['TWO_PCT'];pm3=pa3*r['THREE_PCT']
        fgmiss=fga-pm2-pm3
        oreb_off=fgmiss*r['OREB_RATE'];oreb=oreb_off*r['OREB_CREDIT']
        return {'POSS':pace,'OFFICIAL_TO':to_off,'MARKET_TO':to_off*r['TO_CREDIT'],
                'FTA':ftas,'FTM':ftas*r['FT_PCT'], 'FG2A':pa2,'FG3A':pa3,
                'FG2M':pm2,'FG3M':pm3,'FG_MISSES':fgmiss,'OFFICIAL_OREB':oreb_off,
                'MARKET_OREB':oreb,'AST':(pm2+pm3)*r['AST_RATE'],
                'FT_MISSES':ftas*(1-r['FT_PCT']), 'FG2_MISSES':pa2-pm2,
                'RATES':r}
    H=side(rh);A=side(ra)
    for mine,other in ((H,A),(A,H)):
        avail=max(0.,other['FG_MISSES']-other['OFFICIAL_OREB']+FT_REBOUND_PROXY*other['FT_MISSES'])
        mine['MARKET_DREB']=avail*mine['RATES']['DREB_RATE']
        mine['MARKET_REB']=mine['MARKET_OREB']+mine['MARKET_DREB']
        mine['STL']=other['MARKET_TO']*mine['RATES']['STL_RATE']
        mine['BLK']=other['FG2_MISSES']*mine['RATES']['BLK_RATE']
        mine['PTS']=2*mine['FG2M']+3*mine['FG3M']+mine['FTM']
        mine.pop('RATES')
    return {'home':H,'away':A,'pace':pace}


def simulate_two_sides(home:pd.Series,away:pd.Series,models:dict|None,*,n:int=10000,seed:int=2026)->dict:
    """Coherent event-level draws; BLOCKS conditional on misses, STL on credited TO.

    Not an approved betting probability generator: correlation/variance still
    need interval/backtest validation.  Retain full draw arrays in memory only.
    """
    if n<10 or n>100000:raise ValueError('n outside accepted range 10..100000')
    p=project_two_sides(home,away,models)
    rh,ra=_model_rates(home,models),_model_rates(away,models)
    rng=np.random.default_rng(seed)
    pace=np.clip(np.rint(rng.normal(p['pace'], 4.8, n)),55,99).astype(int)
    def simside(r):
        to=rng.binomial(pace,r['TO_RATE'])
        market_to=rng.binomial(to,r['TO_CREDIT'])
        fta=np.minimum(rng.poisson(pace*r['FTA_RATE']),pace)
        ftm=rng.binomial(fta,r['FT_PCT'])
        base=np.maximum(1,np.rint(pace-to-.44*fta).astype(int))
        attempts=base.copy()
        fg2a=np.zeros(n,int);fg3a=np.zeros(n,int)
        fg2m=np.zeros(n,int);fg3m=np.zeros(n,int)
        credited_oreb=np.zeros(n,int);official_oreb=np.zeros(n,int)
        # Every OREB creates one additional attempt; geometric tail up to 8 renewals.
        for i in range(9):
            pa3=rng.binomial(attempts,r['THREE_SHARE'])
            pa2=attempts-pa3
            pm3=rng.binomial(pa3,r['THREE_PCT']);pm2=rng.binomial(pa2,r['TWO_PCT'])
            fg3a+=pa3;fg2a+=pa2;fg3m+=pm3;fg2m+=pm2
            miss=attempts-pm2-pm3
            new_orb=(rng.binomial(miss,r['OREB_RATE']) if i<8 else np.zeros(n,int))
            credited_oreb+=rng.binomial(new_orb,r['OREB_CREDIT'])
            official_oreb+=new_orb
            attempts=new_orb
        ast=rng.binomial(fg2m+fg3m,r['AST_RATE'])
        return {'POSS':pace,'OFFICIAL_TO':to,'MARKET_TO':market_to,'FTA':fta,'FTM':ftm,
                'FG2A':fg2a,'FG3A':fg3a,'FG2M':fg2m,'FG3M':fg3m,'MARKET_OREB':credited_oreb,
                'OFFICIAL_OREB':official_oreb,'AST':ast,'PTS':2*fg2m+3*fg3m+ftm,
                'FG_MISSES':fg2a+fg3a-fg2m-fg3m,'FG2_MISSES':fg2a-fg2m,
                'FT_MISSES':fta-ftm,'BASE_FGA':base}
    H,A=simside(rh),simside(ra)
    for side,other,rate in ((H,A,rh),(A,H,ra)):
        ft_rebounds=rng.binomial(other['FT_MISSES'],FT_REBOUND_PROXY)
        available=np.maximum(0,other['FG_MISSES']-other['OFFICIAL_OREB']+ft_rebounds)
        side['MARKET_DREB']=rng.binomial(available,rate['DREB_RATE'])
        side['MARKET_REB']=side['MARKET_OREB']+side['MARKET_DREB']
        side['STL']=rng.binomial(other['MARKET_TO'],rate['STL_RATE'])
        side['BLK']=rng.binomial(other['FG2_MISSES'],rate['BLK_RATE'])
    return {'home':H,'away':A,'pace':pace,
            'limitations':'SHADOW: variance and correlations not yet approved for fair odds'}


def evaluate(design:pd.DataFrame,models:dict,holdout_ids:set[str])->dict:
    df=design[design.GAME_ID.isin(holdout_ids)]
    reports={k:{'actual':[],'pred':[],'base':[]} for k in
             ['POSS','PTS','FG2A','FG3A','FTA','FG2M','FG3M','FTM','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK']}
    for gid,g in df.groupby('GAME_ID',sort=False):
        if len(g)!=2:raise ValueError('Missing paired game in holdout')
        home=g[g.HOME_AWAY.eq('H')].iloc[0];away=g[g.HOME_AWAY.eq('A')].iloc[0]
        forecast=project_two_sides(home,away,models)
        baseline=project_two_sides(home,away,{})
        for actual,side in ((home,'home'),(away,'away')):
            for k in reports:
                observation=actual.PACE_TARGET if k=='POSS' else actual['ACTUAL_'+k]
                reports[k]['actual'].append(float(observation))
                reports[k]['pred'].append(float(forecast[side][k]))
                reports[k]['base'].append(float(baseline[side][k]))
    out={}
    for k,vals in reports.items():
        truth=np.asarray(vals['actual']);pred=np.asarray(vals['pred']);base=np.asarray(vals['base'])
        rmse=lambda x:float(np.sqrt(np.mean((truth-x)**2)))
        err=rmse(pred);baseline=rmse(base)
        out[k]={'rmse':round(err,4),'baseline_rmse':round(baseline,4),
                'improvement_pct':round(100*(1-err/max(baseline,EPS)),2),
                'bias':round(float(np.mean(pred-truth)),4),'n_team_games':int(len(truth)),
                'win':bool(err<baseline)}
    return out


def chronological_split(design:pd.DataFrame, holdout_share=.2)->tuple[set[str],set[str],str]:
    game_dates=design[['GAME_ID','DAY']].drop_duplicates().sort_values(['DAY','GAME_ID'])
    days=sorted(game_dates.DAY.unique());cutoff=days[int(len(days)*(1-holdout_share))]
    train=set(game_dates.loc[game_dates.DAY<cutoff,'GAME_ID'])
    test=set(game_dates.loc[game_dates.DAY>=cutoff,'GAME_ID'])
    if len(train)<150 or len(test)<35:raise ValueError('Too few chronological games for evaluation')
    return train,test,str(cutoff)


def refit_shadow_after_audit(design:pd.DataFrame,selected_models:dict)->dict:
    """Refit the already-selected specifications for NEXT games; holdout metrics unchanged.

    Uses no future outcomes relative to next projection, but does include prior
    holdout outcomes in the next-game refit. Never use these coefficients to
    regenerate the historical holdout score.
    """
    fresh={}
    for k,v in selected_models.items():
        choices={'league':[], 'off_only':[0], 'off_def':[0,1],
                 'off_def_home':[0,1,2], 'off_def_25':[0,1], 'off_def_100':[0,1]}
        cols=choices[v['selected']]
        if not cols:
            fresh[k]={**v}
            continue
        d=design.loc[design.HOME_AWAY.eq('H')] if k=='PACE' else design
        x,y,w=_fitted_arrays(d,k,cols)
        mdl=Ridge(alpha=float(v['alpha'])).fit(x,y,sample_weight=w/np.mean(w))
        coeffs=[0.,0.,0.]
        for c,p in zip(cols,mdl.coef_):coeffs[c]=float(p)
        fresh[k]={**v,'coefficients':coeffs,'intercept':float(mdl.intercept_)}
    return fresh


def upcoming_pair(event_rows:pd.DataFrame,*,season:int,home_id:str,away_id:str,asof_date)->tuple[pd.Series,pd.Series]:
    """Create genuine pregame snapshot for an as-yet-unplayed matchup.

    All source events are strictly BEFORE asof_date (UTC calendar day).
    Unknown/incorrect team IDs and impossible dates fail clearly.
    """
    cutoff=pd.Timestamp(asof_date)
    if cutoff.tzinfo is None:cutoff=cutoff.tz_localize('UTC')
    else:cutoff=cutoff.tz_convert('UTC')
    cutoff=cutoff.floor('D')
    if home_id==away_id:raise ValueError('Select different home and away teams')
    # Refit full-history coefficients are ONLY valid for dates after the last loaded game.
    # Historical what-if queries must use the original frozen walk-forward coefficients.
    last_observed=event_rows.GAME_DATE.dt.floor('D').max()
    if cutoff <= last_observed:
        raise ValueError('Shadow future-preview requires a date strictly after the latest loaded game; historical as-of backtest is separate')
    d=event_rows[event_rows.GAME_DATE.dt.floor('D')<cutoff].copy()
    league=d.loc[d.GAME_DATE.dt.floor('D') >= cutoff-pd.Timedelta(days=HISTORY_DAYS)]
    if len(league)<35:raise ValueError('Not enough strictly historical league rows')
    current=d[d.SEASON.astype(int)==int(season)]
    seen=set(current.TEAM_ID.astype(str))
    if home_id not in seen or away_id not in seen:
        raise ValueError(f'Season {season} teams not found in validated pregame history')
    lrecords=league.to_dict('records')
    lookup={k:_sumratio(lrecords,k) for k in ('PACE',)+FACTOR_NAMES}
    def create(team,opp,side):
        # Defense against previous opponents appears on rows with OPP_ID=opp.
        attack=current[current.TEAM_ID.astype(str)==team].sort_values('GAME_DATE').tail(MAX_TEAM_HISTORY)
        defense=current[current.OPP_ID.astype(str)==opp].sort_values('GAME_DATE').tail(MAX_TEAM_HISTORY)
        ar=attack.to_dict('records');dr=defense.to_dict('records')
        if len(ar)<1 or len(dr)<1:raise ValueError('Missing same-season observations for a team')
        result={'GAME_ID':'FUTURE_SHADOW','DAY':cutoff,'GAME_DATE':cutoff,
                'SEASON':int(season),'TEAM_ID':team,'OPP_ID':opp,'HOME_AWAY':side,
                'HOME':float(side=='H'),'HOME_TEAM_GAMES':len(ar),'OPP_DEF_GAMES':len(dr)}
        for k in ('PACE',)+FACTOR_NAMES:
            ln,le=lookup[k];base=ln/max(le,EPS)
            prior=PRIOR_TEAM_GAMES*le/max(len(lrecords),1)
            on,oe=_sumratio(ar,k);dn,de=_sumratio(dr,k)
            result[f'{k}_BASE']=base
            result[f'{k}_OFF']=(on+prior*base)/(oe+prior)
            result[f'{k}_DEF']=(dn+prior*base)/(de+prior)
        return pd.Series(result)
    return create(home_id,away_id,'H'),create(away_id,home_id,'A')
