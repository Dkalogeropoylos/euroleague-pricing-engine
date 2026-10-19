"""EuroLeague 40-minute current-season-only opponent-adjusted team model v1.1.
Historical games (2023-25) select stat-specific priors/opponent elasticities;
all live offensive, opponent allowed, league baselines are from current season.
No previous-season team identity; no market odds when distribution validation fails.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict

COUNTS=('POSS','FG2A','FG3A','FTA','MARKET_TO','MARKET_OREB','MARKET_DREB','STL','BLK','AST','MARKET_REB','PTS')
RATES=('P2','P3','PF','A_RATE','CREDIT_TO')
ALL=COUNTS+RATES
SOURCES={'POSS':('SHARED_PACE_LABEL',None),'FG2A':('FG2A',None),'FG3A':('FG3A',None),
 'FTA':('FTA',None),'MARKET_TO':('MARKET_TO',None),'MARKET_OREB':('MARKET_OREB',None),
 'MARKET_DREB':('MARKET_DREB',None),'STL':('STL',None),'BLK':('BLK',None),
 'AST':('AST',None),'MARKET_REB':('MARKET_REB',None),'PTS':('PTS',None),
 'P2':('FG2M','FG2A'),'P3':('FG3M','FG3A'),'PF':('FTM','FTA'),
 'A_RATE':('AST','FGM'),'CREDIT_TO':('PLAYER_CREDITED_TO','MARKET_TO')}
PRIOR_GRID=(0,2,4,8,16)
BETA_GRID=(0,.25,.5,.75,1.)
EPS=1e-9

def tidy_data(team_csv, *,excluded_ot=True):
 from core.structural_v06 import verified_games
 d=verified_games(team_csv,exclude_ot=excluded_ot).copy()
 d['DAY']=d.GAME_DATE.dt.floor('D')
 d['FGM']=d.FG2M+d.FG3M
 return d

def _moments(hist,stat):
 num,den=SOURCES[stat]
 if not hist:return 0.,0.
 if den is None:return sum(float(x[num]) for x in hist),float(len(hist))
 return sum(float(x[num]) for x in hist),sum(float(x[den]) for x in hist)

def _base(records,k):
 a,b=_moments(records,k)
 return a/max(b,EPS)

def _feat(records,base,league,k,prior):
 numer,denom=_moments(records,k)
 lg_n,lg_d=_moments(league,k)
 prior_exp=prior * lg_d / max(len(league),1)
 return (numer+prior_exp*base)/max(denom+prior_exp,EPS) if (denom+prior_exp)>0 else base

def snapshots(df,season,prior,*,min_matches=2,min_league_games=20):
 """Always embargo full calendar date. Opponent allowed from other opponents this season ONLY."""
 season_df=df[df.SEASON.eq(int(season))].sort_values(['DAY','GAME_ID','TEAM_ID'])
 own=defaultdict(list);allow=defaultdict(list);league=[];rows=[]
 for day,group in season_df.groupby('DAY',sort=True):
  if len(league)>=2*min_league_games:
   bases={k:_base(league,k) for k in ALL}
   for r in group.to_dict('records'):
    h=own[str(r['TEAM_ID'])];opp=allow[str(r['OPP_ID'])]
    if len(h)<min_matches or len(opp)<min_matches:continue
    q={'GAME_ID':str(r['GAME_ID']),'DAY':day,'SEASON':int(season),'HOME_AWAY':r['HOME_AWAY'],
       'TEAM_ID':str(r['TEAM_ID']),'OPP_ID':str(r['OPP_ID']),
       'N_OWN':len(h),'N_OPP':len(opp)}
    for k in ALL:
     b=bases[k]
     q[k+'_OFF']=_feat(h,b,league,k,prior)
     q[k+'_DEF']=_feat(opp,b,league,k,prior)
     q[k+'_BASE']=b
     num,den=SOURCES[k]
     q[k+'_ACTUAL']=float(r[num])/(max(float(r[den]),EPS) if den else 1.)
    rows.append(q)
  for r in group.to_dict('records'):
   league.append(r);own[str(r['TEAM_ID'])].append(r);allow[str(r['OPP_ID'])].append(r)
 return pd.DataFrame(rows)

def _predict_one(row,k,beta):
 v=float(row[k+'_OFF'])+beta*(float(row[k+'_DEF'])-float(row[k+'_BASE']))
 lo,hi= ((.1,1.0) if k in RATES else (0.,125.))
 if k=='POSS':lo,hi=55,95
 return float(np.clip(v,lo,hi))

def select_params(df):
 """Select hyperparameters solely using 2024, report untouched year-2025 test.
 Fixed offensive coefficient 1; single opponent deviation coefficient learned per market.
 Does NOT tune against 2026, bookmaker odds or named matchups.
 """
 prior_snap={p:{s:snapshots(df,s,p) for s in (2023,2024,2025,2026)} for p in PRIOR_GRID}
 result={}; summary={}
 for k in ALL:
  scoring=[]
  for p,tbl in prior_snap.items():
   s24=tbl[2024]
   if k=='POSS':s24=s24[s24.HOME_AWAY.eq('H')]
   y=s24[k+'_ACTUAL'].to_numpy(float)
   for b in BETA_GRID:
    z=(s24[k+'_OFF']+b*(s24[k+'_DEF']-s24[k+'_BASE'])).to_numpy(float)
    if k in RATES:
     # Count-level loss: target successes with actual opportunities in the 2024
     # validation; this tuning IS NOT a forward projection input.
     # Rate scoring used only for coefficient choice; all forecast counts constructed jointly.
     target=s24['FG2A_ACTUAL'].to_numpy() if k=='P2' else (s24['FG3A_ACTUAL'].to_numpy() if k=='P3' else (s24['FTA_ACTUAL'].to_numpy() if k=='PF' else (s24['MARKET_TO_ACTUAL'].to_numpy() if k=='CREDIT_TO' else np.ones(len(s24)))))
     score=float(np.mean(((z-y)*target)**2))
    else:score=float(np.mean((z-y)**2))
    scoring.append((score,p,b))
  score,p,b=min(scoring)
  s25=prior_snap[p][2025];s26=prior_snap[p][2026]
  def met(t):
   if k=='POSS':t=t[t.HOME_AWAY.eq('H')]
   if t.empty:return {'rmse':None,'base_rmse':None,'n':0}
   y=t[k+'_ACTUAL'].to_numpy(float);pr=t[k+'_OFF'].to_numpy(float)+b*(t[k+'_DEF']-t[k+'_BASE']).to_numpy(float)
   return {'rmse':round(float(np.mean((pr-y)**2)**.5),4),'base_rmse':round(float(np.mean((t[k+'_OFF'].to_numpy(float)-y)**2)**.5),4),'n':len(t)}
  result[k]={'prior':p,'beta':b,'cv_2024_rmse':round(float(score**.5),4)}
  summary[k]={'historical_test_2025':met(s25),'watch_2026':met(s26)}
 return {'version':'v1.1','live_contract':'SAME_SEASON_ONLY','training_seasons':[2023,2024,2025],
         'selection_season':2024,'untuned_test_season':2025,'status':'SHADOW_NOT_PRICING_READY',
         'market_definition':'official_inclusive_v2','params':result,'validation':summary}

def live_features(d,config,season,home_id,away_id,asof):
 cut=pd.Timestamp(asof)
 if cut.tzinfo is None:cut=cut.tz_localize('UTC')
 cut=cut.floor('D')
 cur=d[(d.SEASON.eq(int(season)))&(d.DAY<cut)].sort_values('GAME_DATE')
 if cur.GAME_ID.nunique()<20:raise ValueError('Too few current-season games')
 def row(team,opp,side):
  his=cur[cur.TEAM_ID.eq(team)].to_dict('records')
  allowed=cur[cur.OPP_ID.eq(opp)].to_dict('records')
  if min(len(his),len(allowed))<3:raise ValueError('Too few same-season own/opponent games')
  lg=cur.to_dict('records');q={'TEAM_ID':team,'OPP_ID':opp,'HOME_AWAY':side,
      'N_OWN':len(his),'N_OPP':len(allowed),'LAST_DAY':str(cur.DAY.max()),'SOURCE_SEASON':int(season)}
  for k,opt in config['params'].items():
   base=_base(lg,k);prior=opt['prior']
   off=_feat(his,base,lg,k,prior);defe=_feat(allowed,base,lg,k,prior)
   q[k+'_BASE']=base;q[k+'_OFF']=off;q[k+'_DEF']=defe
   # Transparent current-season provenance: raw observed profiles and trained-prior share.
   # For rate variables the weight is in effective opportunities, not just games.
   _,lg_d=_moments(lg,k); _,own_d=_moments(his,k); _,opp_d=_moments(allowed,k)
   pseudo=prior*lg_d/max(len(lg),1)
   q[k+'_RAW_OFF']=_base(his,k);q[k+'_RAW_DEF']=_base(allowed,k)
   q[k+'_PRIOR_SHARE_OFF']=pseudo/max(pseudo+own_d,EPS)
   q[k+'_PRIOR_SHARE_DEF']=pseudo/max(pseudo+opp_d,EPS)
   q[k+'_PRED']=float(np.clip(off+opt['beta']*(defe-base),0,150))
  return q
 return row(home_id,away_id,'H'),row(away_id,home_id,'A')

def _safe(v,lo=0.,hi=150.):return np.clip(v,lo,hi)

# Project all 11 observed central counts onto exact shared-possession closure.
# Variances from the untouched historical year-2025 error profile, NEVER from handpicked matchups.
EQUATIONS=np.array([[-1,1,1,.44,1,-1,0,0,0,0,0],
                    [-1,0,0,0,0,0,1,1,.44,1,-1]],float)
ORDER=('PACE','FG2A','FG3A','FTA','MARKET_TO','MARKET_OREB')

def reconcile_counts(raw_h,raw_a,sd=None):
 if sd is None:sd={k:4. for k in ORDER}
 pace=float((raw_h['POSS_PRED']+raw_a['POSS_PRED'])/2)
 vec=np.array([pace]+[raw_h[k+'_PRED'] for k in ORDER[1:]]+[raw_a[k+'_PRED'] for k in ORDER[1:]],float)
 variances=np.array([sd.get('POSS',4.)**2]+[sd.get(k,4.)**2 for k in ORDER[1:]]*2)
 A=EQUATIONS
 delta=variances[:,None]*A.T@np.linalg.solve((A*variances)@A.T,A@vec)
 vec=vec-delta
 if (vec[1:]<0).any() or not (55<=vec[0]<=95):
  raise ValueError('Pregame inputs require implausible constrained correction — HOLD pricing')
 return vec,{'adjustment_by_side':{'home':(vec[1:6]-np.array([raw_h[k+'_PRED'] for k in ORDER[1:]])).tolist(),
       'away':(vec[6:]-np.array([raw_a[k+'_PRED'] for k in ORDER[1:]])).tolist()},
       'pace_shift':float(vec[0]-pace)}

def projection(h,a,sd=None,home_adjustment=None,margin_target=None,component_weights=None):
 vec,diag=reconcile_counts(h,a,sd)
 pace=vec[0]
 def side(r,vals):
  fg2a,fg3a,fta,to,oreb=[float(x) for x in vals]
  p2=float(_safe(r['P2_PRED'],.25,.8));p3=float(_safe(r['P3_PRED'],.16,.60));pf=float(_safe(r['PF_PRED'],.35,.95))
  fg2m=fg2a*p2;fg3m=fg3a*p3;ftm=fta*pf
  ast=(fg2m+fg3m)*float(_safe(r['A_RATE_PRED'],.15,.9))
  return {'POSS':pace,'FG2A':fg2a,'FG3A':fg3a,'FG2M':fg2m,'FG3M':fg3m,
   'FTA':fta,'FTM':ftm,'MARKET_TO':to,'MARKET_OREB':oreb,'AST':ast,
   'FGM':fg2m+fg3m,'FGA':fg2a+fg3a,'PLAYER_CREDITED_TO':to*float(_safe(r['CREDIT_TO_PRED'],.4,1)),
   'PTS':2*fg2m+3*fg3m+ftm}
 H=side(h,vec[1:6]);A=side(a,vec[6:11])
 if home_adjustment:
  # Symmetric season-general home/away residuals learned only from 2024;
  # applied to shooting makes and FTM, never as an unexplained PTS additive.
  for sign,side_obj,raw in ((1.,H,h),(-1.,A,a)):
   for make,att in (('FG2M','FG2A'),('FG3M','FG3A'),('FTM','FTA')):
    side_obj[make]=float(_safe(side_obj[make]+sign*float(home_adjustment.get(make,0)),0,side_obj[att]))
   side_obj['FGM']=side_obj['FG2M']+side_obj['FG3M']
   side_obj['AST']=side_obj['FGM']*float(_safe(raw['A_RATE_PRED'],.15,.9))
   side_obj['PTS']=2*side_obj['FG2M']+3*side_obj['FG3M']+side_obj['FTM']
 if margin_target is not None:
  if not component_weights:raise ValueError('Learned scoring-variance weights required for margin reconciliation')
  before=float(H['PTS']-A['PTS'])
  half_change=.5*(float(margin_target)-before)
  for sgn,obj,raw in ((1.,H,h),(-1.,A,a)):
   for make,attempt,pts_per in (('FG2M','FG2A',2),('FG3M','FG3A',3),('FTM','FTA',1)):
    want=obj[make]+sgn*half_change*component_weights[make]/pts_per
    obj[make]=float(_safe(want,0,obj[attempt]))
   obj['FGM']=obj['FG2M']+obj['FG3M']
   obj['AST']=obj['FGM']*float(_safe(raw['A_RATE_PRED'],.15,.9))
   obj['PTS']=2*obj['FG2M']+3*obj['FG3M']+obj['FTM']
  diag['margin_target']=float(margin_target)
  diag['margin_realized']=float(H['PTS']-A['PTS'])
  if abs(diag['margin_realized']-diag['margin_target'])>0.8:
   raise ValueError('Team shooting cannot absorb learned margin correction; HOLD forecast')
 for mine,other,r in ((H,A,h),(A,H,a)):
  # Opponent shooting misses and own rebound market are linked, but missing PBP
  # means precise reboundability of missed FT is UNKNOWN (do not fabricate it).
  opp_fgmiss=other['FG2A']-other['FG2M']+other['FG3A']-other['FG3M']
  proxy_opp_chance=opp_fgmiss-other['MARKET_OREB']+.5*(other['FTA']-other['FTM'])
  raw_dreb=float(r['MARKET_DREB_PRED'])
  mine['MARKET_DREB']=float(_safe(raw_dreb,0,max(0,proxy_opp_chance)+2))
  mine['MARKET_REB']=mine['MARKET_OREB']+mine['MARKET_DREB']
  mine['STL']=float(_safe(r['STL_PRED'],0,other['PLAYER_CREDITED_TO']))
  mine['BLK']=float(_safe(r['BLK_PRED'],0,other['FG2A']-other['FG2M']))
 diag['reb_market_opp_proxy']= {'home':float((A['FG2A']-A['FG2M']+A['FG3A']-A['FG3M'])-A['MARKET_OREB']+.5*(A['FTA']-A['FTM'])),
                               'away':float((H['FG2A']-H['FG2M']+H['FG3A']-H['FG3M'])-H['MARKET_OREB']+.5*(H['FTA']-H['FTM']))}
 return {'home':H,'away':A,'pace':pace,'audit':diag,'market_schema_version':'official_inclusive_v2'}

def counts_audit(pred,rh,ra):
 return {side:{'3PA_current_own':float(raw['FG3A_OFF']),'3PA_allowed_opponent':float(raw['FG3A_DEF']),
  '3PA_league':float(raw['FG3A_BASE']),'3PA_unconstrained':float(raw['FG3A_PRED']),
  '3PA_final':float(pred[side]['FG3A']),'REB_current_own':float(raw['MARKET_REB_OFF']),
  'REB_allowed_opponent':float(raw['MARKET_REB_DEF']),'REB_final':float(pred[side]['MARKET_REB']),
  'SHOTS_total':pred[side]['FGA'],'POSS':pred['pace']}
 for side,raw in (('home',rh),('away',ra))}

def bootstrap_simulation(pred, residual_bank, *,n=5000,seed=11,sd=None):
 """Paired empirical residual bootstrap. Historical 2025 prediction errors shared across
 sides, preserving approximate common-game correlations. NOT independently calibrated.
 40-minute research counts; exact PTS and REB arithmetic identities.
 """
 if n<200:raise ValueError('Need at least 200 paired simulation scenarios')
 if not residual_bank:raise ValueError('No historical residual bank')
 rng=np.random.default_rng(seed)
 ix=rng.integers(0,len(residual_bank),size=n)
 # Historical errors can have nonzero positional bias. Center EACH field on the
 # full reference bank before resampling, preserving cross-market covariance.
 # Otherwise MC shifts means, inverting 3-way most-stat probabilities.
 arr={}
 for k in residual_bank[0]:
  if k=='GAME_ID':continue
  population=np.array([r.get(k,0.) for r in residual_bank],dtype=float)
  if not np.isfinite(population).all():raise ValueError('Non-finite calibration residual: '+k)
  arr[k]=population[ix]-population.mean()
 def get(k):return arr.get(k,np.zeros(n,float))
 v=np.empty((n,11),float)
 v[:,0]=_safe(pred['pace']+get('POSS'),55,95)
 for j,side in enumerate(('home','away')):
  for m,k in enumerate(('FG2A','FG3A','FTA','MARKET_TO','MARKET_OREB')):
   v[:,1+j*5+m]=_safe(pred[side][k]+get(side+'_'+k),.1,100.)
 if sd is None:sd={k:4. for k in ORDER}
 variance=np.array([sd.get('POSS',4)**2]+[sd.get(k,4)**2 for k in ORDER[1:]]*2)
 A=EQUATIONS
 transform=(variance[:,None]*A.T)@np.linalg.inv((A*variance)@A.T)
 v-=v@A.T@transform.T
 # Hard constraint after rounding may differ by 1 possession; retain arithmetic scoring/reb consistency.
 v[:,0]=np.rint(_safe(v[:,0],55,95))
 for i in range(1,11):v[:,i]=np.rint(_safe(v[:,i],0,100))
 out={}
 for j,side in enumerate(('home','away')):
  vals=v[:,1+j*5:6+j*5]
  fg2a,fg3a,fta,to,oreb=[vals[:,z].astype(int) for z in range(5)]
  p=pred[side]
  # Paired empirical shooting-rate errors capture the observed shooting fluctuation;
  # do not add independent binomial luck a second time.
  p2=_safe(p['FG2M']/max(p['FG2A'],EPS)+get(side+'_P2'),.01,.99)
  p3=_safe(p['FG3M']/max(p['FG3A'],EPS)+get(side+'_P3'),.01,.99)
  pf=_safe(p['FTM']/max(p['FTA'],EPS)+get(side+'_PF'),.01,.99)
  m2=np.minimum(fg2a,np.rint(fg2a*p2).astype(int));m3=np.minimum(fg3a,np.rint(fg3a*p3).astype(int))
  ftm=np.minimum(fta,np.rint(fta*pf).astype(int))
  ast_rate=_safe(p['AST']/max(p['FGM'],EPS)+get(side+'_AST_RATE'),0,1)
  ast=np.minimum(m2+m3,np.rint((m2+m3)*ast_rate).astype(int))
  out[side]={'FG2A':fg2a,'FG3A':fg3a,'FG2M':m2,'FG3M':m3,
   'FTA':fta,'FTM':ftm,'MARKET_TO':to,'MARKET_OREB':oreb,
   'AST':ast,'PTS':2*m2+3*m3+ftm,'POSS':v[:,0].astype(int),
   'FGA':fg2a+fg3a}
 for side,other in (('home','away'),('away','home')):
  own=out[side];opp=out[other];p=pred[side]
  proxy=(opp['FG2A']-opp['FG2M'])+(opp['FG3A']-opp['FG3M'])-opp['MARKET_OREB']+.5*(opp['FTA']-opp['FTM'])
  dreb=np.rint(_safe(p['MARKET_DREB']+get(side+'_MARKET_DREB'),0,np.maximum(0,proxy)+2)).astype(int)
  own['MARKET_DREB']=dreb;own['MARKET_REB']=own['MARKET_OREB']+dreb
  # Steals can arise only on credited opponent turnovers, not TEAM_ONLY turnovers.
  opp_credit=opp['MARKET_TO']*pred[other]['PLAYER_CREDITED_TO']/max(pred[other]['MARKET_TO'],EPS)
  own['STL']=np.rint(_safe(p['STL']+get(side+'_STL'),0,opp_credit)).astype(int)
  own['BLK']=np.rint(_safe(p['BLK']+get(side+'_BLK'),0,opp['FG2A']-opp['FG2M'])).astype(int)
 return {'home':out['home'],'away':out['away'],'pace':v[:,0].astype(int),
         'market_schema_version':'official_inclusive_v2',
         'limitations':'HOLD: empirically bootstrapped 2025 residuals; 2026 coverage is watch only, not validated for production odds'}


def quality_gate(pred,h,a,external=None,calibration=None):
 """Fail closed when data/lineups/price calibration are insufficient. No subjective retuning."""
 flags=[]
 for side,raw in (('home',h),('away',a)):
  p=pred[side]
  if raw['N_OWN']<6 or raw['N_OPP']<6:flags.append(f'{side}: thin current-season team/opponent sample')
  for k in ('FG2A','FG3A','FTA','MARKET_TO','MARKET_REB'):
   baseline=raw[k+'_OFF'];move=p[k]-(baseline)
   if abs(move)>max(6.,.30*max(baseline,1)):
    flags.append(f'{side} {k}: forecast differs materially from own same-season offensive estimate ({move:+.1f})')
  if abs(pred['audit']['pace_shift'])>3.5:flags.append('Pregame possessions reconciliation moved pace >3.5')
  if min(p['FG3A'],p['FG2A'],p['MARKET_REB'])<=0:flags.append(f'{side}: nonpositive essential market')
 if external:
  if 'total' in external:
   model=pred['home']['PTS']+pred['away']['PTS']
   if abs(model-external['total'])>=7:
    flags.append(f'External full-game total discrepancy {model-external["total"]:+.1f}; note regulation vs OT settlement differs')
  if 'home_margin' in external:
   model=pred['home']['PTS']-pred['away']['PTS']
   if abs(model-external['home_margin'])>=5:
    flags.append(f'External spread discrepancy {model-external["home_margin"]:+.1f}; check roster and venue before any quote')
 if not calibration or calibration.get('production_pricing_gate') is not True:
  flags.append('Distribution calibration not independently verified — all bookmaker quoting HOLD')
 return {'status':'HOLD' if flags else 'REVIEW_REQUIRED','flags':flags}


def full_current_only_forecast(data,config,season,home,away,asof):
    from core.margin_v11 import upcoming_margin
    h,a=live_features(data,config,season,home,away,asof)
    sd=config['chosen_reconciliation_sd_2024']
    home_adj=config['home_adjustment_from_2024_made_baskets']
    basic=projection(h,a,sd,home_adj)
    independent=upcoming_margin(data,season,home,away,asof,config['margin_model'])
    mix=config['margin_model']['blend_weight']
    target=(1-mix)*(basic['home']['PTS']-basic['away']['PTS'])+mix*independent
    final=projection(h,a,sd,home_adj,target,config['margin_model']['score_component_weights'])
    final['audit']['independent_net_margin']=float(independent)
    final['audit']['structural_margin_before_blend']=float(basic['home']['PTS']-basic['away']['PTS'])
    return final,h,a
