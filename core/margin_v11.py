"""Same-season team net-strength opponent adjustment, coefficients chosen in 2024.
This captures strength differences that can be lost in independently shrunk team counts.
No use of previous-season specific team net ratings in a live forecast.
"""
import pandas as pd,numpy as np
from collections import defaultdict


def home_edge_prior(data):
 x=data[(data.SEASON==2023)].pivot(index='GAME_ID',columns='HOME_AWAY',values='PTS')
 return float((x.H-x.A).mean())

def strength_snapshots(data,season,k):
 df=data[data.SEASON.eq(season)].sort_values(['DAY','GAME_ID'])
 own=defaultdict(list);rows=[];league=[]
 for day,group in df.groupby('DAY'):
  if len(league)>=40:
   for gid,g in group.groupby('GAME_ID'):
    if len(g)!=2:continue
    h=g[g.HOME_AWAY=='H'].iloc[0];a=g[g.HOME_AWAY=='A'].iloc[0]
    hs=own[str(h.TEAM_ID)];aw=own[str(a.TEAM_ID)]
    if min(len(hs),len(aw))<2:continue
    rows.append({'GAME_ID':gid,'HOME_ID':h.TEAM_ID,'AWAY_ID':a.TEAM_ID,
                 'HOME_GAMES':len(hs),'AWAY_GAMES':len(aw),
                 'HOME_NET':sum(hs)/(len(hs)+k),'AWAY_NET':sum(aw)/(len(aw)+k),
                 'ACTUAL_MARGIN':float(h.PTS-a.PTS)})
  for _,r in group.iterrows():
   own[str(r.TEAM_ID)].append(float(r.PTS-r.OPP_PTS));league.append(r)
 return pd.DataFrame(rows)

def margin_prediction(snapshot,config):
 return config['home_edge_from_2023']+config['net_coefficient']*(snapshot.HOME_NET-snapshot.AWAY_NET)

def select_margin_config(data,structural_2024):
 c=home_edge_prior(data);candidate=[]
 for k in (0,1,2,4,8,16):
  f=strength_snapshots(data,2024,k)
  for b in (0,.25,.5,.75,1.,1.25,1.5):
   p=c+b*(f.HOME_NET-f.AWAY_NET)
   rmse=float(np.mean((p-f.ACTUAL_MARGIN)**2)**.5)
   candidate.append((rmse,k,b))
 _,k,b=min(candidate)
 config={'home_edge_from_2023':round(c,6),'net_prior_games':int(k),'net_coefficient':float(b),
         'selection_2024_margin_rmse':round(float(min(candidate)[0]),4)}
 m=strength_snapshots(data,2024,k).set_index('GAME_ID')
 pv=structural_2024.pivot(index='GAME_ID',columns='SIDE',values=['PTS','PTS_pred'])
 struct=(pv['PTS_pred']['H']-pv['PTS_pred']['A']).rename('STRUCT_MARGIN')
 realized=(pv['PTS']['H']-pv['PTS']['A']).rename('TRUE_MARGIN')
 j=m.join(pd.concat([struct,realized],axis=1),how='inner')
 independent=margin_prediction(j,config)
 scores=[]
 for mix in (0.,.25,.5,.75,1.):
  pred=j.STRUCT_MARGIN*(1-mix)+independent*mix
  scores.append((float(np.mean((pred-j.TRUE_MARGIN)**2)**.5),mix))
 score,mix=min(scores)
 config['blend_weight']=mix;config['selection_2024_blend_rmse']=round(score,4)
 # Error allocation to 2PM/3PM/FTM has no arbitrary stat weights; learned from
 # mean squared 2024 boxscore miss errors, projected into point units.
 variance={k:float(np.mean((structural_2024[k]-structural_2024[k+'_pred'])**2))
           for k in ('FG2M','FG3M','FTM')}
 scales={'FG2M':4*variance['FG2M'],'FG3M':9*variance['FG3M'],'FTM':variance['FTM']}
 total=sum(scales.values());config['score_component_weights']={k:v/total for k,v in scales.items()}
 return config

def upcoming_margin(data,season,home,away,asof,config):
 cut=pd.Timestamp(asof)
 if cut.tzinfo is None:cut=cut.tz_localize('UTC')
 df=data[(data.SEASON==season)&(data.DAY<cut.floor('D'))]
 def net(team):
  g=df[df.TEAM_ID==team]
  if len(g)<2:raise ValueError('Current net strength unavailable')
  return float((g.PTS-g.OPP_PTS).sum()/(len(g)+config['net_prior_games']))
 return config['home_edge_from_2023']+config['net_coefficient']*(net(home)-net(away))
