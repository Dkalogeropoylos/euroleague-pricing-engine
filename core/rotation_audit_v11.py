"""Team role and rotation audit from CURRENT-SEASON player box scores only.
Descriptive, NOT automatic injury availability inference or pricing override.
"""
import pandas as pd,numpy as np

def current_rotation_audit(players,season,team,opponent,asof):
 d=players.copy()
 d['GAME_DATE']=pd.to_datetime(d.GAME_DATE,utc=True)
 cut=pd.Timestamp(asof)
 if cut.tzinfo is None:cut=cut.tz_localize('UTC')
 d=d[(d.SEASON==season)&(d.GAME_DATE.dt.floor('D')<cut.floor('D'))]
 own=d[d.TEAM_ID.eq(team)].copy()
 allowed=d[d.OPP_ID.eq(opponent)].copy()
 if own.empty:return pd.DataFrame(),pd.DataFrame(),[]
 last=sorted(own.GAME_DATE.dt.floor('D').unique())
 latest=set(last[-2:]);earlier=set(last[:-2])
 rows=[];flags=[]
 for pid,g in own.groupby('PLAYER_ID'):
  recent=g[g.GAME_DATE.dt.floor('D').isin(latest)]
  past=g[g.GAME_DATE.dt.floor('D').isin(earlier)]
  rmin=float(recent.MINUTES.mean()) if not recent.empty else 0.
  pmin=float(past.MINUTES.mean()) if not past.empty else float('nan')
  if np.isfinite(pmin) and abs(rmin-pmin)>=9 and max(rmin,pmin)>9:
   flags.append(f'{str(g.iloc[-1].PLAYER_NAME)} minutes in last two {rmin:.1f} vs earlier {pmin:.1f}; verify rotation')
  rows.append({'Player':str(g.iloc[-1].PLAYER_NAME),'Position category':str(g.iloc[-1].POSITION),
               'Games':int(g.GAME_ID.nunique()),'Season MIN':float(g.MINUTES.mean()),
               'Last 2 MIN':rmin,'Season 3PA':float(g.FG3A.mean()),
               'Last 2 3PA':float(recent.FG3A.mean()) if not recent.empty else 0.,
               'Appeared last 2':int(recent.GAME_ID.nunique())})
 # Opponent-specific 3PA by raw dataset category; do not infer true defensive position.
 allowed_rows=[]
 for position,x in allowed.groupby('POSITION'):
  pergame=x.groupby('GAME_ID').FG3A.sum()
  allowed_rows.append({'Position category':str(position),'Games':len(pergame),
                       '3PA allowed/game':float(pergame.mean()),'3PA total':float(pergame.sum())})
 return pd.DataFrame(rows).sort_values('Season 3PA',ascending=False),pd.DataFrame(allowed_rows),flags
