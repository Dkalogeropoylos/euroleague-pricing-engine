"""Train from 2023-2025, evaluate 2026 using current-season-only pregame state."""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from core.structural_v06 import verified_games,event_rate_rows,project_two_sides
from core.current_only_v10 import (train_historical_only,build_current_only_snapshots,current_only_project,
    FACTORS)

def eval_current(events, conf):
    keys=('PTS','FG2A','FG3A','FG2M','FG3M','FTA','FTM','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK')
    snap=build_current_only_snapshots(events,prior_games=4.,season_filter={2026})
    full=events[events.SEASON.eq(2026)]
    errors={k:{'y':[],'pred':[],'baseline':[]} for k in keys}
    accepted=0;rejected=[]
    # Each game predicted with current-season only data available BEFORE the game day
    for gid, g in full.groupby('GAME_ID'):
        if len(g)!=2:continue
        h=g[g.HOME_AWAY.eq('H')].iloc[0];a=g[g.HOME_AWAY.eq('A')].iloc[0]
        try:
            pred,rows,mod=current_only_project(events,conf,season=2026,home_id=str(h.TEAM_ID),away_id=str(a.TEAM_ID),asof_date=h.GAME_DATE)
            baseline=project_two_sides(rows[0],rows[1],{})
        except ValueError as e:
            rejected.append((str(gid),str(e)));continue
        accepted+=1
        for raw,side in ((h,'home'),(a,'away')):
            for k in keys:
                errors[k]['y'].append(float(raw[k]));errors[k]['pred'].append(float(pred[side][k]));errors[k]['baseline'].append(float(baseline[side][k]))
    summary={}
    for k,z in errors.items():
        if not z['y']:continue
        y=np.asarray(z['y']);p=np.asarray(z['pred']);b=np.asarray(z['baseline'])
        rmse=lambda x:float(np.sqrt(np.mean((y-x)**2)))
        summary[k]={'n_team_games':len(y),'rmse':round(rmse(p),3),'baseline_rmse':round(rmse(b),3),
                    'bias':round(float(np.mean(p-y)),3),'std_pred':round(float(np.std(p)),3),
                    'std_actual':round(float(np.std(y)),3)}
    return {'games_evaluated':accepted,'games_skipped_before_minimum':len(rejected),'metrics':summary,
            'limitations':['2026 previously scrutinized: NOT untouched holdout',
            'Only current-season observational box score context, no lineups/injuries',
            'Event-rate coefficient validation is NOT same as joint fair-odds calibration',
            'DREB FT proxy not PBP; odds and simulation still SHADOW']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/euroleague/exports')
    p.add_argument('--out',default='data/euroleague/shadow_v10/current_only_v10_report.json')
    a=p.parse_args();d=Path(a.data);target=Path(a.out)
    verified=verified_games(pd.read_csv(d/'team_games.csv'),pd.read_csv(d/'player_games.csv'))
    events=event_rate_rows(verified)
    config=train_historical_only(events)
    config['dataset']={'all_games':int(verified.GAME_ID.nunique()),'2026_games':int(verified[verified.SEASON.eq(2026)].GAME_ID.nunique()),'rows':len(verified), 'last_utc':str(verified.GAME_DATE.max())}
    config['watch_2026']=eval_current(events,config)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(config,indent=2,allow_nan=False))
    print('Current-only research report:',target,'eval:',config['watch_2026']['games_evaluated'])
    for k in ('PACE','THREE_SHARE','THREE_PCT','STL_RATE','TO_RATE','OREB_RATE','DREB_RATE'):
        q=config['factors'][k];print(k, 'prior',q['prior_games'],'mode',q['model']['selected'],
                                    '2025 rate rmse',round(q['validation_2025_rate_rmse'],4),
                                    'lg',round(q['league_2025_rate_rmse'],4))
    for k in ('PTS','FG3A','FG3M','MARKET_REB','MARKET_TO','STL'):
        print(k,config['watch_2026']['metrics'].get(k))
if __name__=='__main__':main()
