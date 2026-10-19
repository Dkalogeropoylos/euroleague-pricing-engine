import pandas as pd
import numpy as np
from core.current_only_v10 import (current_only_upcoming, build_current_only_snapshots,FACTORS)

def fake_event_history():
    rows=[]
    for season in (2025,2026):
        for day in range(1,31):
            # Game: TEAM1 home vs TEAM2, half time alternating home/away.
            for team,opp,side in [('AAA','BBB','H'),('BBB','AAA','A')]:
                x={'SEASON':season,'GAME_ID':f'{season}_{day}','TEAM_ID':team,'OPP_ID':opp,
                   'HOME_AWAY':side,'GAME_DATE':pd.Timestamp(f'{season}-01-01',tz='UTC')+pd.Timedelta(days=day)}
                for k in FACTORS:
                    den=1. if k=='PACE' else 10.
                    num=70.+day if k=='PACE' else (5.+day*.25)
                    x[f'N_{k}']=num;x[f'E_{k}']=den
                for field in ('PTS','FG2A','FG3A','FG2M','FG3M','FTA','FTM',
                              'MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK'):
                    x[field]=5.+day*.25
                x['SHARED_PACE_LABEL']=70.+day
                rows.append(x)
    return pd.DataFrame(rows)

def test_forecast_inputs_exclusively_current_season():
    x=fake_event_history()
    h,a=current_only_upcoming(x,season=2026,home_id='AAA',away_id='BBB',asof_date='2026-02-01')
    x2=x[x.SEASON.eq(2026)].copy()
    hh,aa=current_only_upcoming(x2,season=2026,home_id='AAA',away_id='BBB',asof_date='2026-02-01')
    for v,z in ((h,hh),(a,aa)):
        for k in FACTORS:
            for suffix in ('BASE','OFF','DEF'):
                assert v[f'{k}_{suffix}']==z[f'{k}_{suffix}']

def test_future_games_and_same_day_cannot_leak():
    x=fake_event_history()
    h,a=current_only_upcoming(x,season=2026,home_id='AAA',away_id='BBB',asof_date='2026-01-28')
    change=x.copy()
    change.loc[change.GAME_DATE>=pd.Timestamp('2026-01-28',tz='UTC'),'N_FG3A' if 'N_FG3A' in x.columns else 'N_THREE_SHARE']=1e6
    h2,a2=current_only_upcoming(change,season=2026,home_id='AAA',away_id='BBB',asof_date='2026-01-28')
    for v,z in ((h,h2),(a,a2)):
        for k in FACTORS:
            assert v[f'{k}_BASE']==z[f'{k}_BASE']
            assert v[f'{k}_OFF']==z[f'{k}_OFF']

def test_current_season_strict_guard():
    x=fake_event_history()
    try:current_only_upcoming(x,season=2027,home_id='AAA',away_id='BBB',asof_date='2027-03-01')
    except ValueError:pass
    else:raise AssertionError('Should fail without current-season data')

def test_snapshots_start_after_minimum_prior_games():
    x=fake_event_history()
    d=build_current_only_snapshots(x,prior_games=4,min_team_games=2,min_league_rows=8)
    assert not d.empty
    assert (d.CURRENT_TEAM_GAMES>=2).all()
    assert (d.CURRENT_OPP_ALLOWED_GAMES>=2).all()
