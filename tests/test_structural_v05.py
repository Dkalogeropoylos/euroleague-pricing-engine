from __future__ import annotations
import numpy as np
import pandas as pd
from core.structural_v05 import (FACTOR_NAMES,chronological_design,
                                 project_two_sides,simulate_two_sides,upcoming_pair)


def faux_feature(team,side,game='SIM'):
    defaults={'PACE':73.,'TO_RATE':.169,'TO_CREDIT':.94,'FTA_RATE':.26,
              'THREE_SHARE':.41,'TWO_PCT':.55,'THREE_PCT':.35,'FT_PCT':.75,
              'OREB_RATE':.32,'OREB_CREDIT':.82,'DREB_RATE':.88,
              'AST_RATE':.65,'STL_RATE':.55,'BLK_RATE':.14}
    base={'GAME_ID':game,'TEAM_ID':team,'OPP_ID':'AWAY' if side=='H' else 'HOME',
          'HOME_AWAY':side,'HOME':float(side=='H')}
    for k,v in defaults.items():
        for s in ['BASE','OFF','DEF']:base[f'{k}_{s}']=v
    return pd.Series(base)


def test_structural_mean_coherence():
    h,a=faux_feature('HOME','H'),faux_feature('AWAY','A')
    p=project_two_sides(h,a,{})
    assert p['home']['POSS']==p['away']['POSS']==p['pace']
    for side in ('home','away'):
        d=p[side]
        assert np.isclose(d['PTS'],2*d['FG2M']+3*d['FG3M']+d['FTM'])
        assert np.isclose(d['MARKET_REB'],d['MARKET_OREB']+d['MARKET_DREB'])
        assert np.isclose(d['FG2A']+d['FG3A'],d['POSS']-d['OFFICIAL_TO']-.44*d['FTA']+d['OFFICIAL_OREB'])
        assert d['MARKET_OREB']<=d['OFFICIAL_OREB']
        assert d['FG2M']<=d['FG2A'] and d['FG3M']<=d['FG3A']
        assert d['AST']<=d['FG2M']+d['FG3M']
    assert p['home']['STL']<=p['away']['MARKET_TO']
    assert p['away']['BLK']<=p['home']['FG2_MISSES']


def test_montecarlo_draw_identities_and_reproducibility():
    h,a=faux_feature('HOME','H'),faux_feature('AWAY','A')
    sims=simulate_two_sides(h,a,{},n=12000,seed=123)
    again=simulate_two_sides(h,a,{},n=12000,seed=123)
    assert np.array_equal(sims['home']['PTS'],again['home']['PTS'])
    assert np.array_equal(sims['home']['POSS'],sims['away']['POSS'])
    for side in ('home','away'):
        x=sims[side]
        np.testing.assert_array_equal(x['PTS'],2*x['FG2M']+3*x['FG3M']+x['FTM'])
        np.testing.assert_array_equal(x['MARKET_REB'],x['MARKET_OREB']+x['MARKET_DREB'])
        np.testing.assert_array_equal(x['FG2A']+x['FG3A'],x['BASE_FGA']+x['OFFICIAL_OREB'])
        assert (x['FG2M']<=x['FG2A']).all() and (x['FG3M']<=x['FG3A']).all()
        assert (x['AST']<=x['FG2M']+x['FG3M']).all()
        assert (x['MARKET_OREB']<=x['OFFICIAL_OREB']).all()
        assert (x['MARKET_TO']<=x['OFFICIAL_TO']).all()
        assert all((x[k]>=0).all() for k in ['POSS','MARKET_TO','FG2A','FG3A','FG2M','FG3M','MARKET_REB','STL','BLK'])
    assert (sims['home']['STL']<=sims['away']['MARKET_TO']).all()
    assert (sims['away']['STL']<=sims['home']['MARKET_TO']).all()
    assert (sims['home']['BLK']<=sims['away']['FG2_MISSES']).all()
    assert (sims['away']['BLK']<=sims['home']['FG2_MISSES']).all()


def fake_events():
    rec=[]
    for d in range(54):
        for team,opp,side in [('H','A','H'),('A','H','A')]:
            obj={'GAME_ID':f'G{d}','GAME_DATE':pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=d),
                 'TEAM_ID':team,'OPP_ID':opp,'SEASON':2023,'HOME_AWAY':side}
            for k in ('PACE',)+FACTOR_NAMES:
                obj[f'N_{k}']=73 if k=='PACE' else 10
                obj[f'E_{k}']=1 if k=='PACE' else 60
            for k in ['PTS','FG2A','FG3A','FTA','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK','FG2M','FG3M','FTM']:
                obj[k]=12
            rec.append(obj)
    return pd.DataFrame(rec)


def test_calendar_day_embargo_and_future_window():
    d=fake_events()
    r=chronological_design(d,min_prior_games=20)
    last=r[r.GAME_ID.eq('G53')].copy().sort_values('TEAM_ID')
    # Mutating this game's target must NOT change its own pregame features.
    altered=d.copy(); altered.loc[altered.GAME_ID.eq('G53'),'N_THREE_SHARE']=50
    altered.loc[altered.GAME_ID.eq('G53'),'N_PACE']=95
    rr=chronological_design(altered,min_prior_games=20)
    nxt=rr[rr.GAME_ID.eq('G53')].sort_values('TEAM_ID')
    for key in ['PACE_BASE','PACE_OFF','PACE_DEF','THREE_SHARE_BASE','THREE_SHARE_OFF','THREE_SHARE_DEF']:
        np.testing.assert_allclose(last[key],nxt[key])
    # Future previews may only look backwards from requested UTC game day.
    h,a=upcoming_pair(d,season=2023,home_id='H',away_id='A',asof_date='2024-03-01')
    assert h.HOME_AWAY=='H' and a.HOME_AWAY=='A'
    with np.testing.assert_raises(ValueError):
        upcoming_pair(d,season=2023,home_id='H',away_id='H',asof_date='2024-03-01')
