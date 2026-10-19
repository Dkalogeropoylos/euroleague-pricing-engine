"""v0.9 regression checks; this code must never produce certified fair odds."""
import json
import numpy as np
import pandas as pd
import pytest
from calibrate_shrinkage_v09 import prepare_games, build_snapshots,feature_array

def sample_data(path,n=40):
    rows=[]
    for i in range(n):
        for team,other,side in [('A','B','H'),('B','A','A')]:
            rows.append({'SEASON':2025,'GAME_ID':f'G{i}','TEAM_ID':team,'OPP_ID':other,
                'GAME_DATE':f'2025-01-{i+1:02d}T19:00:00+00:00' if i<28 else
                f'2025-02-{i-27:02d}T19:00:00+00:00',
                'HOME_AWAY':side,'OT_FLAG':False,'POSS_RAW':73+i%3,'FG3A':20+i%9,
                'STL':5+i%3,'TO':11+i%4,'OREB':10+i%2,'DREB':22+i%5,
                'REB':32+i%2+i%5,'PTS':80+i%9,'MARKET_SCHEMA_VERSION':'official_inclusive_v2'})
    d=pd.DataFrame(rows)
    for field in ('OREB','DREB','REB','TO'):
        d['MARKET_'+field]=d[field]
    d.to_csv(path,index=False)
    return d

def test_wrong_schema_fails(tmp_path):
    p=tmp_path/'team.csv';d=sample_data(p)
    d['MARKET_SCHEMA_VERSION']='players_only_v1';d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Settlement schema mismatch'):prepare_games(p)

def test_wrong_settlement_totals_fail(tmp_path):
    p=tmp_path/'team.csv';d=sample_data(p)
    d['MARKET_REB']=d['REB']-2;d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Official-inclusive target mismatch'):prepare_games(p)

def test_same_day_features_do_not_change_when_result_changes(tmp_path):
    p=tmp_path/'team.csv';sample_data(p)
    original=prepare_games(p)
    snap_a=build_snapshots(original)
    target=snap_a.DAY.max()
    edited=original.copy();edited.loc[edited.DAY.eq(target),'FG3A']+=100
    snap_b=build_snapshots(edited)
    ss_a=snap_a[snap_a.DAY.eq(target)].sort_values('TEAM_ID')
    ss_b=snap_b[snap_b.DAY.eq(target)].sort_values('TEAM_ID')
    assert len(ss_a)==2
    for col in ['B_FG3A','off_FG3A','def_FG3A','recent_off_FG3A','recent_def_FG3A']:
        np.testing.assert_allclose(ss_a[col],ss_b[col])
    assert not np.array_equal(ss_a.Y_FG3A.to_numpy(),ss_b.Y_FG3A.to_numpy())

def test_features_finite(tmp_path):
    p=tmp_path/'team.csv';sample_data(p)
    s=build_snapshots(prepare_games(p))
    x=feature_array(s,'FG3A',4.,.5,'multiscale')
    assert np.isfinite(x).all()
    assert x.shape[1]==5
