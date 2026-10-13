import numpy as np
import pandas as pd
from core.efficiency_benchmark_v07 import chrono_ppp_design,upcoming_efficiency_benchmark,fit_efficiency_audit

def fdata(n=800):
    rows=[]
    for i in range(n):
        for j in (0,1):
            rows.append({'GAME_ID':f'G{i}','GAME_DATE':pd.Timestamp('2023-01-01',tz='UTC')+pd.Timedelta(days=i),
              'SEASON':2023+(i//270),'TEAM_ID':str((i+j)%20),'OPP_ID':str((i+1-j)%20),
              'HOME_AWAY':'H' if j==0 else 'A','PTS':float(70+(i*3+j)%25),
              'OPP_PTS':float(75+(i+j)%20),'SHARED_PACE_LABEL':float(68+(i%4))})
    return pd.DataFrame(rows)

def test_same_day_no_leak():
    d=fdata(90);one=chrono_ppp_design(d,8)
    mutate=d.copy();mutate.loc[mutate.GAME_ID.eq('G72'),'PTS']=300
    two=chrono_ppp_design(mutate,8)
    p1=one[one.GAME_ID.eq('G72')].reset_index(drop=True)
    p2=two[two.GAME_ID.eq('G72')].reset_index(drop=True)
    assert np.allclose(p1[['BASE','X_OFF','X_DEF']],p2[['BASE','X_OFF','X_DEF']])

def test_next_date_only_history():
    d=fdata(90);r={'data_definition':'official_inclusive_v2','prior_games':8,
      'model':{'coefficients':[.5,.5,0.],'intercept':0.}}
    v=upcoming_efficiency_benchmark(d,r,2023,'2','3','2023-03-23',71)
    assert 80<v['projected_total_points']<210
    assert v['status']=='INDEPENDENT_SHADOW_BENCHMARK_NO_BLEND'
