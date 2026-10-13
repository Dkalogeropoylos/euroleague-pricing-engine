"""v0.8 regression tests: no invented multipliers or settlement changes."""
import numpy as np
import pandas as pd
from core.discrimination_v08 import (_features,_rate,_fit,_candidate_pred,
                                      apply_challenger,TARGETS,FEATURES)

def test_prior_history_is_exposure_weighted():
    a=[{'N_PACE':70.,'E_PACE':1.},
       {'N_PACE':80.,'E_PACE':1.}]
    assert abs(_rate(a,'PACE',74.,0)-75.)<1.e-9
    assert _rate([], 'PACE', 74., 8.)==74.

def test_missing_previous_season_is_neutral_not_invented():
    h=[{'N_THREE_SHARE':3.,'E_THREE_SHARE':10.}]
    x=_features(h,[],[],[],0.25,'THREE_SHARE',True,4.)
    assert len(x)==len(FEATURES)
    assert x[0]>0 and x[2]==0. and x[5]==0.
    assert x[-1]==1.

def test_rate_forecast_respects_physical_probability_bounds():
    k='THREE_SHARE'
    d={f'{k}_BASE':0.28}
    d.update({f'V8_{k}_{x}':1.0 for x in FEATURES})
    table=pd.DataFrame([d])
    model={'kind':'multiscale','intercept':100.0,'coefficients':[30.]*7}
    result=float(_candidate_pred(table,k,model)[0])
    assert 0.12<=result<=0.76

def test_previous_model_passthrough_and_schema_not_mutated():
    h=pd.Series({'GAME_ID':'FUTURE_SHADOW','HOME_AWAY':'H','PACE_BASE':73.,'THREE_SHARE_BASE':.32})
    a=pd.Series({'GAME_ID':'FUTURE_SHADOW','HOME_AWAY':'A','PACE_BASE':73.,'THREE_SHARE_BASE':.32})
    mdl={'PACE':{'selected':'league'},'THREE_SHARE':{'selected':'league'}}
    ch={'targets':{'PACE':{'kind':'v06'},'THREE_SHARE':{'kind':'league'}}}
    x,y,m=apply_challenger(h,a,mdl,ch)
    assert x['PACE_BASE']==73 and y['THREE_SHARE_BASE']==.32
    assert m==mdl
    assert h['PACE_BASE']==73

def test_full_challenger_factor_list():
    assert {'PACE','THREE_SHARE','OREB_RATE','DREB_RATE','TWO_PCT','THREE_PCT'}.issubset(TARGETS)
