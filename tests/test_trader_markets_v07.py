import numpy as np
import pytest
from core.trader_markets_v07 import *

def fixture_sim(n=3000):
    rng=np.random.default_rng(2)
    h={k:rng.poisson(v,n) for k,v in [('PTS',83),('FG3M',9),('FG3A',25),('FG2M',21),('FG2A',38),('FTA',18),('FTM',14),('MARKET_REB',35),('MARKET_OREB',11),('MARKET_DREB',24),('AST',18),('MARKET_TO',13),('STL',6),('BLK',3)]}
    a={k:rng.poisson(v,n) for k,v in [('PTS',78),('FG3M',8),('FG3A',23),('FG2M',20),('FG2A',36),('FTA',16),('FTM',13),('MARKET_REB',33),('MARKET_OREB',10),('MARKET_DREB',23),('AST',17),('MARKET_TO',12),('STL',7),('BLK',2)]}
    return {'market_schema_version':'official_inclusive_v2','home':h,'away':a}

def test_probabilities_exact_109_115():
    for probs,margin in [([.52,.48],1.09),([.39,.14,.47],1.15)]:
        fair,book=pricing_probabilities(probs,margin)
        assert np.isclose(sum(1/o for o in fair),1.)
        assert np.isclose(sum(1/o for o in book),margin)

def test_push_free_halfpoint_restriction():
    x=np.arange(70,120).astype(float)
    assert sum(threshold_counts(x,88.5))==1.
    with pytest.raises(ValueError): threshold_counts(x,89)

def test_main_board_bookmaker_probabilities():
    tables=build_trader_board(fixture_sim())
    assert set(tables)=={'teams','match','most','handicap'}
    for name in ('teams','match'):
        for _,g in tables[name].groupby(['Market','Line']):
            if name=='teams' or name=='match':
                assert np.isclose((1/g.Book_109).sum(),1.09)
    for _,g in tables['most'].groupby('Market'):
        assert np.isclose((1/g.Book_115).sum(),1.15)
        assert len(g)==3

def test_no_stale_market_definitions():
    sim=fixture_sim();sim['market_schema_version']='players_only_v1'
    with pytest.raises(ValueError):build_trader_board(sim)

def test_market_totals_use_both_teams():
    sim=fixture_sim()
    tbl=build_trader_board(sim)['match']
    row=tbl[(tbl.Market=='Match Rebounds') & (tbl.Side=='Over')].iloc[0]
    exp=float(np.mean(sim['home']['MARKET_REB']+sim['away']['MARKET_REB']))
    assert np.isclose(row.Mean,exp,atol=.01)

def test_three_way_ties_settled_exactly():
    sim=fixture_sim()
    sim['away']['PTS']=sim['home']['PTS'].copy()
    g=build_trader_board(sim)['most']
    x=g[g.Market=='Most Points']
    assert x[x.Side=='Tie'].P.iloc[0]==1
    assert np.isinf(x[x.Side=='Home'].Fair.iloc[0])
    assert np.isnan(x.Book_115).all()  # no manufactured odds for an impossible side

def test_external_ev_math_is_shadow():
    d=evaluate_external_odds(.60,1.90)
    assert d['Model_EV']==pytest.approx(.14)
    assert 'SHADOW' in d['Status']
