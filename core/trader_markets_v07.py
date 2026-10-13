"""Research-only odds presentation built on coherent v0.6 simulated events.

No probabilistic calibration claim. 109% two-way and 115% three-way
are PROPORTIONAL multiplication of normalized *implied* probabilities.
For all full O/U boards we require half-integer thresholds to eliminate pushes.
Three-way most uses exact ties, settlement measured at end of REGULATION.
This market does NOT include OT; v0.6 simulations are regulation-only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TWOWAY_OVERROUND = 1.09
THREEWAY_OVERROUND = 1.15
MARKET_SCHEMA_VERSION = 'official_inclusive_v2'
MARKETS = (
    ('PTS','Points',0),('FG3M','3PM',1),('FG3A','3PA',2),
    ('FG2M','2PM',3),('FG2A','2PA',4),('FTA','FTA',5),('FTM','FTM',6),
    ('MARKET_REB','Rebounds',7),('MARKET_OREB','Offensive rebounds',8),
    ('MARKET_DREB','Defensive rebounds',9),('AST','Assists',10),
    ('MARKET_TO','Turnovers',11),('STL','Steals',12),('BLK','Blocks',13),
)


def pricing_probabilities(probs, overround:float) -> tuple[list[float],list[float]]:
    """Return fair decimal odds and multiplicative margin odds; no clipping.
    Cannot price a zero-probability outcome with finite decimal odds.
    """
    arr=np.asarray(probs,dtype=float)
    if not np.isfinite(arr).all() or np.any(arr<0) or not np.isclose(arr.sum(),1,atol=1e-8):
        raise ValueError('Outcome probabilities must be finite, nonnegative, sum to 1')
    if overround < 1: raise ValueError('Overround cannot be less than 100%')
    fair=[float(1/p) if p>0 else float('inf') for p in arr]
    # Do NOT invent a positive mass for never-observed rare outcomes.
    # Also decimal odds <1 are invalid: a proportional overround can cause
    # this for very concentrated probabilities. Fail closed for that market.
    if np.any(arr==0) or np.any(overround*arr>=1.):
        quoted=[float('nan') for _ in arr]   # not quoteable; fair odds still visible
    else:
        quoted=[float(1/(overround*p)) for p in arr]
    return fair,quoted


def _r(a):
    v=np.asarray(a)
    if v.ndim!=1 or v.size<50 or not np.isfinite(v).all() or np.any(v<0):
        raise ValueError('Expected >=50 finite, nonnegative simulation draws')
    return v


def threshold_counts(draws, threshold:float):
    a=_r(draws)
    if not np.isclose((threshold*2)%2,1,atol=1e-8):
        raise ValueError('O/U thresholds must be half-integers for push-free settlement')
    over=float(np.mean(a>threshold));under=float(np.mean(a<threshold))
    if not np.isclose(over+under,1.,atol=1e-8):raise ValueError('Unexpected push / invalid counts')
    return over,under


def balanced_half_line(draws:np.ndarray, target_price:float=1.90) -> float:
    """Price a practical main half-point line nearest desired O price, without inventing a market line.
    Uses realized simulation quantiles (research preview) and a bounded local grid.
    """
    x=_r(draws)
    mid=float(np.median(x))
    center=np.floor(mid)+.5
    candidates=np.arange(max(.5,center-8),center+8.6,1.)
    ideal=1.0/(TWOWAY_OVERROUND*target_price)
    return float(min(candidates,key=lambda v:abs(np.mean(x>v)-ideal)))


def _smry(a):
    a=_r(a)
    return {'Mean':round(float(np.mean(a)),2),'Median':round(float(np.median(a)),2),
            'P10':round(float(np.quantile(a,.10)),1),'P90':round(float(np.quantile(a,.90)),1),
            'MC_SD':round(float(np.std(a,ddof=1)),2)}


def _two_rows(a,label,group,line=None):
    x=_r(a);line=balanced_half_line(x) if line is None else float(line)
    o,u=threshold_counts(x,line)
    fair,book=pricing_probabilities([o,u],TWOWAY_OVERROUND)
    common={'Market':label,'Group':group,'Line':line, **_smry(x), 'Status':'SHADOW / UNCALIBRATED'}
    return [{**common,'Side':'Over','P':o,'Fair':fair[0],'Book_109':book[0]},
            {**common,'Side':'Under','P':u,'Fair':fair[1],'Book_109':book[1]}]


def _three_rows(h,a,label,group):
    x,y=_r(h),_r(a)
    if len(x)!=len(y):raise ValueError('Home/away simulation lengths differ')
    p=[float(np.mean(x>y)),float(np.mean(x==y)),float(np.mean(x<y))]
    fair,book=pricing_probabilities(p,THREEWAY_OVERROUND)
    return [{'Market':label,'Group':group,'Side':side,'P':prob,'Fair':f,'Book_115':b,
             'Status':'SHADOW / UNCALIBRATED'} for side,prob,f,b in zip(('Home','Tie','Away'),p,fair,book)]


def build_trader_board(sim:dict, home_name='HOME',away_name='AWAY') -> dict[str,pd.DataFrame]:
    if sim.get('market_schema_version')!=MARKET_SCHEMA_VERSION:
        raise ValueError('Wrong settlement schema; pricing refused')
    h,a=sim['home'],sim['away']
    n=len(_r(h['PTS']))
    if len(_r(a['PTS']))!=n:raise ValueError('Mismatched lengths')
    team_rows=[];combined_rows=[];most_rows=[]
    for key,name,sort_order in MARKETS:
        hv,av=_r(h[key]),_r(a[key])
        # Per-team markets
        for side,label,arr in ((home_name,'Home',hv),(away_name,'Away',av)):
            for row in _two_rows(arr,f'{side} {name}','Team'):
                row['Order']=sort_order;row['Team']=label;team_rows.append(row)
        # Both teams combined statistic, including 40-minute game total points
        for row in _two_rows(hv+av, f'Match {name}', 'Match total'):
            row['Order']=sort_order;combined_rows.append(row)
        # 'Most' markets: direct count comparison, exact draw, regulation only
        for row in _three_rows(hv,av,f'Most {name}','3-Way Most'):
            row['Order']=sort_order;most_rows.append(row)
    # Handicap from the SAME paired simulations; home handicap sign convention
    margin=_r(h['PTS'])-_r(a['PTS'])
    handicap=[]
    # Anchor the handicap ladder to the simulation's median margin, not a
    # hardcoded spread around a pick'em game. This remains a shadow preview.
    main_handicap=float(np.floor(-np.median(margin))+.5)
    for hdcp in [main_handicap]+[main_handicap+delta for delta in (-6.,-4.,-2.,2.,4.,6.)]:
        m=margin+hdcp
        p=float(np.mean(m>0))
        fair,book=pricing_probabilities([p,1-p],TWOWAY_OVERROUND)
        handicap.append({'Market':'Regulation handicap','Handicap_Home':hdcp,
                         'Side':'Home','P':p,'Fair':fair[0],'Book_109':book[0],
                         'Status':'SHADOW / UNCALIBRATED'})
        handicap.append({'Market':'Regulation handicap','Handicap_Home':hdcp,
                         'Side':'Away','P':1-p,'Fair':fair[1],'Book_109':book[1],
                         'Status':'SHADOW / UNCALIBRATED'})
    t=pd.DataFrame(team_rows)
    t['TeamOrder']=t.Team.map({'Home':0,'Away':1})
    t['SideOrder']=t.Side.map({'Over':0,'Under':1})
    t=t.sort_values(['Order','TeamOrder','Line','SideOrder']).drop(columns=['TeamOrder','SideOrder'])
    b=pd.DataFrame(combined_rows)
    b['SideOrder']=b.Side.map({'Over':0,'Under':1})
    b=b.sort_values(['Order','Line','SideOrder']).drop(columns=['SideOrder'])
    m=pd.DataFrame(most_rows)
    m['SideOrder']=m.Side.map({'Home':0,'Tie':1,'Away':2})
    m=m.sort_values(['Order','SideOrder']).drop(columns=['SideOrder'])
    return {'teams':t.reset_index(drop=True),'match':b.reset_index(drop=True),
            'most':m.reset_index(drop=True), 'handicap':pd.DataFrame(handicap)}


def evaluate_external_odds(model_probability:float,market_odds:float):
    if not (0<model_probability<1) or market_odds<=1:
        raise ValueError('Invalid probability or odds')
    # Shadow only; EV only an arithmetic diagnostic, not validated value.
    return {'Break_even_p':1/market_odds,'Model_p':model_probability,
            'Model_EV':model_probability*market_odds-1,
            'Status':'SHADOW: MC not calibrated; EV not actionable'}
