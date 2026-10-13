from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REQUIRED=('PACE','THREE_SHARE','TWO_PCT','THREE_PCT','FT_PCT','FTA_RATE','OREB_RATE','DREB_RATE')
def main():
    root=Path('data/euroleague')
    report=json.loads((root/'shadow_v08'/'discrimination_v08_report.json').read_text())
    if report.get('market_definition')!='official_inclusive_v2' or report.get('status')!='SHADOW_NOT_PRICING_READY':
        raise ValueError('V08 schema/status mismatch')
    if report.get('no_same_day_leakage') is not True:
        raise ValueError('Missing embargo flag')
    if report['games']<500 or report['train_games']<300:
        raise ValueError('Too few verified EuroLeague matches')
    exports=root/'exports'/'team_games.csv'
    if not exports.exists():
        raise FileNotFoundError('Curated team games missing')
    df=pd.read_csv(exports,usecols=['GAME_ID','OT_FLAG'])
    regular_games=df.loc[~df.OT_FLAG.fillna(False).astype(bool),'GAME_ID'].nunique()
    if int(regular_games)!=report['games']:
        raise ValueError('Stale v0.8 report: game count disagrees with current exports')
    if set(report['next_game_challenger']['targets'])!=set(REQUIRED):
        raise ValueError('Incomplete coefficients')
    for k,v in report['next_game_challenger']['targets'].items():
        if v['kind'] not in ('league','v06','recent','season','carryover','multiscale'):
            raise ValueError('Invalid coefficient candidate '+k)
        if v['kind'] not in ('league','v06'):
            if len(v['names'])!=len(v['coefficients']):
                raise ValueError('Mismatched model parameter dimensions')
    print('V0.8 SHADOW REPORT VERIFIED. No betting approval.')
if __name__=='__main__':main()
