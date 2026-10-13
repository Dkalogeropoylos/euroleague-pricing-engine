"""Train the experimental structural EuroLeague model from checked-in exports.

Explicitly read-only with respect to existing model/data artifacts. Writes ONLY
shadow files to data/euroleague/shadow_v05, never replaces production coeffs.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from core.structural_v05 import (verified_games,event_rate_rows,chronological_design,
                                 chronological_split,fit_factors,evaluate,refit_shadow_after_audit)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/euroleague/exports'))
    parser.add_argument('--out',type=Path,default=Path('data/euroleague/shadow_v05'))
    args=parser.parse_args()
    raw=pd.read_csv(args.data/'team_games.csv',low_memory=False)
    players=pd.read_csv(args.data/'player_games.csv',low_memory=False)
    data=verified_games(raw,players)
    factors=event_rate_rows(data)
    features=chronological_design(factors)
    train,test,cutoff=chronological_split(features)
    fitted=fit_factors(features,train,test)
    scores=evaluate(features,fitted,test)
    heldout=features[features.GAME_ID.isin(test)]
    by_season={str(int(year)): {'games': int(group.GAME_ID.nunique()),
                                 'metrics':evaluate(features,fitted,set(group.GAME_ID))}
               for year,group in heldout.groupby('SEASON')}
    report={'status':'SHADOW_NOT_PRICING_READY','market_definition':'players_only_v1',
            'game_count':int(len(data)//2),'train_games':len(train),'test_games':len(test),
            'cutoff_utc':cutoff,'date_embargo':True,'exclude_ot':True,
            'ft_rebound_opportunity_proxy':0.5,
            'limitations':['No player absence/lineup adjustment','No PBP rebound classification',
                'No schedule/rest adjustment','Possessions are paired boxscore proxies',
                'Variance and interval calibration not complete',
                'This one holdout must not be repeatedly tuned upon'],
            'coefficients':fitted,'next_game_shadow_coefficients':refit_shadow_after_audit(features,fitted), 'metrics':scores, 'outer_holdout_by_season':by_season,
            'training_data_through_utc':str(data.GAME_DATE.max())}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'structural_v05_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(f"Prepared {len(features)} team-game frozen snapshots, {len(train)} training games, {len(test)} held-out games.")
    for key in ['POSS','PTS','FG2A','FG3A','FTA','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK']:
        m=scores[key];print(f"{key:14s} RMSE={m['rmse']:7.3f} baseline={m['baseline_rmse']:7.3f} change={m['improvement_pct']:6.2f}% bias={m['bias']:+.3f}")
    print('Shadow report:',args.out/'structural_v05_report.json')
if __name__=='__main__':main()
