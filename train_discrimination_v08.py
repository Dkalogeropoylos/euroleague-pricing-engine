"""Train v0.8 challenger on frozen day-embargo history; old v0.6 retained."""
from __future__ import annotations
from pathlib import Path
import argparse,json
import pandas as pd
from core.structural_v06 import verified_games,event_rate_rows,chronological_design,chronological_split
from core.discrimination_v08 import add_state_features,fit_challenger,compare_holdout,refit_challenger

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data/euroleague/exports'))
    p.add_argument('--v06',type=Path,default=Path('data/euroleague/shadow_v06/structural_v06_report.json'))
    p.add_argument('--out',type=Path,default=Path('data/euroleague/shadow_v08'))
    args=p.parse_args()
    v06=json.loads(args.v06.read_text())
    if v06.get('market_definition')!='official_inclusive_v2':
        raise RuntimeError('Wrong settlement schema. STOP.')
    d=verified_games(pd.read_csv(args.data/'team_games.csv'),
                     pd.read_csv(args.data/'player_games.csv'))
    ev=event_rate_rows(d)
    design=add_state_features(ev,chronological_design(ev))
    tr,te,cutoff=chronological_split(design)
    hist_model=fit_challenger(design,v06['coefficients'],tr)
    score=compare_holdout(design,v06['coefficients'],hist_model,te)
    current_model=refit_challenger(design,hist_model)
    report={
      'model_version':'0.8-team-state-challenger',
      'status':'SHADOW_NOT_PRICING_READY',
      'market_definition':'official_inclusive_v2',
      'games':int(d.GAME_ID.nunique()),'train_games':len(tr),'audit_games':len(te),
      'cutoff_utc':cutoff,
      'no_same_day_leakage':True,
      'training_data_through_utc':str(d.GAME_DATE.max()),
      'historical_challenger':hist_model,
      'next_game_challenger':current_model,
      'audit_against_v06':score,
      'restrictions':['Results reuse an already scrutinized historical holdout',
                       'Not independent untouched validation',
                       'Official rebound totals include team-only events',
                       'No availability, venue relocation, fatigue, PBP/FT rebound classification',
                       'Monte Carlo variance not calibrated; fair odds diagnostic only',
                       'Do not manually inflate coefficients to achieve line separation']
    }
    args.out.mkdir(parents=True,exist_ok=True)
    target=args.out/'discrimination_v08_report.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('v0.8 SHADOW:',target)
    print('Selected:',{k:v['kind'] for k,v in hist_model['targets'].items()})
    for k in ('POSS','PTS','FG3A','MARKET_OREB','MARKET_DREB','MARKET_REB'):
        v=score[k];print(f"{k:<16} v06={v['rmse_v06']:.3f} v08={v['rmse_v08']:.3f} change={v['change_pct']:+.2f}%")
if __name__=='__main__':main()
