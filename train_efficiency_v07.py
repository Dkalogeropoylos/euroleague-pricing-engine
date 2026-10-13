"""Read-only independent points/pace cross-check fitted from historical EuroLeague."""
from pathlib import Path
import json, argparse
import pandas as pd
from core.structural_v06 import verified_games
from core.efficiency_benchmark_v07 import fit_efficiency_audit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('data/euroleague/exports'))
    ap.add_argument('--out',type=Path,default=Path('data/euroleague/shadow_v07'))
    z=ap.parse_args()
    team=pd.read_csv(z.data/'team_games.csv',low_memory=False)
    player=pd.read_csv(z.data/'player_games.csv',low_memory=False)
    d=verified_games(team,player)
    report=fit_efficiency_audit(d)
    z.out.mkdir(parents=True,exist_ok=True)
    path=z.out/'efficiency_benchmark_v07.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str))
    print('Saved independent efficiency shadow audit:',path)
    print('Model selected:', report['selection'],'prior=',report['prior_games'], 'alpha=',report['alpha'])
    print('Outer regulation total RMSE: model',round(report['outer_total_ppp_model_rmse_pts'],3),
        'league baseline',round(report['outer_total_league_rmse_pts'],3), 'using observed pace in evaluation ONLY')
if __name__=='__main__':main()
