"""Fail-closed GitHub Actions release gate for official-inclusive shadow models."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from market_truth import MARKET_SCHEMA_VERSION


def main(base: Path = Path('data/euroleague')) -> None:
    ex = base / 'exports'; shadow = base / 'shadow_v06'
    mf = json.loads((ex / 'manifest.json').read_text(encoding='utf-8'))
    coeff = json.loads((ex / 'league_coefficients.json').read_text(encoding='utf-8'))
    report = json.loads((shadow / 'structural_v06_report.json').read_text(encoding='utf-8'))
    team = pd.read_csv(ex / 'team_games.csv', low_memory=False)
    if mf.get('market_schema_version') != MARKET_SCHEMA_VERSION:
        raise RuntimeError('Manifest settlement schema mismatch')
    if coeff.get('market_schema_version') != MARKET_SCHEMA_VERSION:
        raise RuntimeError('League coefficient settlement schema mismatch')
    if report.get('market_definition') != MARKET_SCHEMA_VERSION:
        raise RuntimeError('Structural model settlement schema mismatch')
    if report.get('status') != 'SHADOW_NOT_PRICING_READY':
        raise RuntimeError('Model status not shadow')
    if not mf.get('training_gate',{}).get('met'):
        raise RuntimeError('Training gate did not pass')
    if len(team) != 2 * mf.get('games',-1):
        raise RuntimeError('Manifest / validated team rows mismatch')
    if mf['games'] < 500 or len(set(team.SEASON)) < 3:
        raise RuntimeError('Historic dataset insufficient')
    for stat in ('OREB','DREB','REB','TO'):
        needed=[f'MARKET_{stat}',f'OFFICIAL_{stat}',f'PLAYER_CREDITED_{stat}',f'TEAM_ONLY_{stat}']
        if any(k not in team.columns for k in needed):
            raise RuntimeError(f'Missing reconciled {stat} fields')
        if not (team[f'MARKET_{stat}'] == team[stat]).all():
            raise RuntimeError(f'Market {stat} is not official-inclusive')
        if not (team[f'MARKET_{stat}'] == team[f'PLAYER_CREDITED_{stat}']+team[f'TEAM_ONLY_{stat}']).all():
            raise RuntimeError(f'Player/team-only reconciliation failed for {stat}')
    if not (team.MARKET_REB==team.MARKET_OREB+team.MARKET_DREB).all():
        raise RuntimeError('REB identity failure')
    for k in ('OREB','DREB','TO'):
        if k not in coeff.get('markets',{}):
            raise RuntimeError(f'League coefficient missing market {k}')
    for k in ('MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO'):
        if k not in report.get('metrics',{}):
            raise RuntimeError(f'Structural shadow holdout missing {k}')
    print(f'RELEASE_GATE_PASS: {mf["games"]} games; official_inclusive_v2; '
          f'shadow holdout {report["test_games"]} games. No fair odds enabled.')


if __name__ == '__main__':
    main()
