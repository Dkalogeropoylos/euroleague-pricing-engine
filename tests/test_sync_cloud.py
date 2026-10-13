"""CI regression: cloud updates append only new validated games; no live network."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from providers.euroleague import EuroLeagueProvider
from sync_cloud import sync


class OfflineFakeProvider(EuroLeagueProvider):
    def __init__(self, tmp_path):
        super().__init__(root=tmp_path / 'raw', minimum_interval=0)
        self.network_game_reads = 0

    def games(self, year):
        assert year == 2026
        return [{'gameCode': 21, 'date': '2026-10-09T20:00:00Z',
                 'played': True, 'phaseType': 'RS',
                 'local': {'club': {'code': 'AAA', 'name': 'Alpha'}, 'score': 80},
                 'road': {'club': {'code': 'BBB', 'name': 'Beta'}, 'score': 80}}]

    def game_stats(self, year, game_code):
        self.network_game_reads += 1
        team = {'points': 80, 'fieldGoalsMade2': 20, 'fieldGoalsAttempted2': 38,
                'fieldGoalsMade3': 10, 'fieldGoalsAttempted3': 26,
                'freeThrowsMade': 10, 'freeThrowsAttempted': 14,
                'offensiveRebounds': 9, 'defensiveRebounds': 22,
                'assistances': 18, 'steals': 7, 'blocks': 3,
                'turnovers': 12, 'foulsCommited': 17, 'foulsReceived': 18}
        result = {}
        for side, label in [('local', 'AAA'), ('road', 'BBB')]:
            players = []
            for i in range(10):
                # Valid fake: individual totals really reconcile to team boxscore.
                stats = {}
                for stat, total in team.items():
                    if stat == 'points':
                        continue
                    stats[stat] = total // 10 + (1 if i < total % 10 else 0)
                stats['points'] = (2 * stats['fieldGoalsMade2'] +
                                   3 * stats['fieldGoalsMade3'] + stats['freeThrowsMade'])
                stats['timePlayed'] = 1200
                players.append({'player': {'person': {'code': f'{label}{i}', 'name': f'{label} {i}'},
                                           'position': 'G'}, 'stats': stats})
            result[side] = {'total': team, 'players': players}
        return result


def test_incremental_is_resumable_without_redownloading(tmp_path):
    provider = OfflineFakeProvider(tmp_path)
    exports = tmp_path / 'exports'
    a = sync([2026], export_dir=exports, provider=provider)
    assert a['new_games'] == 1 and a['issues'] == 0
    assert provider.network_game_reads == 1
    assert len(pd.read_csv(exports / 'team_games.csv')) == 2
    assert len(pd.read_csv(exports / 'player_games.csv')) == 20
    b = sync([2026], export_dir=exports, provider=provider)
    assert b['new_games'] == 0 and b['issues'] == 0
    assert provider.network_game_reads == 1
    assert len(pd.read_csv(exports / 'team_games.csv')) == 2
