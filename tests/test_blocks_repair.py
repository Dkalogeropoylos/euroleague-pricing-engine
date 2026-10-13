"""Protect EuroLeague BLK parser and healing old exports. Offline only."""
import pandas as pd
from providers.euroleague import numeric_row
from sync_cloud import sync
from test_sync_cloud import OfflineFakeProvider


def test_euroleague_blocks_favour_is_parsed():
    assert numeric_row({'blocksFavour': 3})['BLK'] == 3.0
    assert numeric_row({'blocksFavour': 0})['BLK'] == 0.0


def test_repair_reimports_old_games_with_missing_blocks(tmp_path):
    provider = OfflineFakeProvider(tmp_path)
    exports = tmp_path / 'exports'
    a = sync([2026], export_dir=exports, provider=provider)
    assert a['new_games'] == 1 and a['repaired_games'] == 0
    old_team = pd.read_csv(exports / 'team_games.csv')
    old_player = pd.read_csv(exports / 'player_games.csv')
    old_team['BLK'] = float('nan')
    old_player['BLK'] = float('nan')
    old_team.to_csv(exports / 'team_games.csv', index=False)
    old_player.to_csv(exports / 'player_games.csv', index=False)
    b = sync([2026], export_dir=exports, provider=provider)
    assert b['new_games'] == 0 and b['repaired_games'] == 1
    t = pd.read_csv(exports / 'team_games.csv')
    p = pd.read_csv(exports / 'player_games.csv')
    assert len(t) == len(old_team) and len(p) == len(old_player)
    assert t.BLK.notna().all() and p.BLK.notna().all()
    assert not t.duplicated(['GAME_ID', 'TEAM_ID']).any()
    assert not p.duplicated(['GAME_ID', 'PLAYER_ID']).any()
