"""Market settlement and historical leakage regression tests, offline only."""
from __future__ import annotations

import pandas as pd
import pytest

from market_truth import attach_market_truth, market_audit, MarketTruthError
from league_coefficients import clean_games, pregame_features
from test_sync_cloud import OfflineFakeProvider
from test_euroleague import mock_games


def _synthetic(tmp_path):
    p = OfflineFakeProvider(tmp_path)
    g = p.games(2026)[0]
    team, players, errors = p.normalize_game(g, p.game_stats(2026, 21), 'E', 2026)
    assert not errors
    return pd.DataFrame(team), pd.DataFrame(players)


def test_player_sum_market_truth_and_official_pace_separation(tmp_path):
    t, p = _synthetic(tmp_path)
    # Add team-only events WITHOUT crediting any player.
    ix = 0
    t.loc[ix, 'OREB'] += 3
    t.loc[ix, 'DREB'] += 2
    t.loc[ix, 'REB'] += 5
    t.loc[ix, 'TO'] += 1
    # Box-score possessions must use the official, not player-credited counts.
    t.loc[ix, 'POSS_RAW'] = t.loc[ix, 'FG2A'] + t.loc[ix, 'FG3A'] + .44*t.loc[ix, 'FTA'] - t.loc[ix, 'OREB'] + t.loc[ix, 'TO']
    m = attach_market_truth(t, p)
    assert m.loc[ix, 'MARKET_OREB'] == 9
    assert m.loc[ix, 'MARKET_DREB'] == 22
    assert m.loc[ix, 'MARKET_REB'] == 31
    assert m.loc[ix, 'MARKET_TO'] == 12
    assert m.loc[ix, 'TEAM_ONLY_OREB'] == 3
    assert m.loc[ix, 'TEAM_ONLY_DREB'] == 2
    assert m.loc[ix, 'TEAM_ONLY_TO'] == 1
    assert m.loc[ix, 'OFFICIAL_REB'] == 36
    assert m.loc[ix, 'POSS_RAW'] == t.loc[ix, 'POSS_RAW']
    trained = clean_games(m)
    assert trained.loc[ix, 'OREB'] == 9
    assert trained.loc[ix, 'DREB'] == 22
    assert trained.loc[ix, 'TO'] == 12
    assert trained.loc[ix, 'OFFICIAL_OREB'] == 12
    assert market_audit(m)['REB']['official_minus_players_total'] == 5


def test_incomplete_player_rows_fail_closed(tmp_path):
    t,p = _synthetic(tmp_path)
    with pytest.raises(MarketTruthError, match='coverage mismatch'):
        attach_market_truth(t, p.iloc[1:].copy())
    with pytest.raises(MarketTruthError, match='missing/non-numeric'):
        bad = p.copy()
        bad.loc[0,'OREB'] = None
        attach_market_truth(t, bad)


def test_unreconciled_official_team_export_must_not_be_used_for_ml():
    raw = mock_games(40).drop(columns=['MARKET_OREB'])
    with pytest.raises(ValueError, match='Unreconciled player-sum markets'):
        clean_games(raw)


def test_current_game_pace_target_cannot_enter_own_features():
    raw = mock_games(160)
    f0 = pregame_features(clean_games(raw))
    target_day = pd.to_datetime(f0.GAME_DATE, utc=True).dt.floor('D').unique()[2]
    mut = raw.copy()
    # Change ALL observed outcomes for this day, including the pace proxy and
    # market counts. The current-day pregame features MUST remain unchanged.
    day_mask = pd.to_datetime(mut.GAME_DATE, utc=True).dt.floor('D') == target_day
    mut.loc[day_mask,'POSS_RAW'] += 15
    mut.loc[day_mask,'FG3A'] += 8
    f1 = pregame_features(clean_games(mut))
    for key in ['POSS','PACE_BASE','PACE_TEAM_LOG','PACE_OPP_LOG','FG3A_BASE','FG3A_OFF','FG3A_DEF']:
        x = f0.loc[pd.to_datetime(f0.GAME_DATE, utc=True).dt.floor('D') == target_day,key]
        y = f1.loc[pd.to_datetime(f1.GAME_DATE, utc=True).dt.floor('D') == target_day,key]
        assert len(x) == len(y) and list(x) == pytest.approx(list(y)), key
    # Training label changes; proves the test did not accidentally freeze the game.
    x = f0.loc[pd.to_datetime(f0.GAME_DATE, utc=True).dt.floor('D') == target_day,'PACE_TARGET_ONLY']
    y = f1.loc[pd.to_datetime(f1.GAME_DATE, utc=True).dt.floor('D') == target_day,'PACE_TARGET_ONLY']
    assert (x.reset_index(drop=True) != y.reset_index(drop=True)).all()


def test_existing_cache_migrates_market_schema_without_api_game_download(tmp_path):
    from sync_cloud import sync
    provider = OfflineFakeProvider(tmp_path)
    export_dir = tmp_path / 'exports'
    a = sync([2026], export_dir=export_dir, provider=provider)
    assert a['new_games'] == 1
    first_count = provider.network_game_reads
    # Simulate an old validated export with no market columns.
    path = export_dir / 'team_games.csv'
    old = pd.read_csv(path)
    old = old.drop(columns=[c for c in old.columns if c.startswith(('MARKET_', 'TEAM_ONLY_', 'OFFICIAL_'))])
    old.to_csv(path, index=False)
    updated = sync([2026], export_dir=export_dir, provider=provider)
    assert updated['new_games'] == 0
    assert updated['migrated_market_schema'] is True
    assert provider.network_game_reads == first_count
    converted = pd.read_csv(path)
    assert converted['MARKET_REB'].notna().all()
    assert converted['MARKET_TO'].notna().all()
    import json
    manifest = json.loads((export_dir / 'manifest.json').read_text())
    assert manifest['market_schema_version'] == 'players_only_v1'
