import json
import numpy as np
import pandas as pd
from pathlib import Path

from providers.euroleague import EuroLeagueProvider, minutes, numeric_row
from league_coefficients import estimate_coefficients, clean_games, pregame_features, STAT_KEYS


def team_stats(pts=80):
    # 20 x 2pt + 10 x 3pt + 10 FTM = 80 points
    return {"points": pts, "fieldGoalsMade2": 20, "fieldGoalsAttempted2": 38,
            "fieldGoalsMade3": 10, "fieldGoalsAttempted3": 26,
            "freeThrowsMade": 10, "freeThrowsAttempted": 14,
            "offensiveRebounds": 9, "defensiveRebounds": 22, "assistances": 18,
            "steals": 7, "blocks": 3, "turnovers": 12, "foulsCommited": 17, "foulsReceived": 18}


def sample_game():
    game = {"gameCode": 1, "date": "2026-10-01T18:30:00Z", "played": True,
            "phaseType": "RS", "local": {"club": {"code": "ABC", "name": "Alpha"}, "score": 80},
            "road": {"club": {"code": "XYZ", "name": "Beta"}, "score": 80}}
    player_stats = dict(team_stats(), timePlayed=1200)
    payload = {"local": {"total": team_stats(), "players": [{"player": {"person": {"code": "001", "name": "John Doe"}, "position": "G"}, "stats": player_stats}]},
               "road": {"total": team_stats(), "players": []}}
    return game, payload


def test_minutes():
    assert minutes(1200) == 20.0
    assert abs(minutes("18:08") - (18 + 8/60)) < 1e-9


def test_normalization_and_score_reconciliation():
    game, payload = sample_game()
    teams, players, issues = EuroLeagueProvider.normalize_game(game, payload, "E", 2026)
    assert not issues or all("minutes sum atypical" in issue for issue in issues)
    assert len(teams) == 2 and len(players) == 1
    assert teams[0]["POSS_RAW"] == 64 - 9 + 12 + .44 * 14
    assert players[0]["PLAYER_ID"] == "E:001"
    assert players[0]["MINUTES"] == 20
    assert teams[0]["OPP_FG3A"] == 26
    assert teams[0]["FG3A"] + teams[0]["FG2A"] == teams[0]["FGA"]


def test_cache_mode_no_network(tmp_path):
    e = EuroLeagueProvider(root=tmp_path)
    p = tmp_path / "E" / "E2026" / "schedule_0.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"total": 1, "data": [sample_game()[0]]}))
    assert e.games(2026)[0]["gameCode"] == 1


def mock_games(num=160):
    rng = np.random.default_rng(777)
    rows = []
    names = [f"T{i:02d}" for i in range(12)]
    for i in range(num):
        home = names[i % 12]
        away = names[(i // 12 + i % 12 + 1) % 12]
        if home == away:
            away = names[(i + 5) % 12]
        date = pd.Timestamp("2024-10-01", tz="UTC") + pd.Timedelta(days=i//5)
        for t, opp, loc in ((home,away,"H"),(away,home,"A")):
            row = {"GAME_ID": f"E2024_{i+1}", "GAME_DATE":date.isoformat(),
                   "TEAM_ID": t, "OPP_ID":opp, "SEASON":2024, "HOME_AWAY": loc,
                   "POSS_RAW": 69.0, "OT_FLAG":False}
            for s in STAT_KEYS:
                avg = {"FG2A":39,"FG3A":25,"FTA":18,"OREB":10,"DREB":23,"AST":18,
                       "STL":6,"BLK":2,"TO":12,"PF":19,"FD":19}[s]
                row[s] = int(rng.poisson(avg))
            rows.append(row)
    return pd.DataFrame(rows)


def test_calibration_fits_only_pre_game_evidence():
    data = mock_games()
    clean = clean_games(data)
    features = pregame_features(clean)
    assert len(features)>75
    # Two sides of a given game receive the same frozen pregame reference.
    grp = features.groupby("GAME_ID")
    for _, pair in list(grp)[:3]:
        assert pair["FG3A_BASE"].nunique() == 1
    fitted = estimate_coefficients(data)
    assert "FG3A" in fitted["markets"]
    assert fitted["holdout"]["test_game_count"] > 0
    assert fitted["markets"]["FG3A"]["status"] in {"FITTED", "BASELINE_WINS", "OUTER_HOLDOUT_FAIL"}


def test_duplicate_team_games_rejected():
    x = mock_games(10)
    with np.testing.assert_raises(ValueError):
        clean_games(pd.concat([x,x.iloc[[0]]]))
