"""Market settlement truth vs official technical box-score events.

The team's market OREB, DREB, REB and turnovers count ONLY the sum of
individual players' credited box-score statistics.  Official totals can include
team-only events, so retain them separately for analytical use.

This module has NO network access and NEVER guesses missing individual numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_STATS = ("OREB", "DREB", "REB", "TO")
CONTROL_STATS = ("PTS", "FG2M", "FG2A", "FG3M", "FG3A", "FTM", "FTA", "AST", "STL", "BLK")
KEYS = ["GAME_ID", "TEAM_ID"]
REQUIRED = set(KEYS) | set(MARKET_STATS) | set(CONTROL_STATS)
MARKET_SCHEMA_VERSION = "players_only_v1"


class MarketTruthError(ValueError):
    """Fail closed instead of training on a different settlement definition."""


def _valid_integer_stats(df: pd.DataFrame, cols: tuple[str, ...], label: str) -> pd.DataFrame:
    for col in cols:
        num = pd.to_numeric(df[col], errors="coerce")
        if num.isna().any() or (~np.isfinite(num.to_numpy(dtype=float))).any():
            raise MarketTruthError(f"{label}: {col} has missing/non-numeric values")
        if (num < 0).any() or not np.allclose(num.to_numpy(), np.rint(num.to_numpy()), atol=1e-8):
            raise MarketTruthError(f"{label}: {col} has negative or fractional event counts")
        df[col] = num.astype("int64")
    return df


def attach_market_truth(team_games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Return enriched team rows with MARKET_*, OFFICIAL_* and TEAM_ONLY_* fields.

    Checks complete player coverage via all independently tallyable shot/point/
    defensive event box-score fields. Never uses official REB/TO as a fallback.
    Official POSS_RAW is kept unchanged and is NOT the player-sum TO/OREB pace.
    """
    if team_games.empty or player_games.empty:
        raise MarketTruthError("Need both nonempty team and player game exports")
    t, p = team_games.copy(), player_games.copy()
    for name, frame in (("team", t), ("player", p)):
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise MarketTruthError(f"{name}: missing {sorted(missing)}")
        for k in KEYS:
            if frame[k].isna().any():
                raise MarketTruthError(f"{name}: missing {k} identity")
            frame[k] = frame[k].astype(str).str.strip()
        if (frame[KEYS] == "").any().any():
            raise MarketTruthError(f"{name}: blank game/team identity")
        _valid_integer_stats(frame, MARKET_STATS + CONTROL_STATS, name)
    if t.duplicated(KEYS).any():
        raise MarketTruthError("Duplicate GAME_ID/TEAM_ID team row")
    if not (t.groupby("GAME_ID").size() == 2).all():
        raise MarketTruthError("Every game must have exactly two team rows")
    if "PLAYER_ID" not in p or p.PLAYER_ID.isna().any() or p.duplicated(KEYS + ["PLAYER_ID"]).any():
        raise MarketTruthError("Missing or duplicate player identity per team/game")
    if (p.REB != p.OREB + p.DREB).any():
        raise MarketTruthError("Player individual REB != OREB + DREB")
    if (t.REB != t.OREB + t.DREB).any():
        raise MarketTruthError("Official team REB != OREB + DREB")

    sums = p.groupby(KEYS, sort=False)[list(MARKET_STATS + CONTROL_STATS)].sum(min_count=1)
    counts = p.groupby(KEYS, sort=False).size()
    tkey = pd.MultiIndex.from_frame(t[KEYS])
    if not sums.index.is_unique or len(sums) != len(t) or not tkey.isin(sums.index).all():
        raise MarketTruthError("Player/team game groups do not correspond one-to-one")
    if (counts < 5).any():
        raise MarketTruthError("Fewer than 5 player box scores for a team game")
    aligned = sums.reindex(tkey)
    if aligned.isna().any().any():
        raise MarketTruthError("Missing player credited statistics; cannot infer from team totals")
    # Independent reconciliation: these are expected to match exactly in this feed.
    for c in CONTROL_STATS:
        official = t[c].to_numpy(dtype=int)
        player_sum = aligned[c].to_numpy(dtype=int)
        if not np.array_equal(official, player_sum):
            mismatches = np.where(official != player_sum)[0]
            i = int(mismatches[0])
            raise MarketTruthError(f"Player boxscore coverage mismatch {c} game {t.iloc[i].GAME_ID}, "
                                   f"team {t.iloc[i].TEAM_ID}: official {official[i]} / players {player_sum[i]}")
    for c in MARKET_STATS:
        official = t[c].to_numpy(dtype=int)
        credited = aligned[c].to_numpy(dtype=int)
        if np.any(credited > official):
            i = int(np.where(credited > official)[0][0])
            raise MarketTruthError(f"Players {c} exceed official team {c}: "
                                   f"game {t.iloc[i].GAME_ID}, team {t.iloc[i].TEAM_ID}")
        t[f"OFFICIAL_{c}"] = official
        t[f"MARKET_{c}"] = credited
        t[f"TEAM_ONLY_{c}"] = official - credited
    if not np.array_equal(t.MARKET_REB.to_numpy(), (t.MARKET_OREB + t.MARKET_DREB).to_numpy()):
        raise MarketTruthError("Market REB identity failure")
    if "POSS_RAW" in t.columns:
        expected = (pd.to_numeric(t.FG2A) + pd.to_numeric(t.FG3A) + 0.44*pd.to_numeric(t.FTA)
                    - t.OFFICIAL_OREB + t.OFFICIAL_TO)
        # PBP-adjusted possession estimates may be added later under a NEW name.
        actual = pd.to_numeric(t.POSS_RAW, errors="coerce")
        if actual.isna().any() or (abs(actual - expected) > 0.011).any():
            raise MarketTruthError("POSS_RAW does not use official OREB/TO; do not use player-sum events for pace")
    t["MARKET_SCHEMA_VERSION"] = MARKET_SCHEMA_VERSION
    return t


def market_audit(team_games: pd.DataFrame) -> dict:
    """Lightweight audit for manifest/dashboard, not a model-quality certificate."""
    data = {"market_schema_version": MARKET_SCHEMA_VERSION, "team_game_rows": int(len(team_games))}
    for s in MARKET_STATS:
        col = "TEAM_ONLY_" + s
        if col not in team_games:
            raise MarketTruthError(f"Missing {col}; market target layer not initialized")
        vals = pd.to_numeric(team_games[col], errors="coerce")
        if vals.isna().any():
            raise MarketTruthError(f"Missing {col} values")
        data[s] = {"official_minus_players_total": int(vals.sum()),
                   "team_games_with_team_only_events": int((vals > 0).sum()),
                   "average_official_minus_players": round(float(vals.mean()), 4)}
    return data
