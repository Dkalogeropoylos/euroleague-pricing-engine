"""Estimate EuroLeague-specific attack/defence coefficients via chronological features.

This module DOES NOT ship invented league averages or manually selected market weights.
Run it only after validated game data has been downloaded. Matches never see their
own or future results when forming offense/defense/league reference features.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

STAT_KEYS = ["FG2A", "FG3A", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TO", "PF", "FD"]
MIN_TRAIN_ROWS = 70


def clean_games(raw: pd.DataFrame, exclude_ot: bool = True) -> pd.DataFrame:
    df = raw.copy()
    if df.empty:
        raise ValueError("No downloaded games. First run the provider's backfill.")
    expected = {"GAME_ID", "GAME_DATE", "TEAM_ID", "OPP_ID", "SEASON", "HOME_AWAY", "POSS_RAW", *STAT_KEYS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing source fields: {sorted(missing)}")
    if df.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate game/team rows")
    if not (df.groupby("GAME_ID").size() == 2).all():
        raise ValueError("Every completed game must contain exactly 2 team rows")
    df["GAME_DATE"] = pd.to_datetime(df.GAME_DATE, utc=True, errors="coerce")
    df = df.dropna(subset=["GAME_DATE"])
    if "OT_FLAG" in df.columns and exclude_ot:
        bad = df.groupby("GAME_ID")["OT_FLAG"].transform(lambda x: x.fillna(False).astype(bool).any())
        df = df.loc[~bad]
    df = df[(df.POSS_RAW >= 40) & (df.POSS_RAW <= 115)]
    good_game_ids = df.groupby("GAME_ID").filter(lambda x: len(x) == 2).GAME_ID.unique()
    df = df.loc[df.GAME_ID.isin(good_game_ids)].copy()
    for k in STAT_KEYS:
        df[k] = pd.to_numeric(df[k], errors="coerce")
        df["ALLOWED_" + k] = df.groupby("GAME_ID")[k].transform(lambda s: s.iloc[::-1].to_numpy())
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)


def pregame_features(data: pd.DataFrame, min_league_games: int = 20, prior_games: float = 8.0) -> pd.DataFrame:
    """Chronological matchup feature builder, no same-game future leakage.

    Reference prior is league possessions-weighted average over past 550 days.
    Team form state is current-season only; prior strength is empirical Bayes,
    8 league-average games until enough same-season information accumulates.
    """
    data = clean_games(data, exclude_ot=False) if "ALLOWED_FG2A" not in data else data.copy()
    league_hist = deque()  # (date, per-stat counts, possessions)
    team_hist = defaultdict(list)  # key (season,team) -> previous game rows
    results = []
    # Treat an entire calendar date as one embargoed block. At 18:00 we cannot
    # borrow a result from a game scheduled for 20:30, even if its game code is lower.
    data = data.copy()
    data["_asof_day"] = data.GAME_DATE.dt.floor("D")
    for asof_day, day_group in data.groupby("_asof_day", sort=True):
        while league_hist and (asof_day - league_hist[0][0]).days > 550:
            league_hist.popleft()
        exposures = sum(z[2] for z in league_hist)
        n_games = len(league_hist) // 2
        baseline = {s: sum(z[1][s] for z in league_hist) / exposures if exposures else np.nan for s in STAT_KEYS}
        avg_possessions = exposures / max(len(league_hist), 1)
        if n_games >= min_league_games and avg_possessions > 0:
            for _, r in day_group.iterrows():
                season = int(r.SEASON)
                team = str(r.TEAM_ID)
                opp = str(r.OPP_ID)
                row = {"GAME_ID": r.GAME_ID, "GAME_DATE": r.GAME_DATE,
                       "SEASON": season, "TEAM_ID": team, "OPP_ID": opp,
                       "HOME_AWAY": r.HOME_AWAY, "POSS": float(r.POSS_RAW),
                       "HOME": 1.0 if r.HOME_AWAY == "H" else 0.0,
                       "AWAY": 1.0 if r.HOME_AWAY == "A" else 0.0}
                hist_o = team_hist[(season, team)]
                hist_d = team_hist[(season, opp)]
                for stat in STAT_KEYS:
                    base = baseline[stat]
                    if not np.isfinite(base) or base <= 0:
                        continue
                    prior_exposure = prior_games * avg_possessions
                    off_expo = sum(h["POSS_RAW"] for h in hist_o)
                    def_expo = sum(h["POSS_RAW"] for h in hist_d)
                    off_count = sum(h[stat] for h in hist_o if pd.notna(h[stat]))
                    def_count = sum(h["ALLOWED_" + stat] for h in hist_d if pd.notna(h["ALLOWED_" + stat]))
                    off = (off_count + base * prior_exposure) / (off_expo + prior_exposure)
                    defense = (def_count + base * prior_exposure) / (def_expo + prior_exposure)
                    row[f"{stat}_BASE"] = base
                    row[f"{stat}_OFF"] = float(np.log(max(off / base, 1e-8)))
                    row[f"{stat}_DEF"] = float(np.log(max(defense / base, 1e-8)))
                    row[f"{stat}_Y"] = float(r[stat]) if pd.notna(r[stat]) else np.nan
                results.append(row)
        # Commit results only AFTER every match on the date has its predictions.
        for _, r in day_group.iterrows():
            obj = r.to_dict()
            league_hist.append((asof_day, {s: float(r[s]) if pd.notna(r[s]) else 0.0 for s in STAT_KEYS}, float(r.POSS_RAW)))
            team_hist[(int(r.SEASON), str(r.TEAM_ID))].append(obj)
    return pd.DataFrame(results)


def estimate_coefficients(df: pd.DataFrame, holdout_fraction: float = 0.20) -> dict:
    """Nested chronological validation: inner train selects penalty or baseline; outer holdout is untouched.

    For Poisson-style counts, error metric is exposure-normalized prediction RMSE
    on latest game dates; probabilities/market EV are NOT trained in this stage.
    """
    if not 0 < holdout_fraction < 0.45:
        raise ValueError("holdout_fraction must be in (0, .45)")
    base = clean_games(df)
    features = pregame_features(base)
    if len(features) < MIN_TRAIN_ROWS:
        raise ValueError(f"Insufficient complete chronological team-game samples: {len(features)}; need at least {MIN_TRAIN_ROWS}")
    game_order = features[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values(["GAME_DATE", "GAME_ID"]).copy()
    game_order["DAY"] = game_order.GAME_DATE.dt.floor("D")
    dates = sorted(game_order.DAY.unique())
    split_idx = int(len(dates) * (1 - holdout_fraction))
    if split_idx < 10 or len(dates) - split_idx < 3:
        raise ValueError("Not enough training and holdout dates for calibration")
    first_holdout_day = dates[split_idx]
    training_games = set(game_order.loc[game_order.DAY < first_holdout_day, "GAME_ID"])
    test_games = set(game_order.loc[game_order.DAY >= first_holdout_day, "GAME_ID"])
    if len(training_games) < 25 or len(test_games) < 10:
        raise ValueError("Not enough train/holdout games for calibration")
    result = {"method": "pregame exposure-normalized Ridge log-rate, date-embargoed chronological holdout",
              "feature_prior_games": 8, "league_history_days": 550,
              "holdout": {"training_game_count": len(training_games), "test_game_count": len(test_games),
                          "start": str(first_holdout_day)},
              "markets": {}}
    for s in STAT_KEYS:
        cols = [f"{s}_OFF", f"{s}_DEF", "HOME", "AWAY"]
        needed = cols + [f"{s}_BASE", f"{s}_Y", "POSS"]
        ok = features.dropna(subset=needed).copy()
        ok = ok[(ok[f"{s}_BASE"] > 0) & (ok.POSS > 0)]
        tr = ok[ok.GAME_ID.isin(training_games)]
        te = ok[ok.GAME_ID.isin(test_games)]
        if len(tr) < MIN_TRAIN_ROWS or len(te) < 18:
            result["markets"][s] = {"status": "INSUFFICIENT", "train_rows": len(tr), "test_rows": len(te)}
            continue
        # Inner time-block validation selects hyperparameters WITHOUT looking at outer holdout.
        train_g = tr[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values(["GAME_DATE", "GAME_ID"]).copy()
        train_g["DAY"] = train_g.GAME_DATE.dt.floor("D")
        inner_days = sorted(train_g.DAY.unique())
        inner_idx = int(len(inner_days) * 0.75)
        boundary_day = inner_days[inner_idx]
        inner_train_ids = set(train_g.loc[train_g.DAY < boundary_day, "GAME_ID"])
        inner_valid_ids = set(train_g.loc[train_g.DAY >= boundary_day, "GAME_ID"])
        inner_train = tr[tr.GAME_ID.isin(inner_train_ids)]
        inner_valid = tr[tr.GAME_ID.isin(inner_valid_ids)]
        if len(inner_train) < 40 or len(inner_valid) < 16:
            result["markets"][s] = {"status": "INSUFFICIENT_INNER_VALIDATION", "train_rows": len(tr), "test_rows": len(te)}
            continue

        def target_y(z):
            return np.log((z[f"{s}_Y"].to_numpy(float) + .5) /
                          (z.POSS.to_numpy(float) * z[f"{s}_BASE"].to_numpy(float) + .5))

        def expected_counts(model, z):
            neutral = z.POSS.to_numpy(float) * z[f"{s}_BASE"].to_numpy(float)
            if model is None:
                return neutral
            return neutral * np.exp(np.clip(model.predict(z[cols].to_numpy(float)), -0.75, 0.75))

        def rmse(model, z):
            return float(np.sqrt(np.mean((expected_counts(model, z) - z[f"{s}_Y"].to_numpy(float))**2)))

        inner_base_rmse = rmse(None, inner_valid)
        best_alpha = None
        best_inner_rmse = inner_base_rmse
        for alpha in (0.5, 5.0, 25.0, 100.0):
            trial = Ridge(alpha=alpha).fit(inner_train[cols].to_numpy(float), target_y(inner_train))
            score = rmse(trial, inner_valid)
            if score < best_inner_rmse:
                best_inner_rmse, best_alpha = score, alpha

        # Freeze the inner-selected specification, refit ONLY on all historical train games,
        # and evaluate exactly once on the untouched outer chronological holdout.
        final_model = (Ridge(alpha=best_alpha).fit(tr[cols].to_numpy(float), target_y(tr))
                       if best_alpha is not None else None)
        outer_rmse = rmse(final_model, te)
        outer_base_rmse = rmse(None, te)
        coeffs = [float(x) for x in final_model.coef_] if final_model is not None else [0.0]*4
        result["markets"][s] = {
            "status": ("BASELINE_WINS" if final_model is None else
                       ("FITTED" if outer_rmse < outer_base_rmse else "OUTER_HOLDOUT_FAIL")),
            "features": ["offense_form", "opponent_allowed_form", "home", "away"],
            "coefficients": coeffs, "intercept": float(final_model.intercept_) if final_model is not None else 0.0,
            "alpha": best_alpha, "inner_baseline_rmse": inner_base_rmse,
            "inner_winner_rmse": best_inner_rmse,
            "holdout_rmse": outer_rmse, "baseline_rmse": outer_base_rmse,
            "improvement_pct": 100 * (1 - outer_rmse / max(outer_base_rmse, 1e-6)),
            "train_rows": len(tr), "test_rows": len(te),
        }
    # Baselines by season: descriptive ONLY; don't accidentally treat them as predictions.
    baselines = {}
    for season, season_df in base.groupby("SEASON"):
        p = float(season_df.POSS_RAW.sum())
        baselines[str(int(season))] = {"team_games": int(len(season_df)), "mean_possessions_est": float(season_df.POSS_RAW.mean()),
                                       "per_100_possessions": {s: float(season_df[s].sum() / p * 100) for s in STAT_KEYS}}
    result["descriptive_league_baselines"] = baselines
    return result


def save_coefficients(source_csv: str | Path, output_json: str | Path) -> dict:
    out = estimate_coefficients(pd.read_csv(source_csv))
    dest = Path(output_json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
