"""Standalone shadow calibration. For deployment, prefer the validated GitHub sync."""
import argparse
import json
from pathlib import Path
import pandas as pd
from market_truth import attach_market_truth, MARKET_SCHEMA_VERSION
from league_coefficients import estimate_coefficients

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/euroleague/exports/team_games.csv")
    ap.add_argument("--player-input", default="data/euroleague/exports/player_games.csv")
    ap.add_argument("--output", default="data/euroleague/exports/league_coefficients.json")
    args = ap.parse_args()
    teams = pd.read_csv(args.input, low_memory=False)
    if "MARKET_SCHEMA_VERSION" not in teams.columns or not (teams.MARKET_SCHEMA_VERSION == MARKET_SCHEMA_VERSION).all():
        players_path = Path(args.player_input)
        if not players_path.exists():
            raise FileNotFoundError(f"Market settlement requires player logs: {players_path}")
        teams = attach_market_truth(teams, pd.read_csv(players_path, low_memory=False))
    out = estimate_coefficients(teams)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for stat, value in out.get("markets", {}).items():
        print(stat, value.get("status"), "holdout improvement:",
              round(value.get("improvement_pct", 0), 2), "%")
    print("Saved SHADOW coefficients to", path, "; NOT approved for live pricing")
