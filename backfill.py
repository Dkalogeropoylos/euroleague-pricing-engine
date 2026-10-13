"""Resumable cache-first downloader. Usage: python backfill.py --seasons 2024 2025 --max-games 5"""
import argparse
from pathlib import Path
import pandas as pd
from providers.euroleague import EuroLeagueProvider
from market_truth import attach_market_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True, help="Starting years, e.g. 2024 2025")
    ap.add_argument("--competition", choices=["E", "U"], default="E")
    ap.add_argument("--max-games", type=int, default=None, help="5 is a smoke test; omit for season")
    ap.add_argument("--data-dir", default="data/euroleague")
    ap.add_argument("--allow-postseason", action="store_true")
    ap.add_argument("--non-strict", action="store_true", help="Keep invalid games for inspection ONLY")
    args = ap.parse_args()
    provider = EuroLeagueProvider(root=args.data_dir, competition=args.competition)
    teams, players, audits = [], [], []
    for year in args.seasons:
        pack = provider.load_season(year, max_games=args.max_games, strict=not args.non_strict,
                                    regular_only=not args.allow_postseason)
        print(pack["metadata"], "issues:", len(pack["audit"]))
        teams.append(pack["team"])
        players.append(pack["player"])
        audits.append(pack["audit"])
    dest = Path(args.data_dir) / "exports"
    dest.mkdir(parents=True, exist_ok=True)
    team_df = pd.concat(teams, ignore_index=True) if teams else pd.DataFrame()
    player_df = pd.concat(players, ignore_index=True) if players else pd.DataFrame()
    audit_df = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    # Same market settlement contract as the GitHub incremental updater.
    team_df = attach_market_truth(team_df, player_df)
    for filename, df in (("team_games.csv", team_df),
                         ("player_games.csv", player_df),
                         ("data_audit.csv", audit_df)):
        df.to_csv(dest / filename, index=False)
        print(filename, len(df), "rows")

if __name__ == "__main__":
    main()
