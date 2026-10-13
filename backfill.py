"""Resumable cache-first downloader. Usage: python backfill.py --seasons 2024 2025 --max-games 5"""
import argparse
from pathlib import Path
import pandas as pd
from providers.euroleague import EuroLeagueProvider


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
    for filename, chunks in (("team_games.csv", teams), ("player_games.csv", players), ("data_audit.csv", audits)):
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        df.to_csv(dest / filename, index=False)
        print(filename, len(df), "rows")

if __name__ == "__main__":
    main()
