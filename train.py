"""Usage: python train.py --input data/euroleague/exports/team_games.csv"""
import argparse
from league_coefficients import save_coefficients

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/euroleague/exports/team_games.csv")
    ap.add_argument("--output", default="data/euroleague/exports/league_coefficients.json")
    args = ap.parse_args()
    out = save_coefficients(args.input, args.output)
    for s, v in out["markets"].items():
        print(s, v["status"], f"holdout improvement {v.get('improvement_pct', 0):.1f}%")
    print("Saved", args.output)
