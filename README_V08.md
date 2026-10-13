# EuroLeague v0.8 — Team State Challenger (SHADOW)

## What changed
The v0.7 Trader Board previously used v0.6 ML coefficients that compressed every matchup toward one league mean.

v0.8 **does not inflate coefficients artificially**. It learns pregame pace, 3PA shot allocation, 2P%, FT%, FTA rate, OREB rate and DREB rate from time-frozen histories. Training candidates include last five/season-to-date opponent offense and defense, plus last-season evidence when available. Selected candidate and its ridge alpha are determined using two growing chronological inner validation folds. If the reference v0.6 or league baseline beats alternatives, we retain it.

All data sources are prior to the requested calendar day; markets still use **official-inclusive_v2** team rebounds and turnovers.

## Validated on user dataset
1,031 games / 1,984 regulation team-game rows.
Training games: 777. Historical audit: 194 (previously examined; NOT a fresh untouched test).

| Statistic | v0.6 RMSE | v0.8 RMSE | Change |
|---|---:|---:|---:|
| Pace | 3.983 | 3.719 | 6.63% better |
| FG3A | 5.058 | 4.737 | 6.34% better |
| PTS | 10.479 | 10.273 | 1.96% better |
| REB official | 5.397 | 5.305 | 1.70% better |

Across 190 synthetic unordered pairings of 20 teams:
- Pace v0.6: 74.43–74.67; v0.8: 72.57–77.06
- Home FG3A v0.6: 24.51–28.35; v0.8: 22.17–33.60
- Home official REB v0.6: 33.97–34.73; v0.8: 33.42–35.98
- Regulation game total v0.6: 166.99–169.93; v0.8: 162.65–174.34

**Important:** OREB_RATE, DREB_RATE, and THREE_PCT did not win on the inner CV and still use league references. A bigger predicted team-to-team range does NOT prove superior accuracy. Rebounding is not solved.

## Upload to your EXISTING private GitHub repo
Do NOT create a new repository. Do NOT delete any stored `data/euroleague/exports` or old shadow reports.

1. Upload the contents of this ZIP preserving paths, especially `.github/workflows/euroleague-refresh.yml` and `.gitignore`.
2. Commit to `main`.
3. Actions → EuroLeague Data Refresh → Run workflow → `daily`, NOT `bootstrap`.
4. Confirm stages `Fit team-state discrimination SHADOW v0.8` and `Verify team-state discrimination v0.8` are green.
5. Confirm `data/euroleague/shadow_v08/discrimination_v08_report.json` exists in GitHub.
6. Refresh/reboot Streamlit if needed; choose `v0.8 team-state challenger` in the existing Trader Board. Compare with `v0.6 reference`.

Never publish API tokens, secrets, data provider credentials.

## Pricing warning
The Monte Carlo distributions are reused from v0.6 and **remain uncalibrated**, particularly rebound and free-throw tails. Fair odds, 109% O/U and 115% three-way in the UI are **research diagnostics only**, not production trader prices. No absence/rotation adjustment, no PBP classification, no validated post-selection heldout sample.

## Technical
New files:
- `core/discrimination_v08.py` — trainable frozen team-state features, models, simulation adapter.
- `train_discrimination_v08.py` — re-estimates coefficients using historical exports; no API fetch needed.
- `verify_discrimination_v08.py` — checks data/coefficients alignment.
- `tests/test_discrimination_v08.py` — new regression tests.

Updated:
- `app_trader_panel_v07.py` — select between v0.8 challenger and v0.6 reference.
- `.github/workflows/euroleague-refresh.yml` — train/verify and force-add only validated shadow JSON.
- `.gitignore` — allow shadow_v08 JSON commits.

This package does not replace `core/structural_v06.py`, the existing dataset, or the WNBA engine.
