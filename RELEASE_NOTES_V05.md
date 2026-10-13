# EuroLeague Structural SHADOW v0.5 — single integrated release

**Status: EXPERIMENTAL — absolutely not ready for bookmaker pricing.**

## What has been implemented (tested against 1,031 real user-supplied games)

- The **players_only_v1** settlement contract is preserved unmodified: `MARKET_OREB`, `MARKET_DREB`, `MARKET_REB`, `MARKET_TO` are *only* the sums of credited player statistics; `OFFICIAL_*` remains distinct for game-possession analysis.
- A frozen UTC calendar-day embargo. Same-day games (and target games) never contribute to their own historical features.
- EuroLeague-only empirical-Bayes offense and opponent-allowed profiles. Team features use **current-season** matches with prior shrinkage toward league history, which avoids treating last year's transfers as the same team.
- Chronological training (777 games) and untouched outer test (194 games), hyperparameters selected on an earlier nested block.
- One **shared** predicted game possession count; turnover and FT opportunity rates; two-/three-point attempt allocation; shooting make probabilities.
- Opportunity-based offensive rebounds, with a separate player-credit fraction, generate extra attempts. Defensive credited rebounds depend on the opponent's unrecovered misses plus a **proxy**, not an exact identification, for FT misses that could be rebounded.
- Assists conditional on team field goals, steals conditional on opponent **credited** turnovers, blocks conditional on opponent missed 2PA.
- Randomized 40-minute, game-correlated simulation with guaranteed shot/point/credit-count identities in each draw.
- Independent **Structural SHADOW v0.5** read-only Streamlit tab; before showing a matchup, uses future as-of dates only. Results are *means*, no fair lines/odds.

## Historical holdout — compare *structural model* vs same structural baseline without extra opponent/home learned multipliers

| Field | Model RMSE | Baseline RMSE | RMSE reduction |
|---|---:|---:|---:|
| PACE | 3.983 | 4.066 | 2.05% |
| PTS | 10.170 | 10.499 | 3.13% |
| FG2A | 5.575 | 5.803 | 3.94% |
| FG3A | 4.977 | 5.349 | 6.96% |
| FTA | 6.281 | 6.519 | 3.65% |
| MARKET_OREB | 3.233 | 3.239 | 0.19% |
| MARKET_DREB | 3.861 | 3.885 | 0.63% |
| MARKET_REB | 4.997 | 5.022 | 0.49% |
| MARKET_TO | 3.242 | 3.339 | 2.93% |
| AST | 4.218 | 4.321 | 2.39% |
| STL | 2.489 | 2.515 | 1.03% |
| BLK | 1.647 | 1.644 | **-0.19%** |

Do not interpret small one-sample improvements as proof of real fair probability improvement. The baseline itself uses only pregame league rates, not perfect-information scores.

**Statistical caveat**: the 2026–27 season has only four games per team; the system deliberately applies strong shrinkage.

## Non-negotiable unresolved issues for the next gate

1. Exact free-throw rebound opportunities and lineup possessions need PBP; the 0.50 reboundable missed-FT coefficient is **an admitted heuristic**, not trained/calibrated or a guaranteed valid league value.
2. No schedule congestion, rest, double-week, venue-neutral correction, injured-player availability, minutes, or lineup role redistribution yet. Do **not** mistake the pregame output for a fully matchup- or absence-adjusted number.
3. Draw-level variance, cross-market dependence, OT exposure, confidence intervals, probability calibration, Brier/log-loss, and market closing-line historical tracking **have not yet been validated**. **Do not price bets from raw Monte Carlo tails.**
4. A single historical outer test was consulted once in this prototype. Freeze it; introduce newer chronological validation periods before further model changes. Never repeatedly tune by choosing winners based on that holdout.
5. `POSS_RAW` remains an official team box-score possession *proxy*, averaged between both game sides. It is not exact play-by-play.
6. PF/FD structural state has **not** been deployed. Build coupled foul event logic (technical/nonshooting etc.) after extracting PBP, not from a blind independent Poisson regression.
7. Team-by-position allowances require a role mapping / genuine lineup stints; the raw API position field is three broad groups, not a complete 5-position map.

## Controlled deployment

**Never overwrite `data/euroleague/exports/`**. Keep the existing app as entry point `app_euroleague.py`.

Upload every file from this patch to the **matching repository-relative path**, including the dot-prefixed `.github/workflows/euroleague-refresh.yml` and the `core` package. Your already-deployed Streamlit app will expose a new fifth tab; the baseline four tabs are preserved. The workflow runs the new `train_structural_v05.py` after the existing daily data refresh and commits the **shadow** report in `data/euroleague/shadow_v05`. It does not overwrite `league_coefficients.json` and does not touch WNBA.

Run **Actions → EuroLeague Data Refresh → daily** once, then examine `data/euroleague/shadow_v05/structural_v05_report.json` and the new fifth Streamlit tab. There is no need for another historical bootstrap.

The initial report in the patch is calculated from the four-season user-provided exports for reproducible smoke testing. GitHub will regenerate the report from its own committed exports.

**Simple smoke test**: in the new fifth tab, select Paris Basketball (`PRS`) and Virtus Bologna (`VIR`) for a future date. It should return consistent rows and mark results SHADOW. Team names/codes are read from the actual CSV, not hardcoded.
