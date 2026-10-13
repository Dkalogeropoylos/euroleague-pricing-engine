# EuroLeague Pricing Engine — v0.6 official-inclusive unified correction

**Status: SHADOW ONLY. No production fair odds.** Release date: 2026-10-13.

## Why this release exists

The user corrected the target definition for team betting markets: **official team OREB, DREB, REB and TO include team-only events**, not just the sum credited to individual players. The prior `players_only_v1` contract is INVALID for the intended team markets. This patch replaces the entire contract, coefficient labels, technical model, historical holdout and UI in one version.

## What has been tested

- 19/19 automated offline tests pass.
- 1,031 historical actual games, 2,062 official team-games and 24,535 individual player-game lines reconciled.
- Realistic offline legacy-to-new migration test with `new_games=0`, full historical CSVs and stale legacy coefficient file: PASS; `league_coefficients.json` regenerated with `official_inclusive_v2`.
- Structural shadow uses regulation games and 777 training / 194 chronological holdout games. Some historic observations are only used to form prior features; OT excluded from model training. All four stored seasons remain available for data analysis.
- `verify_release_v06.py` ensures consistent metadata + reconstructed official markets + two refreshed coefficient files. A bad schema causes the GitHub action to FAIL before commit.

## Do not conflate market settlement with opportunity physics

`MARKET_OREB`, `MARKET_DREB`, `MARKET_REB`, `MARKET_TO` now equal official team box-score `OREB`, `DREB`, `REB`, `TO`. The corresponding `PLAYER_CREDITED_*` and `TEAM_ONLY_*` are retained and must sum to the market for each team/game.

- Pace uses `POSS_RAW`, reconstructed from official counts. Do not use the player-only market targets to estimate pace.
- **STL** opportunities are constrained by opponent *player-credited* turnovers, never all official TO.
- **OREB**: credited OREB is an approximate renewal proxy for additional field-goal attempts; team-only OREB is counted in markets but does NOT necessarily renew a possession. To do better, parse actual play-by-play.
- **DREB**: miss + FT proxy is an exposure for training, NOT a hard physical cap. Some actual official DREB exceed that imperfect proxy. The shadow simulator uses a conditional Poisson rather than impossible binomial truncation, and does NOT claim that its variance is calibrated.
- Core scoring and count identities are tested in expected values and draws: `REB = OREB + DREB`, `PTS = 2*2PM + 3*3PM + FTM`, `FGM <= FGA`, `STL <= opposing credited TO`.

## Models and safety

`league_coefficients.py`: chronological Ridge/regularization plus baseline selection, retrained on **official-inclusive** rebound and turnover labels; also already includes PF/FD. `core/structural_v06.py`: separate structural offense/defense factor selection plus joint shot-rebound shadow simulation, retrained from history. The previous v0.5 training entry point is disabled on purpose.

Unresolved research work from v0.5/v0.6 plan, NOT shipped as completed here: actual lineup/injury/rotation state; rest and travel; player-level 2P%/3P% skill mixture; possession-accurate PBP FT/technical/dead-ball rebounds; production variance/correlation calibration; multiple outer folds on untouched data; bookmaker margin/fair odds. These cannot honestly be marked finished just because team-market settlements were repaired.

## Deploy safely

1. Back up or tag the currently functioning commit. Do not delete any CSV.
2. Extract `euroleague_v06_official_retrain_unified.zip`. Upload its CONTENTS to the existing GitHub repository on `main`, preserving `.github/workflows/`, `core/`, and `tests/`, replacing same-name files.
3. GitHub Actions -> `EuroLeague Data Refresh` -> `Run workflow` -> `daily`. **No bootstrap** required; even with `new_games=0`, the schema migration and coefficient regeneration trigger automatically.
4. The Actions run must pass `Run tests before data update`, `Fit structural SHADOW v0.6`, `Verify official-inclusive v0.6 release gate`, and `Commit...`. On failure, check the failing log. Don't accept a green Streamlit page alone.
5. `data/euroleague/exports/manifest.json` must show `market_schema_version: official_inclusive_v2`, `training_gate.met: true` and 1,031+ games; `league_coefficients.json` and `data/euroleague/shadow_v06/structural_v06_report.json` must both declare official-inclusive.
6. Open Streamlit; confirm the `Structural SHADOW v0.6` tab displays official-inclusive market labels. It may take a re-open/reboot to see the newly deployed code.

The old `shadow_v05` report can remain as an archival artifact but **must never be used** for the new market. New software exclusively references `shadow_v06`. The old `train_structural_v05.py` and `core.structural_v05` are disabled deliberately.

## Genuine next step, not silently implemented

Before turning on betting prices: derive reboundable FT and live/dead ball events from EuroLeague PBP; redesign DREB joint/extra rebound opportunity distribution; integrate player availability/minutes/usage; run rolling-origin variance and correlation calibration; validate on distinct future holdout games. For user intent, do not make per-match hand-tuned adjustments.
