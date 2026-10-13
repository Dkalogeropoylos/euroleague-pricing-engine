# EUROLeague pricing engine — handoff updated 2026-10-13 (v0.6)

## Most important current truth

**User explicitly corrected team-market settlement.** Official team offensive, defensive and total rebounds **INCLUDE TEAM-ONLY REBOUNDS**, and team turnovers **INCLUDE TEAM-ONLY TURNOVERS**. The previous `players_only_v1` rule and ALL its market forecasts, confidence metrics, and downstream prices are invalid for this use. Current intended contract is `official_inclusive_v2`. **Confirm actual sportsbook's settlement methodology before taking real prices.** Player-credited shares remain preserved for player props and steal opportunity attribution.

## Files delivered in this turn

- `euroleague_v06_official_retrain_unified.zip`: GitHub patch with a new unified official-inclusive market truth, revised league coefficients, structural v0.6 shadow, tests, migration and GitHub release gate.
- `euroleague_v06_validation_report.json`: genuine offline metrics on historic uploaded dataset. `status=SHADOW`, not fair odds.
- this handoff document.

## User infrastructure / previous milestones

Private GitHub repository created by user, Streamlit app deployed from `app_euroleague.py`, GitHub Actions daily+bootstrap. Raw EuroLeague API ingested to team/player CSVs. Successful historical bootstrap: 1,031 validated games in 2023,2024,2025,2026, one old extreme 4OT outlier missed but not a priority. Player blocks field mapping repaired. Previously deployed data contract `players_only_v1`; **this is superseded**.

## Critical decisions (keep stable)

1. Use EuroLeague game APIs, cached in GitHub CSVs. Avoid Streamlit performing historical API backfill or fitting 1000s of games on every rerun.
2. No same-day or same-game leakage in ML calibration. Same-season team form is empirical-Bayes shrunk to rolling league history. Outcomes after the game are never used as pregame inputs.
3. Separate `OFFICIAL_*`, `PLAYER_CREDITED_*`, `TEAM_ONLY_*`, and `MARKET_*`; `MARKET_* == OFFICIAL_* == PLAYER_CREDITED_* + TEAM_ONLY_*` for OREB,DREB,REB,TO.
4. Shot-count and point identities must hold; opponent steal opportunities limited by **credited turnovers**, regardless of official TO total.
5. Avoid double-counting overlaps of season/recent/H2H windows. Opponent-by-position and historical H2H must be residualized where possible. Never fabricate injury news or pretend rotations are currently integrated when they aren't.
6. Do not make a betting recommendation/fair price until future-time variance, correlation and distribution calibration has been proven. For later pricing, two-way margin default 9%, three-way margin 16% unless overridden.
7. Keep WNBA production untouched; EuroLeague remains separate shadow sandbox.

## What v0.6 REALLY implements

- Migration of 1,031 stored games without new network boxscore downloads. Running `daily` migrates and retrains even when zero new games. Old schema fails the new Streamlit/schema gate.
- `league_coefficients.py` Ridge calibrated for official-inclusive OREB/DREB/TO, plus existing markets FG2A/FG3A/FTA/AST/STL/BLK/PF/FD. League baseline wins allowed.
- `core/structural_v06.py`: official-inclusive market simulation and FGA/PTS identities. Official OREB is distinct from the credited part used as shot-renewal proxy. Official DREB is conditional on an imperfect FT/FG misses opportunity exposure; not falsely capped as though PBP were complete. `DREB_CREDIT` and `TO_CREDIT` learned separately. Player-credited turnovers still constrain steals.
- Deterministic holdout metrics and sample audit. `verify_release_v06.py` fails CI on stale schemas or data/coeff mismatches.
- Streamlit `Structural SHADOW v0.6` read-only team preview and audits. No bookmaker lines.

## Empirical offline results, actual uploaded 1,031 games

Structural evaluation uses 777 training matches and 194 later held-out matches, excluding OT. Per team-game RMSE (model / same-structure baseline):

- Pace 3.983/4.066; PTS 10.479/11.016; FG2A 5.571/6.077; FG3A 5.058/5.465; FTA 6.281/6.519
- **Official inclusive OREB** 3.571/3.601; **DREB** 4.054/4.165; **REB** 5.397/5.570; **TO** 3.315/3.417.
- AST 4.344/4.459; STL 2.489/2.515; BLK 1.646/1.648.
- REB and FTA biases and variance require further attention. These are NOT comparable to older players-only REB/TO evaluation.

Example current saved matchup 2026-10-13 Partizan (home)–Panathinaikos (away), SHADOW with no absence handling: OREB 11.19/11.14, DREB 23.17/23.27, REB 34.36/34.41, TO 11.25/12.81. **Not production lines.**

## Deployment instructions

Upload only the patch ZIP contents (including nested `.github` and `core`). Do not delete historic CSVs. Run Actions -> `EuroLeague Data Refresh` -> `daily`. The GitHub workflow should pass pre-update tests, retrain, run the v0.6 release gate, commit CSVs/coefficients/model report. Verify `manifest.market_schema_version`, `league_coefficients.market_schema_version` and `shadow_v06.structural_v06_report.market_definition` are all `official_inclusive_v2`. The older v0.5 shadow reports are stale.

## Tasks NOT completed: next evidence-based release

- Real PBP rebound tagging and free-throw sequence classification; team-only dead ball vs live distinction, unlinked FT events.
- Shot efficiency mixtures/player attempts and absence/rotation redistribution by position and role; player-specific shooting and foul-drawing effects.
- Schedule/rest/travel, game totals/handicap market reconciliation, possible OT/blowout scenarios.
- Multi-fold forward validation and per-market distribution and correlation calibration.
- Only then fair lines, fair odds, target EV and stakes; track subsequent results for model drift.

Do not ask user to re-explain prior decisions. Use this handoff and latest repository/CSV source, never resurrect obsolete players_only_v1 methodology.
