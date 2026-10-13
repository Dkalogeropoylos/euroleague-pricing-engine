# EuroLeague Gate 0/1 — consolidated delivery

This upload is a **single replacement for the earlier v0.4 ZIP**, not a new development branch.
Apply once to the existing repository only after the historic bootstrap committed all files.

## Included
- `market_truth.py` verifies that team OREB/DREB/REB/TO for sportsbook markets equal sums of individual credited player events; official totals used separately for technical pace.
- `league_coefficients.py` fixes same-game pace exposure leakage; trained coefficients remain shadow-only.
- `providers/euroleague.py` validates team minutes against 200 + 25 * overtime periods (accepts the 300-minute four-overtime historical exception), never accepts arbitrary minute totals.
- `train.py` uses player-credited settlement truth even on an unmigrated historical team export.
- tests, app, sync, docs for one combined repo upload.

## Cloud deployment
1. Keep `data/euroleague/exports/` untouched; **DO NOT** upload stale raw CSVs.
2. Add/replace files in this ZIP, **preserving relative folder paths**.
3. Commit once.
4. GitHub -> Actions -> EuroLeague Data Refresh -> Run workflow -> `daily`.
5. `manifest.json` must contain `market_schema_version: players_only_v1`, `market_audit`, `training_gate.met: true`.
6. Check `data_audit.csv` for any `E2023_170` error; to ingest the missing 4OT game, run `bootstrap` for 2023 **after** code update; it will skip already-fetched games. If your workflow does not expose Season, its bootstrap may re-scan multiple seasons, but no previously validated games should be re-fetched.
7. All coefficients and predictions remain SHADOW; no moneyline or fair pricing enabled.

## Important independent audit facts
- Dataset as of 2026-10-13: 1031 games; 305 of expected 306 in 2023/24, 306/306 in 2024/25, 380/380 in 2025/26, and 40/40 currently in 2026/27.
- Missing E2023_170 = Real Madrid vs Anadolu Efes quadruple overtime, 130–126.
- Reconciliation verified on every existing game/team row.
- Home court neutral flags are always `False` in these exports and cannot substantiate neutral-court effects without separate venue data.
- Running train.py uses a chronological holdout, but the single holdout report is not a production-ready probabilistic calibration certificate.
