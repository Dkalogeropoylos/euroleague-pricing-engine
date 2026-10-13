# EuroLeague Cloud v0.3 — Block repair and staged history

## Observed real user data (E2026 only)
- 40 games, 20 teams, 80 team rows, 959 player rows.
- All 80 team rows and 959 player rows had BLK null.
- Live EuroLeague game stat field uses `blocksFavour`, previously ignored by loader.
- The 40 prior games were skipped by incremental refresh, so the fix needs a repair mode.

## Changes
1. `providers/euroleague.py`: read `blocksFavour` first, fall back to older variants.
2. `sync_cloud.py`: automatically re-download games with missing BLK; replace only the old rows after validated incoming stats; never duplicate or overwrite healthy games. `repaired_games` recorded in manifest.
3. `.github/workflows/euroleague-refresh.yml`: add `season` dropdown for manual bootstrap. Run one season at a time: 2026 repair; 2025, 2024, 2023 historical. Daily run unchanged.
4. `tests/test_blocks_repair.py`: regression checks for blocks mapping and repair idempotence.

## Deploy
Upload the replacement files to the same paths in your repo (not a new repo). Commit all changes before rerunning Actions. Do NOT upload CSVs from your computer or delete existing exports.

## Next
Run bootstrap season 2026 to repair old rows, then one historical season per action. Verify green success and `manifest.json` seasons, BLK column non-null, and data audit. Numerical calibration is shadow-only until real holdout validation.
