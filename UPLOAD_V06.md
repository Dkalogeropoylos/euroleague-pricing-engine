# One-time GitHub upload — EuroLeague v0.6

1. Keep the current live app, data exports and GitHub history intact. Optional: create a backup branch before uploading.
2. Extract the downloaded `euroleague_v06_official_retrain_unified.zip` locally. Upload its files and subfolders into the repository root, **not** the ZIP file itself. Ensure GitHub shows `.github/workflows/euroleague-refresh.yml`, `market_truth.py`, `league_coefficients.py`, `sync_cloud.py`, `core/structural_v06.py`, `train_structural_v06.py`, `verify_release_v06.py`, and `app_structural_panel.py` at the expected paths. Don't overwrite `data/euroleague/exports/*.csv`.
3. Commit on main once.
4. Actions → EuroLeague Data Refresh → Run workflow → mode=`daily` (NOT bootstrap). The workflow uses current cached CSVs and recalculates both coefficient layers after migration.
5. Confirm success, including the step `Verify official-inclusive v0.6 release gate`. If any step is red, open that log before further actions.
6. Verify manifest at `data/euroleague/exports/manifest.json` says `official_inclusive_v2` and the files `league_coefficients.json` and `data/euroleague/shadow_v06/structural_v06_report.json` contain the matching schema. The `Structural SHADOW v0.6` Streamlit tab now has official-inclusive market definitions.

**NO need** to re-download all 1,031 games; **NO fair odds yet**. Do not reuse old v0.5 predictions for official-inclusive rebounds/turnovers.
