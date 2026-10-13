# NEXT ACTION — one unified GitHub update

1. **DO NOT** bootstrap the seasons again. **DO NOT** touch the 1,031 stored games or any file under `data/euroleague/exports/`.
2. Download/unzip the V0.5 patch. Upload the **contents**, preserving directory structure (including hidden `.github` and `.gitignore`), to the existing EuroLeague private repo. The patch intentionally includes only changed/new files; do not delete any other existing source file.
3. Commit once. The project's normal `app_euroleague.py` remains the Streamlit entrypoint. Python 3.11 in GitHub Actions and Python 3.12 in Streamlit are both compatible with this code.
4. Actions → EuroLeague Data Refresh → `daily` once; check `Fit structural SHADOW v0.5 on complete history` and that all tests pass. It writes `data/euroleague/shadow_v05/structural_v05_report.json` without altering the old `league_coefficients.json`.
5. Open Streamlit and use the new `Structural SHADOW v0.5` tab. The user can select two EuroLeague teams and a future as-of date. This preview is not calibrated odds and ignores player availability.
6. Save `PROJECT_HANDOFF.md` somewhere you can attach to a new chat to continue without revisiting earlier work.

If GitHub's web upload flattens the relative paths, STOP and check that `core/structural_v05.py`, `.github/workflows/euroleague-refresh.yml`, and `app_structural_panel.py` are at the right locations before running Actions.
