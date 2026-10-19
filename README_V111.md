# EuroLeague v1.1.1 — targeted shadow accuracy/traceability hotfix

This is an **incremental patch for the already uploaded full v1.1 repo**, not a fresh GitHub repository. Keep all existing v1.1 scripts and workflows. It does **not** change historical model coefficients or the 16-game rebound prior because reducing that prior worsened the 2024-2025 historical DREB evaluation.

Install by copying these two files on top of existing paths:
- `core/matchup_v11.py`
- `app_trader_panel_v11.py`

Add:
- `tests/test_v111_mean_and_prior_audit.py`

Then GitHub Commit, Run workflow `daily` only if data refresh is required; normally simply reboot Streamlit Cloud and rerun the v1.1 Trader Board. The UI adds a strong-shrinkage source audit and non-pricing sensitivity table. Historical residuals now center to zero before paired Monte Carlo resampling.

Check: `python -m pytest -q` (51 tests passed on verified data). All markets remain `HOLD`; these are NOT bookmaker-ready fair prices. See `EUROLEAGUE_V111_SHRINKAGE_MC_AUDIT.md` separately for the empirical evidence and team matchup comparisons.
