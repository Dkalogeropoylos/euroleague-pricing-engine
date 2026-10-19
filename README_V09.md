# EuroLeague v0.9 — Empirical Shrinkage Calibration (RESEARCH / SHADOW)

## Purpose

Do not manually widen team lines. This patch tests whether historical EuroLeague observations support distinct season-to-date shrinkage by statistic, using pseudo-game priors and independently regularized regression. **It does not authorize fair odds, change existing structural simulations, or replace the current Trader Board's bookmaker prices.**

## Main technical findings

- A single 4- or 8-game pseudo-prior is not optimal for every team statistic.
- One fixed alpha for all markets is not justified. Hyperparameters here are chosen for each count using two chronological validation windows.
- The old v0.8 selector compares an already full-train fitted `v06` candidate with models refitted within inner folds: that is not a fair comparison. This independent audit **does not use that candidate** in its parameter selection.
- Old `THREE_PCT`, `STL_RATE`, `OREB_RATE`, `DREB_RATE` largely fell back to league priors; `DREB` misses/opportunities cannot be identified correctly from box scores without play-by-play.
- Metric-specific independent count predictions are NOT a substitute for the coherent possession/shot/points/rebounds simulation. In particular, directly forecast REB need not equal separately forecast OREB + DREB.
- **Player OREB/DREB/REB/TO are NOT market targets.** Official EuroLeague team totals inclusive of team-only events are the required markets.
- Future-season roster changes/injuries and EuroLeague schedule/fatigue remain unmodelled.

## Study protocol (reproducible)

Input: 1,031 official EuroLeague games from 2023–2026, omit 39 games with OT; same-UTC-calendar-day data embargo. `70` earlier league team-game observations are required before modelling. League baseline uses a rolling 550-day historical window.

Each statistic has a dynamic offensive profile and opponent *allowed* profile. Candidate priors of 2/4/8/16/32 effective games and prior-year weight 0/0.5/1 are evaluated with alpha 1/10/100 and team/opponent/both/multiscale predictor sets. All feature values are constructed using only games strictly earlier than target day.

Pre-specified annual score windows: Jan–May 2025 and Jan–May 2026. Every fold fits the model on games strictly earlier than its validation start. Full realized counts are the target (rather than realized-exposure-adjusted rate scoring). The 40 already-inspected 2026–27 games (September/October 2026) form a descriptive forward-watch diagnostic, not an untouched holdout. Inner-fold winner's CV performance is optimistic due to hyperparameter selection; require a future untouched period before approval.

These `effective prior` settings are NOT causal interpretations, physical parameters or guarantees that the same team identity has the same roster season to season.

## Commands after merging the patch into the existing repo

`python -m pytest -q`

`python calibrate_shrinkage_v09.py`

`python verify_calibration_v09.py`

The standard GitHub `daily` workflow includes the two last commands and force-adds *only* the output JSON into `data/euroleague/shadow_v09`. No bootstrap is needed.

Existing `Trader Board SHADOW v0.7` will show an expansion panel **v0.9 independent count benchmark — DOES NOT CHANGE PRICES** once the report is present. The original v0.6/v0.8 selection and Monte Carlo prices are untouched. When troubleshooting a price, compare forecasts but do not mechanically blend them.

## Gate for future integration into actual pricing

1. Fix the v0.8 model selection CV so that reference models are refitted within each fold or restricted to historically frozen data.
2. Rebuild the structural layer with prior + coefficient selection embedded *within* joint predictions (pregame pace × shot allocation × shooting efficiency, with official rebounds/turnovers) and test its entire-game errors, not just independent count RMSE.
3. Add turnover and steal opportunity constraints; fit team rebounding from genuinely reboundable misses once PBP is reliable, preserving official-inclusive team settlement.
4. Test actual pregame injury/roster information, rest, double-round weeks, lineup turnover.
5. Use blocked rolling-origin periods for selection, plus new untouched post-release matches for approval; compare MAE, RMSE, bias and market-level calibration / Brier / log-score. Check MC 90% interval coverage for points, FTA, rebounds and totals.
6. Require no material regression in key markets before promoting any coefficients or probabilities; otherwise retain the old model.

## Relevant peer-reviewed foundations

- Kubatko J, Oliver D, Pelton K, Rosenbaum D (2007). *A Starting Point for Analyzing Basketball Statistics*. Journal of Quantitative Analysis in Sports 3(3). https://doi.org/10.2202/1559-0410.1070
- Terner Z & Franks A (2021). *Modeling Player and Team Performance in Basketball*. Annual Review of Statistics and Its Application 8. https://doi.org/10.1146/annurev-statistics-040720-015536
- Attard P, Suda D, Sammut F (2023). *Bayesian Hierarchical Modelling of Basketball Team Performance*. icSPORTS. https://www.scitepress.org/PublishedPapers/2023/121591/
- Gneiting T & Raftery AE (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA 102(477). https://doi.org/10.1198/016214506000001437
- Gneiting T, Balabdaoui F & Raftery AE (2007). *Probabilistic Forecasts, Calibration and Sharpness*. JRSSB 69(2). https://doi.org/10.1111/j.1467-9868.2007.00587.x
