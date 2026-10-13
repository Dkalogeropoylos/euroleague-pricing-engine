# EuroLeague v0.7 — Trader Board SHADOW + Independent Totals Check

**Experimental output only. THIS IS NOT A PRODUCTION PRICING ENGINE.** Historical 90% interval calibration fails for several markets; no actionable betting edge is certified.

## What this delivers now

1. Keeps existing **1,031 EuroLeague games** and the `official_inclusive_v2` contract. Official team OREB, DREB, total rebounds and turnovers **include** team-only events. Player-credit stats remain separate.
2. Reuses v0.6 coherent possession–shot–score MC (points identity, shared possessions, rebounding coupling). No substitute independent Poisson points/REB generator.
3. Generates easy-to-scan **Team O/U**, **combined match totals**, **3-way Most** markets and **regulation handicaps**. Lists main half-point lines, P, fair odds and margin-included odds. Search/select custom lines, enter external prices to compute **shadow** EV, download CSVs.
4. Fixed two-way **109%**, three-way **115%** overround using proportional scaling. All quoted probabilities add to 109%/115% before display rounding. Does **not** quote book prices when zero MC probability or when direct margin would yield decimal odds <1.00.
5. Fits an independent *pregame points-per-possession* model: league PPP + season-specific recent offense + opponent-allowed defensive PPP + home. Inner chronological validation selected the prior-equivalent game count (from {2,4,8,16}) and Ridge alpha (from {2,15,80}); does not assign arbitrary fixed offense/defense weights. The **future** coefficient refit is separate from the historical evaluation coefficients.
6. Compares game total from v0.6 structural shot process with pregame pace × learned offense/defense efficiency, without forcing an untested blend. Publishes historic benchmark RMSE/bias from paired pregame model.
7. Re-evaluates existing MC 90% intervals. This is a **gate/diagnostic only**, not distribution calibration. A fail still produces a research board with conspicuous warnings and keeps pricing non-production.

## Real historical testing (snapshot 2026-10-13)

- 1,031 stored games (seasons start 2023–2026); **992** non-OT regulation games eligible for the structural model.
- v0.7 trained PPP model: prior=4 equivalent team games; Ridge alpha=2; offense coefficient ≈0.324, opponent allowed ≈0.168, home log offset ≈0.0436. These came from data, **not defaults for all future seasons**.
- Independent PPP-only, *using realized pace for diagnostic decomposition only*: game total RMSE 13.610 vs league PPP baseline 13.855 on 194 previously examined regulation games.
- **Full pregame** with predicted v0.6 pace: game total RMSE **16.610** for PPP comparator, **17.428** for v0.6 structural, **16.884** for league-rate comparator. These are *not independent newly untouched data*; the holdout was examined multiple times.
- 90% simulated interval coverage (388 team sides/194 games): team PTS 92.3%, FTA 79.1%, official DREB 99.5%, official REB 98.7%, total points 91.2%, match margin 96.9%. **Distribution pricing gate remains closed.**

## Install/update the EXISTING repository

Download/unzip the v0.7 GitHub overlay package. Upload its files **into the repo root** with correct paths, replacing same-named files including `.github/workflows/euroleague-refresh.yml`. Do NOT upload the `data/euroleague` sample or overwrite historical exports. Do NOT create a second workflow.

1. Commit changes in GitHub `main`.
2. Open **Actions → EuroLeague Data Refresh → Run workflow → daily**, not `bootstrap`.
3. Check the steps: `Fit structural SHADOW v0.6`, `Fit independent pace-times-efficiency SHADOW v0.7`, `Independent pregame totals audit`, `Audit 90pct MC coverage`, `Verify ... release gate`.
4. After a green run, verify `data/euroleague/shadow_v07/` contains:
   - `efficiency_benchmark_v07.json`
   - `pregame_totals_audit_v07.json`
   - `distribution_audit_v07.json`
5. Reboot Streamlit only if app does not redeploy; open **Trader Board SHADOW v0.7**.

Do not run historical bootstrap again. No API key or external service needed for these computations.

## Mathematics / key restrictions

- Shared predicted game possessions and team/opp scoring-efficiency profiles. No observed game pace at prediction time.
- A two-way half-point market has P(over)+P(under)=1, no pushes; fair odds=1/P; book odds=1/(1.09×P).
- 3-way Most: P(home bigger), P(equal), P(away bigger) computed from **paired** regulation MC; book odds=1/(1.15×P). The natural tie is kept, not split.
- Official-inclusive team markets; steals are constrained by *player-credited* opponent turnovers.
- Standard bookmaker game total/handicap often INCLUDE OVERTIME. These simulations are **40-minute regulation-only** and must **not** be used to quote OT-inclusive contracts. Implement OT simulation and explicit settlement mapping before production.
- Bias, expected attempts, uncertainty, probability calibration, injury/rotation, venue data, true play-by-play rebounds and schedule factors all need further work.
- Never derive an edge or betting units from shadow model probabilities.

## Research foundations

- Kubatko J., Oliver D., Pelton K., Rosenbaum D.T. (2007), *A Starting Point for Analyzing Basketball Statistics*, Journal of Quantitative Analysis in Sports, DOI 10.2202/1559-0410.1070 — possession accounting, efficiency and Four Factors.
- Attard P., Suda D., Sammut F. (2023), *Bayesian Hierarchical Modelling of Basketball Team Performance* — separating 2P/3P/FT scoring abilities with offense/defense and uncertainty (this v0.7 is regularized Ridge, **not** a reproduction of their Bayesian model).
- Gneiting T. and Raftery A.E. (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA, DOI 10.1198/016214506000001437 — calibration and evaluating full forecast distributions.
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, section 5.10 — time-series cross-validation and rolling-origin evaluation.

## Next development gates (do not hide)

1. Improve cross-matchup separation in possession model and shooting mix only if multiple chronological folds demonstrate improvement.
2. Add rotation, injuries, home/travel/rest, distinct seasonal roster priors; separately train calibration for FTA/rebounds/margins.
3. PBP-derived free-throw rebound/dead-ball classification; consistent OT-inclusive regulation-to-final settlement.
4. Score probabilities with rolling test + log loss/Brier/CRPS; report uncertainty intervals and failure cases.
5. Only after calibration and an untouched test should production bookmaker odds be authorized. The shadow board is a **preview**.
