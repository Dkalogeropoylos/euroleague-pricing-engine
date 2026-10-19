# EuroLeague Engine — v0.9 Parameter Calibration Audit
**2026-10-13 | SHADOW, not production pricing**

## Verified source and existing model
- 1,031 games over seasons 2023–2026; 992 regulation games after excluding OT.
- 1,984 regulation team-box-score rows; initial-day embargo yields 1,942 frozen pregame training snapshots.
- Team OREB, DREB, REB, TO = **official inclusive**. Player-credited portions retained separately for player props and stealable turnover opportunities.
- Current v0.8 has 550-day league baselines and one prior=4 for multiple features; v0.6 has prior=8 and per-factor ridge select. In v0.8 inner CV, the `v06` candidate is evaluated with coefficients fitted using the whole outer training set. These include some inner-validation dates, so comparing it with freshly fitted inner-fold challengers is methodologically unfair. This is a flaw in **selection**, not proof all the old coefficients are wrong.
- The opportunity-denominator criterion in the previous rate selector involves realized target-game exposures. While exposures are not provided as live features, model-selection loss is not the same objective as the final prediction of pregame counts. Current audit corrects this by scoring actual counts directly.

## Independently estimated hyperparameters
| Market | Pseudo-games prior | Previous-season component of prior | Feature choice | Selected RMSE / League RMSE in two tuning windows | 2026 watch RMSE / league RMSE |
|---|---:|---:|---|---|---|
| Pace | 2 | 100% | multiscale | 3.703 / 3.950 | 3.410 / 3.894 |
| 3PA | 4 | 50% | multiscale | 4.588 / 5.304 | 4.700 / 5.145 |
| Steals | 32 | 50% | offense + opponent | 2.470 / 2.512 | 2.788 / 2.893 |
| Turnovers | 8 | 100% | offense + opponent | 3.339 / 3.494 | 3.182 / 3.312 |
| Official OREB | 16 | 50% | multiscale | 3.603 / 3.669 | 3.416 / 3.490 |
| Official DREB | 8 | 100% | offense + opponent | 4.124 / 4.199 | 4.254 / 4.372 |
| Official REB | 8 | 100% | multiscale | 5.257 / 5.414 | 5.601 / 5.888 |
| Team PTS | 4 | 100% | multiscale | 10.271 / 11.090 | 9.359 / 10.292 |

*Pseudo-game prior* controls the initial pull of an observed team rate towards its prior-year/league target. It is different from the Ridge `alpha` parameter; both are separately searched. Selection uses the two specified January–May windows, while the September/October 2026 watch data were not used to refit the tuning fold models; however, these 40 games were already used informally during product development and are NOT pristine independent evidence.

### Sensitivity and uncertainty
- 3PA top five candidate shrinkage combinations differ by **< 0.012 RMSE**: there is no precise warrant for insisting the best prior is *exactly* 4 rather than 2 or 8.
- Steals top candidates include priors 16 and 32 with virtually indistinguishable RMSE, so the data say *stronger shrinkage*, not that 32 is a scientifically exact constant.
- 2026 bootstrap-by-match 95% interval for percentage improvement against a simple league baseline: pace **2.8% to 19.7%**, 3PA **1.2% to 14.7%**, steals **0.3% to 6.3%**, turnovers **-2.1% to 9.2%**, rebounds **0.1% to 9.9%**. These are approximate, selection is already made, and 2026 was previously inspected; do NOT describe them as a new clean holdout confidence guarantee.
- OREB and DREB intervals overlap zero improvement: there is insufficient evidence to promote them into live prices.

## Diagnostic matchup check — independent direct-count regression, not the pricing simulation
| Matchup/stat | v0.8 structural mean (CSV) | Independent v0.9 count shadow | Comment |
|---|---:|---:|---|
| Partizan 3PA vs Panathinaikos | 25.88 | 26.50 | Small upward change; no support for a large arbitrary correction |
| Panathinaikos steals vs Partizan | 5.74 | 6.21 | More than old league-only tendency, but far below a naive 4-game mean of 8.25 |
| Panathinaikos official turnovers | 12.83 | 12.87 | Near-identical results support old estimate |
| Paris 3PA vs Virtus | 29.82 | 32.60 | Material disagreement; requires shot-mix/roster study |

The independent benchmark forecasts outcomes **separately** and does not impose the joint 2PA+3PA=FGA, PTS=2×2PM+3×3PM+FTM, or OREB+DREB=REB identities. It must not feed Monte Carlo or odds without coherent integration and separate calibration. Reported independent means are not fresh odds.

## Current verdict
1. **Not a reason to abandon the whole model.** Pregame pace/3PA contain learnable signal, and the old frame's game-event conservation is useful.
2. **Do not approve v0.8 fair odds for betting or bookmaker pricing.** Variance/OT/availability are unresolved, and old Monte Carlo interval coverage deviates sharply for rebounds and FTA.
3. **Do not use one shared shrinkage prior for all markets.** New calibrated priors indicate sport-specific sampling noise differs greatly.
4. **Do not blindly deploy this independent count model.** Re-estimate and calibrate within the joint structural model and use an untouched forward evaluation period before production promotion.

## Files / continuity
- `calibrate_shrinkage_v09.py`: independently calibrated pregame count study.
- `matchup_diagnostic_v09.py`: repeatable future-lookup of direct-count forecasts.
- `verify_calibration_v09.py`: guard before any GitHub JSON report is committed.
- `tests/test_calibration_v09.py`: data schema and same-day-leakage regression tests.
- `data/euroleague/shadow_v09/calibration_v09_report.json`: complete selected coefficients and diagnostics.
- GitHub daily workflow now fits this as a *shadow audit only*; Streamlit expansion panel compares means, but the live v0.6/v0.8 model and offered prices remain unchanged.

## Peer-reviewed methodological grounding
1. Kubatko, Oliver, Pelton & Rosenbaum (2007), *A Starting Point for Analyzing Basketball Statistics*, DOI: 10.2202/1559-0410.1070.
2. Terner & Franks (2021), *Modeling Player and Team Performance in Basketball*, DOI: 10.1146/annurev-statistics-040720-015536.
3. Attard, Suda & Sammut (2023), *Bayesian Hierarchical Modelling of Basketball Team Performance*, icSPORTS, https://www.scitepress.org/PublishedPapers/2023/121591/.
4. Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, DOI: 10.1198/016214506000001437.
5. Gneiting, Balabdaoui & Raftery (2007), *Probabilistic Forecasts, Calibration and Sharpness*, DOI: 10.1111/j.1467-9868.2007.00587.x.
