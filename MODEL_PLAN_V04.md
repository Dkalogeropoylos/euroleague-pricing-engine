# Model research gates — change freeze

| Gate | Required deliverable | Objective acceptance |
|---|---|---|
| 0. Historical raw/market truth | All completed regular-season matchups, player and official stats, separate `MARKET_*` | 3+ seasons/500+ games, all reconciliation checks pass, no unaccounted missing data |
| 1. Pre-game pace | Trained shared game pace against league baseline | No same-game leakage, out-of-time heldout game pace RMSE and bias logged |
| 2. Possession/shot tree | TO and team-only TO, FTA, 2PA/3PA, FG2M/FG3M/FTM | Every draw respects scoring and shot accounting, heldout distribution calibration |
| 3. Credit-only rebounds | Team-only rebound state, credited OREB and DREB competing for eligible missed shots | Every draw `market REB = credited OREB+DREB`, player market expectation not inflated |
| 4. AST, STL, BLK, PF/FD | Event opportunity conditional distributions | Event bounds enforced and out-of-sample models beat or tie simpler baselines |
| 5. Roles and absences | Team minutes total=200 regulation; out-by-role usage, attempts, creation, opponent by positions | No phantom player usage, H2H residual disjoint from recent/opponent baseline |
| 6. Pricing & rollout | Correlated Monte Carlo, market ladders, fair odds, push rules, bookmaker margin | walk-forward reliability/Brier/log score and historical stress tests before deployment |

Design rule: Freeze each gate after documenting data version, tests, holdout
metrics, and known limitations. All new challenger models stay SHADOW until
out-of-time validation passes. Do not tune continuously on the same holdout.

**Basketball references** for foundational mechanisms (not fitted EuroLeague
coefficients): Dean Oliver, *Basketball on Paper* (2004); Kubatko et al.,
*A Starting Point for Analyzing Basketball Statistics* (2007); Bayesian
hierarchical offense-vs-defense modelling such as Attard et al. (2023).
