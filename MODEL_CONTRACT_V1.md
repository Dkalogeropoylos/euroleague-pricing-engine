# EuroLeague Pricing Engine — Frozen model contract v1 (design, not production validation)

Date: 2026-10-13. Scope: EuroLeague regular season, 40-minute regulation; playoffs/OT as explicit contexts, never silently mixed. **The current release is preflight-only. No fair odds are certified by this document.**

## 0. Non-negotiables

1. **As-of cutoff**: every pregame feature must be reproducible using data published before a fixed `prediction_time_utc`. Never use target-game possessions, result, injury information posted after cutoff, or outcomes of other games not yet complete. Keep source timestamp/provenance and missingness flags.
2. **Paired rows**: exactly two team records per game, coherent team/opponent mirror and 200 team player-minutes in regulation; validate point/shot arithmetic before storage. Do not conflate player rebound/turnover totals with team totals; report team-only events separately.
3. **Freeze acceptance policy** before running holdout tests; no repeated tweaks on the final holdout. Different fitted parameters per league; share code architecture with WNBA, not coefficients.
4. **One team-event pool** for each game; player markets condition on that pool, not independent Poisson draws detached from team counts.
5. **Model outputs distinguish** raw baseline, matchup adjustment, rotation adjustment, simulation median/mean, uncertainty, fair probability, and optional bookmaker quotes.
6. **No certification** from successful GitHub Actions alone. Raw downloads, calibrated parameters and market-ready probabilities are three distinct statuses.

## 1. Data source, schemas and update strategy

- Existing EuroLeague v2 public box-score adapter is our upstream ingestion, with a local cache and incremental GitHub Actions. Store canonical keys `(COMPETITION, SEASON, GAME_ID, TEAM_ID)` and player key `(...,PLAYER_ID)`. Preserve team summaries and raw provider versions where permitted; check historical revisions independently.
- Separate `team_game`, `player_game`, `roster_appearance`, `availability_event`, `schedule` and `pbp_event` tables. Add PBP ingestion **only after schema comparison** across historical seasons; use it to refine possessions, live FT trips, substitution intervals and rebound opportunities.
- `POSITION` currently contains only `1,2,3` (broad G/F/C), not five positions. Five-role matchup allowed can only be used after reliable player role classification; otherwise use 3-role groups and show confidence.
- Historical seasons 2023, 2024, 2025 plus current 2026 refer to **season start years** (2023-24...2026-27). Preserve league structural breaks, team entry/exit and roster changes.
- Do not keep large raw PBP in the Streamlit session. Nightly ingestion and coefficient jobs run in GitHub Actions / other worker; Streamlit reads cached compact snapshots.

## 2. Basketball generative graph (do NOT fit all raw prop counts independently)

```text
pregame information -> shared latent game possessions (PACE)
                        |                                  |
                        v                                  v
                    team A opportunities               team B opportunities
                        |                                  |
              turnover rate / FTA trips          turnover rate / FTA trips
                        |                                  |
             field goal attempts & 3PA share    field goal attempts & 3PA share
                        |                                  |
                 binomial 2PM/3PM/FTM           binomial 2PM/3PM/FTM
                        |                                  |
                missed shots -> OREB <---- contested opponent DREB
                        |
              AST/STL/BLK/fouls conditional on event opportunities
                        |
            team pool -> on-court player minutes/usage -> player outcomes
```

**Exact identities per realization:** `FGA=2PA+3PA`, `FGM=2PM+3PM`, `PTS=2*2PM+3*3PM+FTM`, `REB=OREB+DREB`. `3PM<=3PA`, `2PM<=2PA`, `FTM<=FTA`, `AST<=FGM`, `STL<=opponent TO`; assess technical/team events explicitly. `BLK` is constrained by blockable opponent missed attempts, with attempted 3P block exceptions handled in PBP.

**Possessions** from box scores: `POSS_EST=FGA+0.44*FTA-OREB+TO` is an approximation, not truth. Both sides share roughly one pace but their estimates may disagree. Use pair-average as a historical target proxy while training pregame pace on **historical features only**, and track difference/disagreement/OT. The `0.44` factor is an established approximation, not a EuroLeague-estimated fact: calibrate its applicability using PBP if reliable.

**Rebounds**: opportunity pool driven by opponent misses including eligible FT misses and non-reboundable stoppages; OREB% conditioned on opponent DREB contested. Distinguish team rebounds (`team total - sum player credited`); do not allocate them to player props.

**Assists** conditional on field goals and assisted rate; **steals** conditional on opponent turnovers. **Blocks** conditional on shot attempts and defensive lineup. **FTA/PF/FD** relate via foul-type classification, not by forcing `team FD=opp PF` (offensive/technical fouls mean mismatches can be real).

## 3. Coefficient families to fit, per market/league

| Layer | Features / quantities | Candidate distribution / model |
|---|---|---|
| Pace | team adjusted tempo, opponent tempo, home/away, rest, season league rate, rotations, OT handled separately | hierarchical log-rate regression / residual distribution |
| Offensive events | turnovers per possession, FTA trip rate, 3PA share, FG opportunity rate | beta-binomial / regularized binomial rate models |
| Shooting | 2P% and 3P%, offensive skill, defensive concession, venue; shooter/mix effects | binomial hierarchical shrinkage; beta-binomial uncertainty |
| Rebound process | miss pool, OREB% / DREB%, lineup & opponent size | binomial / beta-binomial conditioned on opportunities |
| Other counts | AST/FGM, steals/forced TO, BLK/opponent attempts, fouls | conditional binomial or suitable count distribution with constraints |
| Player allocation | projected minutes, usage, role, on/off, team event pool, role-out states | calibrated opportunity shares + overdispersed exposure model |
| Market probabilities | game-level correlated outcomes; score, possessions, shooting and rebounds consistent | Monte Carlo with calibrated event/parameter uncertainty |

Candidate models may include Ridge/GLM, empirical Bayes, Negative Binomial for unconstrained count residuals, state-space adjustments for role changes. **These are candidates, not claimed fitted results**. Do not use Neural Networks/XGBoost automatically. Select by out-of-sample accuracy and calibration; prefer simpler if ties.

## 4. Evidence/weights and no double-counting

- Construct chronological game-level data; no independent Season/L10/L5 weights applied to overlapping samples. Let recency decay and change-point/role features compete in validation.
- Opponent allowed calibrated on rates conditioned by position where role classification is reliable; shrink samples and exclude garbage-time / tiny-minute player exposures when measuring allowed-by-position.
- H2H only **residual vs pregame matchup baseline**, disjoint from other same game samples, and preferably rotation-similar; if insufficient signal, zero extra impact.
- Current-year team profiles shrink to league/roster-continuity priors. Do not carry last season's team identity unchanged across major roster turnover. Historic seasons are useful chiefly to estimate league-level coefficients and variance.
- Availability by position, with projected minutes adding to **200 team player minutes (regulation)**; redistributed usage/shots/assists/rebounds should be conditioned on which player/role is out. Manual minute and OUT overrides audited.
- Domestic league evidence is **not in current EuroLeague CSV**; any domestic effect requires an independent provider, league strength adjustment and identity mapping. Do not claim it was ingested.

## 5. Validation / experiment design

1. Data integrity audit must pass with no fatal results. Historical-coverage policy for first training: >=3 season-start years and >=500 validated games. **This is a data gate only; not a claim of statistical sufficiency**.
2. Walk-forward chronological training by **game-date/time groups**; hold out entire future blocks, never train and test on opposite teams from one match. All hyperparameters fitted within past-only folds.
3. Compare strong baselines: current-year/shrunk league mean, opponent-adjusted regression, structurally coherent generative candidate. Benchmarks per market: MAE/RMSE for counts, log score for likelihood, Brier and reliability curves for event O/U, interval coverage and predicted-vs-realized variance.
4. Freeze final holdout and tests *before* any hyperparameter experiment. Include at least whole final season stage after long historical sample. Use proper calibration and preserve uncertainty after roster changes.
5. Audit each mathematical identity **for every Monte Carlo draw**; ensure no impossible combinations and correct push/OT rules by market.
6. Launch criteria: improved out-of-sample metrics vs simple prior *and* calibrated probability bins, not merely a better historical in-sample fit. No model-priced value until both met.

## 6. Pricing and interface

- `P(Over x)` and `P(Under x)` from *the same simulated distribution*; integer lines may push, priced accordingly. Fair price = reciprocal of applicable win probability adjusted for push rules.
- Two-way book quotes at requested 9% overround; three-way at 16% overround; always show **fair / bookmaker with margin / confidence** separately.
- EV and unit stakes only after calibrated predictions and explicit bookmaker prices; pass is a valid outcome.
- UI: league, fixture, team markets, player props, minutes/availability overrides, data-freshness and diagnostics, fair lines, reference prices and actual offered bookmaker prices. Compute simulations once, reuse by changing lines; cache raw snapshots.

## 7. Development sequence and change control

- Gate 0: source ingestion, audit, historical multi-season collection and archived version snapshots.
- Gate 1: pregame PACE and possession-calibrated league baselines; prove no realized-POSS leakage.
- Gate 2: coherent shot volume / shooting / rebound event graph, then assists/TO/steals/blocks/fouls; holdout and Monte Carlo identity tests.
- Gate 3: availability, same-role, minutes redistribution, H2H residual and opponent by credible roles.
- Gate 4: pricing validation, bookmaker comparison, clean Streamlit cloud deploy.

**Change management:** keep WNBA production untouched; EuroLeague dev branch; one consolidated, tagged tested release per gate with migration script and rollback, not ad-hoc patches. Log model version, training source versions, source refresh time, metrics and all explicit manual overrides.

## References supporting mechanisms, not numerical priors

- Dean Oliver (2004), *Basketball on Paper*, and Basketball-Reference's Four Factors and Offensive Rating explanations.
- Kubatko et al. (2007), *A Starting Point for Analyzing Basketball Statistics*, Journal of Quantitative Analysis in Sports (basis for possessions approximation, additional caveats).
- Attard, Suda, Sammut (2023), *Bayesian Hierarchical Modelling of Basketball Team Performance* (NBA case study of 2P, 3P, FT offensive/defensive abilities; NegBin comparison). These data were NBA, not a fitted EuroLeague coefficient source.
- scikit-learn, *TimeSeriesSplit* documentation; use custom matchup/date grouped chronology for basketball.
