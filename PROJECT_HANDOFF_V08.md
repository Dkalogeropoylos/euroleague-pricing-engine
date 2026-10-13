# EuroLeague Pricing Engine — Project handoff after v0.8

**Current stage:** v0.8 team-state challenger SHADOW; v0.7 Trader Board retained; v0.6 base structural model unchanged.

**User requirements:** Team OREB/DREB/REB/TO must use EuroLeague OFFICIAL inclusive box-score team totals. Player sums stay separately for player props and stealable-turnover constraints. Team O/U and 2-way handicaps overround 109%; three-way More Stat markets 115%; display fair probabilities/odds beside bookmaker odds, in trader order. Do NOT treat uncalibrated odds as verified value.

**Data:** 2023–2026, 1031 historical games, regulation-only 992 games after 39 OT exclusions. 777 chronological training / 194 historical audit. Python Streamlit + GitHub Actions daily. Documents and code in v0.8 patch.

**Problem fixed:** original team lines were nearly identical, e.g. pace 74.4–74.7 across 190 pairings, OREB/DREB almost identical. v0.8 adds learned recent, season-to-date, predecessor season and opponent time-frozen features; selects Ridge specs and alpha by 2 growing chronological inner folds. v0.8 pace 72.57–77.06 and home FG3A 22.17–33.60 in 190 synthetic matchups. On previously examined audit data: pace RMSE 3.983→3.719; FG3A 5.058→4.737; rebounds 5.397→5.305. Do NOT call this fresh out-of-sample evidence.

**Still unsolved:** Rebounds remain compressed; OREB and DREB event rate chose league baseline, though reb projections vary more due attempts, shooting and pace. Accurate FT rebound opportunities require PBP. MC distribution coverage problems (old REB too wide, FTA too narrow), availability/rotations, schedule/rest, player shooting mix and ownership of team-only events still pending. v0.8 pricing is SHADOW.

**Deployment:** Upload v0.8 patch preserving folder structure; run GitHub daily; verify new shadow_v08 report; Streamlit Trader Board chooses `v0.8 team-state challenger` vs `v0.6 reference`.

**Next research:** cross-season blocked rolling-origin CV with fresh unseen games; hierarchical pace/opponent model and transfer-aware carryover; rebounds with reboundable missed FTs/dead-ball PBP; shooting player weighting and rotations/injuries; recompute calibrated correlated MC market probabilities. Production-gate only with multiple metrics (MAE/RMSE, bias, reliability, 90% interval coverage, Brier/log score) and independently held out data.
