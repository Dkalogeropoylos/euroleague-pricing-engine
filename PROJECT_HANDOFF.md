# PROJECT HANDOFF — EUROleague/WNBA/NBA Basketball Pricing Engine

**Checkpoint:** 2026-10-13. Read this FIRST in a new chat; it replaces the need to recall previous long conversations. Attach this file plus latest GitHub export / repo ZIP if needed.

## User objective and non-negotiables

Build a reliable **online GitHub private repo + Streamlit Cloud** EuroLeague basketball pricing platform using the useful WNBA production architecture, not fragile local execution. Eventually deliver automatic game/team/player downloads, full team markets and player props, modeled fair probabilities/odds and units, plus rigorous player-by-player opponent matchups. NBA integration remains incomplete; WNBA production must remain untouched. User is strongly opposed to endless micro-patches and invented coefficients; consolidate, test, document, and deploy changes one vetted release at a time.

Critical immutable settlement definition: **team-market OREB, DREB, total REB and TO are sums of credited PLAYERS only**, not EuroLeague's official team totals. Maintain BOTH `MARKET_*` and `OFFICIAL_*`; the former is market truth, the latter for technical possession modeling. This rule applies throughout ingestion, opponent-allowed, training, simulation, backtesting and Streamlit. Do not confuse Paris `PRS` and Partizan `PAR`.

For 2-way pricing, user uses **9% bookmaker overround**; for 3-way **16%**; fair probabilities before margin. Do not invent market odds/availability. Must use detailed opponent allowed by position, rotation/minutes, absences, pace, shot profile, rest/double weeks, nonoverlapping H2H residuals and role-state weights before final player props. Excluding OT where relevant, no double-count of Season/L10/L5 and no arbitrary unconditional lineup redistribution.

## Implemented and VERIFIED with real user CSVs

- Private GitHub repo + Streamlit Cloud are user-configured; GitHub Actions daily plus manual bootstrap.
- Historical validated export: 1,031 games = 305 2023–24, 306 2024–25, 380 2025–26, and 40 2026–27 at this checkpoint; 2,062 team-games and 24,535 player-game rows. 1 missing 4OT game `E2023_170` because old parser rejected 300 team-minutes; user explicitly does not care about this one outlier. Avoid overemphasis.
- Current manifest after v0.4: `market_schema_version="players_only_v1"`, `training_gate.met=true`, `coefficient_stage="shadow_only_not_pricing_ready"`, four seasons, 1,031 games. Existing v0.4 `league_coefficients.json` is *shadow* only. Initial audit official-minus-credited sums across 2,062 team-games: OREB 3,616; DREB 3,607; REB 7,223; TO 1,343. Existing model v0.4 is NOT a production betting model.
- EuroLeague API adapter uses public v2 game stats; GitHub Actions incremental pull and Git commit. No local computer needed; user successful bootstraps last about 2h. Data validation required; no blind zeros for absent stats. Old bugs: BLK missing `blocksFavour` now fixed; same-game pace leakage corrected.
- Original v0.4 current project available as `euroleague_gate01_unified_release.zip` (previous conversation file). In a new chat if it cannot be accessed ask user to provide their CURRENT GitHub repo ZIP, not outdated code.

## New implemented structural v0.5 release (this conversation)

Code `core/structural_v05.py` + `train_structural_v05.py`, optional `app_structural_panel.py`, 5th tab integrated in `app_euroleague.py`, updated `.github/workflows/euroleague-refresh.yml`, `tests/test_structural_v05.py`.

As-of-day date embargo + same-season team/opponent rate history; league historical prior 8-games; last 550-day global reference; nested chronological coefficient selection using Ridge / RMSE, experimental counts; shared game possessions, TO/FTA event rates, shot share, 2P/3P/FT makes, recursive official/credited offensive rebound-generated extra FGA, player-credited DREB from remaining opponent missed FGs plus approximate reboundable FT misses, AST bounded by FGM, STL bounded by opp player TO, BLK bounded by opp missed 2P. Sample-level Monte Carlo enforces identities. **No fair odds!** Model remains SHADOW.

Actual one-off 777-train / 194-test games, trained only from earlier dates; results RMSE vs same structural pregame league baseline:
`POSS 3.983 vs 4.066 (+2.05%); PTS 10.170 vs 10.499 (+3.13%); FG2A 5.575 vs 5.803 (+3.94%); FG3A 4.977 vs 5.349 (+6.96%); FTA 6.281 vs 6.519 (+3.65%); MARKET_OREB 3.233 vs 3.239 (+0.19%); MARKET_DREB 3.861 vs 3.885 (+0.63%); MARKET_REB 4.997 vs 5.022 (+0.49%); MARKET_TO 3.242 vs 3.339 (+2.93%); AST 4.218 vs 4.321 (+2.39%); STL 2.489 vs 2.515 (+1.03%); BLK 1.647 vs 1.644 (-0.19%).` Outer cutoff Jan 16 2026. Avoid repeatedly tuning on this holdout; some gains too small to support rollout. League structure varies across roster changes; first 4 matches of 2026–27 have strong shrinkage.

No PBP free throw rebound classification; **FT reboundable-miss proxy 0.50 is a heuristic only**, cannot be validated from box scores. No injury minutes/availability, opponent by 5 distinct positions, PF/FD game couplings, home-neutral venue verification, rest congestion, Brier/log-loss, simulation tail calibrations. Do NOT claim pricing-ready even when `training_gate.met=true`.

Latest deliverables include an upload-ready patch ZIP, independently readable report JSON, and THIS portable handoff. Read README/RELEASE_NOTES in ZIP first. It does NOT contain large raw history CSVs; do not overwrite user data.

## Next gates AFTER user deploys v0.5

1. Verify GitHub Actions executes new shadow training (not just existing `daily`) and `structural_v05_report.json` is committed; test new Streamlit SHADOW tab with full outputs and double-count consistency.
2. Advance to more defensible PBP-based actual possessions, FT rebound opportunities and OT condition. Learn parameters rather than assign ad hoc prior weights; performance on a **new** rolling chronological test. Keep team-only vs credited dual state.
3. Add schedule and rest/double-week factors, player availability/rotation impacts, role-specific positional defense from genuinely identified player roles; never assume arbitrary lineup impact.
4. Fit calibrated event-count variance and correlation vs out-of-time outcomes; interval coverage, Brier score, reliability plots, compare vs baselines and verified bookmaker closing lines where available. Do not call Monte Carlo fair odds before this.
5. Once gated, implement all team props then player props, with 9% / 16% overround where requested; preserve WNBA baseline rather than replace.

## Exact minimal continuation request for a new chat

"Συνέχισε το EuroLeague Pricing Engine από το PROJECT_HANDOFF.md, χωρίς να επαναλάβεις τη δουλειά. Έχουμε 1.031 ιστορικά παιχνίδια, `players_only_v1`, v0.5 structural SHADOW με 777/194 chronological split. Κάνε τεχνικό έλεγχο GitHub/Streamlit από το τρέχον repo, διόρθωσε μόνο αποδεδειγμένα bugs και προχώρησε στο επόμενο model gate. Μην βγάλεις fair odds πριν calibration."
