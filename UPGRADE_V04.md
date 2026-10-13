# EuroLeague v0.4 — frozen market settlement and pregame-data foundation

**Status:** prepared offline; not deployed, not approved for fair-odds production.

## IMPORTANT: apply AFTER the running historical bootstrap completes

Do not overwrite workflow/code while GitHub Actions is running: `git push` from
an old checked-out job could fail, and mixed-version exports are unsafe.

1. Wait for the historical bootstrap run to finish **green** and commit its output.
2. Check `data/euroleague/exports/manifest.json` for seasons `2023, 2024, 2025, 2026`
   and verify game counts by season. A green workflow alone is not a validation certificate.
3. Apply the files in this patch **preserving paths** (`tests/` etc.) on a clean
   GitHub commit. Do **not** upload `team_games.csv`, `player_games.csv` or other
   `data/euroleague/exports/` files from any local ZIP over the fetched GitHub ones.
4. Run GitHub Actions **daily** (manually once) after the patch is committed.
   That will migrate all already-saved season files to the `players_only_v1`
   settlement schema **without redownloading completed games**, and run
   calibration if there are >=3 season-years and >=500 validated games.
5. Inspect `manifest.json` for `market_schema_version: players_only_v1`, the
   `market_audit` gaps, and `training_gate`. The displayed coefficients are
   SHADOW ONLY until independent validation and coherent Monte Carlo pricing.
6. When reviewing a team-market regression, use the `OREB`, `DREB`, `REB`, `TO`
   variables only through `clean_games()`; it replaces official team totals
   with player credited totals, without altering official possession proxies.

If a later update fails the market reconciliation, treat that as an intended
safety stop: inspect upstream boxscore completeness rather than silently
falling back to inflated official team totals.

## Fixed accounting contract

Per `GAME_ID × TEAM_ID`:

    MARKET_OREB = sum(player.OREB)
    MARKET_DREB = sum(player.DREB)
    MARKET_REB  = MARKET_OREB + MARKET_DREB = sum(player.REB)
    MARKET_TO   = sum(player.TO)

    TEAM_ONLY_stat = OFFICIAL_stat - MARKET_stat

The official totals `OFFICIAL_OREB`, `OFFICIAL_DREB`, `OFFICIAL_REB`,
`OFFICIAL_TO` are kept only for auditing and technical possessions.

    POSS_RAW = official FGA + 0.44 * official FTA - OFFICIAL_OREB + OFFICIAL_TO

Do not substitute player sums into that historical possession estimator.
The predictor uses **historical previous-game possessions** and league history,
never the outcome of the game being priced.

**Offensive/defensive rebounds should be mutually coherent**: official missed
shots may end with a team-only rebound or dead-ball uncredited to any player.
Future simulated player-credited OREB and DREB must be bounded by available
rebound opportunities; never force every missed shot into credited rebounds.
Likewise official team turnovers can include events no player is credited with.

## Fail-closed data gates

- Unique teams and players; two teams per game; non-null, integer,
  nonnegative values for credited events.
- Player sums of made/attempted shots, points, assists, steals and blocks must
  reconcile against official team boxes; otherwise reject the data.
- Player rebounds must satisfy `REB == OREB + DREB`.
- Player-credited market counts cannot exceed their official team totals.
- No **official-to-market fallback** when player stats are missing.
- >=3 distinct seasons and >=500 completed games for provisional calibration
  in GitHub Actions; meeting the gate **does not** imply predictive accuracy.
- The data source/current-user screenshots alone do not prove the remote job
  has merged historical seasons; the manifest needs inspection.

## Current ML implementation vs next research stages

Already wired in this release:
- Pregame-only rate exposure; previous implementation erroneously used
  realized current-game `POSS_RAW` in training/validation predictor. Fixed.
- A separate **shadow pace Ridge** test against a league-average baseline.
- Chronological inner regularization and later outer holdout for the existing
  offensive/defensive rate coefficients. Team's and opponent's observations
  embargoed until after all games on the same date.
- Market OREB/DREB/TO targets learned **only from player credited counts**.
- Rebound total is derived from OREB+DREB; no third independent model.

Not implemented / not production-ready:
1. Exact EuroLeague possession and rebound opportunity reconstruction via
   validated play-by-play where available.
2. A coherent latent game-generating simulator enforcing FGA=2PA+3PA,
   2PM<=2PA, 3PM<=3PA, FTM<=FTA, PTS=2*2PM+3*3PM+FTM,
   credited REB=credited OREB+credited DREB; coherent team-only event leakage.
3. Calibrated 2P%/3P%/FT%, 3PA share/FTA and turnover generation with joint
   uncertainty; expected possessions shared by both teams.
4. Assisted FGM, steals and blocks conditional on appropriate eligible events.
5. Rotation/out-by-role and minutes projection, reliable opponent allowance by
   position, player attempt profiles and exposure-shrunk H2H residuals.
6. Simulations, market pushes, fair odds, 9% two-way / 16% three-way book
   overround, and walk-forward probabilistic calibration.

No numerical sport coefficients are hardcoded as truth; prior values in the
first candidate model are model-selection candidates, not ready odds inputs.
Do not reuse current WNBA coefficients in EuroLeague.

## Offline regression checks

Run `python -m pytest -q` from repo root. The tests cover accounting
reconciliation, missing players, duplicate IDs, migration without refetching,
base provider, successful cloud update and a no-same-day-result lookahead test.
All tests are offline; a green test suite does not replace live data/API audit.
