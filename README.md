# Deploy on GitHub + Streamlit Community Cloud

**START HERE: [CLOUD_SETUP.md](CLOUD_SETUP.md)** — cloud-ready project with GitHub Actions daily incremental data refresh, curated exports in the repository and lightweight Streamlit dashboard. No historical model is available before the first successful bootstrap.

# EuroLeague Pricing Data Sandbox — 13 Oct 2026

**Purpose:** A separate sandbox to prove EuroLeague schedule + official game boxscore ingestion, league-specific coefficients, and an analyst-facing data audit **before** merging with the existing WNBA production engine. `NBA` remains a separate workstream.

**Current status:** Code complete for a prototype; offline synthetic unit tests pass. **Live upstream fetches and integration with the actual production repository have NOT been tested** in this execution environment (external DNS unavailable). This is not yet a production EuroLeague pricing tool, and it does not pretend that fitted coefficients already exist.

## Sources

- `https://api-live.euroleague.net/v2/competitions/E/seasons/E2025/games?limit=500&offset=0` — game schedule; repeat with `offset` if necessary.
- `https://api-live.euroleague.net/v2/competitions/E/seasons/E2025/games/{gameCode}/stats` — boxscores.
- `https://live.euroleague.net/api/PlayByPlay?gamecode=1&seasoncode=E2025` — optional follow-on possession and on/off implementation (NOT wired here).
- [giasemidis/euroleague_api](https://github.com/giasemidis/euroleague_api) Python package — alternative cross-check, not a mandatory dependency.
- [Euroleague API recon](https://github.com/McNews11/euroleague-open-data/blob/main/docs/api-notes.md) — field/rate-limit/identifier caveats.
- [Possession and lineup analytics](https://github.com/egemeny13/euroleague-analytics) — methodological reference.

`E2025` means **2025–26 season**, not calendar 2025. `E2026` means **2026–27 season**. Note API may be rate-limited; no official service-level guarantees, and commercial use may need permission from data rights holders.

## Start (Python 3.10+)

```
python -m pip install -r requirements.txt
python -m pytest -q
python backfill.py --seasons 2024 2025 2026 --max-games 5
```

Inspect `data/euroleague/exports/data_audit.csv` before backfilling everything. If schema/date/score checks pass, remove `--max-games`:

```
python backfill.py --seasons 2024 2025 2026
python train.py
streamlit run app_euroleague.py
```

A full historical backfill deliberately runs slowly: one HTTP request every **7 seconds** by default, with HTTP 429 `Retry-After` compliance and permanent JSON cache. Restarting the job reuses cached pages and individual games. Never do the season backfill at Streamlit page-render time. Production should put refreshed files in durable storage; Streamlit's local disk can be ephemeral.

## Delivered features

- One upstream adapter (`providers/euroleague.py`) for EuroLeague and EuroCup.
- Resumable data downloader (`backfill.py`).
- Per-game team and player logs with **separate identity namespace**; canonical team fields: `GAME_ID`, `GAME_DATE`, `TEAM_ID`, `OPP_ID`, `HOME_AWAY`, `PTS`, `FG2A`, `FG3A`, `FTA`, `OREB`, `DREB`, `AST`, `STL`, `BLK`, `TO`, `PF`, `FD`; also `OPP_<stat>` fields.
- Converts official `timePlayed` (seconds) to minutes; marks probable overtime by sum of player minutes; excludes overtime from coefficient fit.
- Data validation: two sides/game, no duplicates, 2PM/3PM and FT score reconstruction, official-score consistency where available, negative/missing stats, player-minute sanity, data-quality quarantine on strict load.
- Estimate pace from paired box-score possessions; labeled **approximate**. Exact play-by-play possessions is a separate phase.
- Multi-market league baselines; Ridge coefficient estimation using historical attack vs opponent-allowed information, plus home/away terms.
- No lookahead in attack/defense feature construction. Entire game day is embargoed until all results for that day are added. Nested chronological validation: choose regularization/baseline on an **inner** training-only time block; evaluate frozen model once on the last 20% holdout. If the challenger wins inner validation but fails outer holdout it is flagged `OUTER_HOLDOUT_FAIL`, not passed to production.
- Streamlit data QA view, with current-season team means and what those teams concede. It does **not** claim fair prices before probability calibration.

## Pricing model integration roadmap

1. Validate all `games`/`stats` endpoint response shapes against the current 2026–27 feed, and compare ingested games and scorelines with official completed results.
2. Archive `E2023`, `E2024`, `E2025` as training/reference. Keep `E2026` as an honest untouched chronological evaluation segment at first; never choose coefficients after seeing results from the dates being evaluated.
3. Feed adapter's canonical team/player CSV into the existing WNBA `core/team_model.py`, `core/matchup.py`, `core/structural_calibration.py`, `core/pace_engine.py`, `core/availability_impact.py` etc **after** confirming the core's actual expected schema in the running production branch. Do not bulk replace existing core code.
4. Fit EuroLeague-specific **possessions, shot allocation, 2P/3P efficiency, OREB/DREB, assist creation, fouls/FTA, turnovers, blocks and steals**. Use statistical holdouts, not hand-set or NBA coefficients. On early 2026–27 season shrink toward prior league baseline but reweight for roster turnover and player arrivals.
5. Play-by-play and lineup on/off for absences, opponent allowed *by position* (requires a curated role crosswalk G/wing/big; raw official `position` alone is insufficient), and fouls drawn. Raw per-game boxscore data cannot truthfully recover all of this.
6. Reconcile shot attempts, rebound opportunities, possession totals, points, player minutes/usage and OT into a coherent game-level simulation. Fit variances/correlations and expected distributions before fair odds (9% 2-way overround, 16% 3-way) are used in production.
7. Shadow backtest by date/team/market against results and timestamped bookmaker closing lines if available. Deploy only after provider uptime, accuracy and guardrails are verified.

## Quality / known limitations

- The upstream rate limit is undocumented; published external research measured HTTP 429 errors. Respect `Retry-After` and local cache.
- The API uses `local`/`road`, `gameCode` integer, and season code `E2026`. `E` and `U` are different competitions. `timePlayed` is seconds in v2.
- `phaseType` can vary by response/version; if your season unexpectedly loads zero games, inspect `data/euroleague/E/E2026/schedule_0.json` and adjust the regular-season filter, **do not silently treat post-season games as regular-season**.
- A 200 HTTP response does not imply usable data; the parser guards empties.
- Sparse or missing team fields are quarantined when strict mode is ON. Run `--non-strict` for audit ONLY, not for pricing. Different league seasons may use different schema.
- We haven't inferred EuroLeague player roles, reconstructed precise possessions, fetched live injury news, or performed real-season calibration in this offline environment.
- The code uses a Python client for the *public feed*, not an official commercially licensed API contract. Check rights and terms before commercial use.
