# EuroLeague model — Preflight/data gate (NOT a new pricing engine)

This archive does **not** replace `sync_cloud.py`, `app_euroleague.py` or WNBA production.
Its purpose is to freeze design rules and inspect every export before fit/price.

Files:
- `audit_dataset.py`: offline, read-only audit of two exported CSVs.
- `MODEL_CONTRACT_V1.md`: mathematical and ML design + acceptance criteria.
- `test_audit_dataset.py`: self-contained tests that must pass with only pandas.
- `data_quality_report.json`: the actual audit of the two 2026-only CSVs provided on 2026-10-13 (not a training outcome).

## Usage

Install `pandas` (already present in the existing engine requirements), then:

```bash
python audit_dataset.py --team data/euroleague/exports/team_games.csv --player data/euroleague/exports/player_games.csv --out data_quality_report.json
python -m unittest test_audit_dataset -v
```

Only when ingestion has completed for historical seasons and audit shows `ready_for_training: true` do we begin experimental model training. That flag is a coverage gate, **not** authorization to publish odds.

## Next actions in GitHub

In **Actions -> EuroLeague Data Refresh -> Run workflow**, select `mode=bootstrap` and `season=2025`, wait for success; repeat sequentially for `2024`, then `2023`. Each run should commit new exports; verify that manifest's `seasons` expands. If an intended year does not appear or reports parsing warnings, inspect `data_audit.csv` and job logs before continuing. Historical updates can take a long time: 7 seconds between HTTP calls × roughly hundreds of games, plus retries.

Use a separate development branch for any training engine. Current Streamlit can remain on existing code until validated.
