# EuroLeague v1.1 — chronologically audited mean model, blocked quote gate

**Not production-ready bookmaker prices.** The new v1.1 default board uses only
current-season offensive, opponent-allowed and net-strength data in its live estimates;
2023-2025 are used exclusively to select general coefficients and to calibrate
empirical error distributions. 109% / 115% quote preview remains **HOLD**.

Workflow: upload the files from the GitHub patch preserving directories → run
`EuroLeague Data Refresh / daily` → verify `shadow_v11` JSON files → review
`Trader Board v1.1 AUDITED`. Do NOT perform historical bootstrap again.

For standalone checks in an existing full repo:

```
python -m pytest -q
python train_matchup_v11.py
python audit_distribution_v11.py --draws 1200
python verify_matchup_v11.py
python audit_all_matchups_v11.py
```

Read `PROJECT_HANDOFF_V11_GREEK.md` for detailed methodological rationale,
explicit unsolved issues, chronology, comparisons, and forward-work instructions.
