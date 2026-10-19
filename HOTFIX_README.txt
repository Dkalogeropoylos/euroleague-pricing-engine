EUROLEAGUE v1.1 HOTFIX — v1.0 script dependencies missing from previous v1.1 archive

Problem: .github/workflows/euroleague-refresh.yml runs python train_current_only_v10.py,
but the previously supplied v1.1 ZIP omitted the v1.0 source files.

Upload all FOUR files to repository main, preserving paths:
- train_current_only_v10.py (repository ROOT)
- verify_current_only_v10.py (repository ROOT)
- core/current_only_v10.py (inside core/)
- tests/test_current_only_v10.py (inside tests/)

Do not modify the workflow or historical CSVs. Commit and rerun daily.
This fixes missing-file failure; it does NOT certify fair odds for production.
