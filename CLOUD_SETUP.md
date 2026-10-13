# Deploy in GitHub + Streamlit Community Cloud

**Repo name:** `euroleague-pricing-engine` (**Private**, recommended). This is a shadow / data QA tool, not production fair odds.

## 1. GitHub (no local Python required)
1. Create a new **private** repository at https://github.com/new, named `euroleague-pricing-engine`.
2. Extract `euroleague_github_streamlit_ready.zip`. Upload the **contents** (not the enclosing folder or zip file) to the root of the repository. Make sure the hidden `.github/workflows/euroleague-refresh.yml` file is included. GitHub's web uploader can take folder drag-and-drop in supported browsers; alternatively use Codespaces or a Git client.
3. Commit the files to the `main` branch.
4. From GitHub, select `Actions` → `EuroLeague Data Refresh` → `Run workflow` → mode **bootstrap**. This retrieves 2023–24 through 2026–27 regular-season data, subject to upstream availability and rate limits, trains provisional shadow coefficients, and commits the validated CSV files. It can use significant free Actions minutes. If GitHub limits workflow write permissions, update Settings → Actions → General → Workflow permissions accordingly.
5. For later updates a `daily` workflow runs automatically at 04:37 UTC; a manual daily run is also available. Historic data files are appended incrementally, not re-downloaded unnecessarily. The refresh will retry invalid/missing games in the next run.

## 2. Streamlit
1. Sign in to https://share.streamlit.io/ with GitHub.
2. Create app → select `yourusername/euroleague-pricing-engine` → `main` → app entrypoint `app_euroleague.py` → Deploy. Streamlit will use `requirements.txt` at repo root. Select Python **3.11** if prompted.
3. For a private repo, the Streamlit app can be private. Community Cloud currently allows only one private app per account. It may sleep after inactivity and has limited RAM/CPU.
4. The Streamlit page will show a **no historical CSV yet** message until the GitHub Actions historical bootstrap succeeds and commits exports.

## Architecture
- GitHub Actions handles HTTP requests and numerical coefficient fitting.
- Checked-in `data/euroleague/exports/*.csv` and `.json` are immutable (until a further Actions commit) validated, curated data and audit outputs.
- Streamlit reads only these CSV and JSON exports; on browser reruns it neither fetches full seasons nor refits historical coefficients.
- Raw API responses, local files, token files and secrets are excluded from git.

## Safety / accuracy limitations
- We have tested synthetic offline payloads only. The current live endpoint response and 2026–27 season schema still need a first successful integration run.
- Historical feature estimation and shadow regression are not calibrated probability distributions or fair prices.
- Position-dependent conceded stats, actual play-by-play possessions, availability and player-on/off need further data extraction and validation.
- Avoid publishing or commercially repurposing third-party source data without checking data permissions/terms.
- `Actions` commits depend on `GITHUB_TOKEN` write permission and supported GitHub Actions quota.
