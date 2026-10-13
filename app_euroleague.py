"""Streamlit read-only shadow dashboard. No live API calls in app."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from league_coefficients import STAT_KEYS, clean_games
from market_truth import MARKET_SCHEMA_VERSION, MARKET_STATS, market_audit

st.set_page_config(page_title="EuroLeague Data & Coefficients Sandbox", page_icon="🏀", layout="wide")
st.title("🏀 EuroLeague — Data & League Coefficients (SANDBOX)")
st.caption("Local cache only. Does NOT modify the WNBA production engine and does NOT claim calibrated fair odds.")

base = Path(__file__).resolve().parent / "data/euroleague/exports"
team_path = base / "team_games.csv"
player_path = base / "player_games.csv"
audit_path = base / "data_audit.csv"
coeff_path = base / "league_coefficients.json"
if not team_path.exists():
    st.error("Δεν υπάρχουν ακόμη validated EuroLeague exports στο GitHub. Από GitHub → Actions → EuroLeague Data Refresh → Run workflow διάλεξε bootstrap.")
    st.info("Όταν ολοκληρωθεί επιτυχώς το GitHub Action, τα δεδομένα θα προστεθούν στο repository και η εφαρμογή θα τα εμφανίσει. Ο υπολογιστής σου δεν χρειάζεται να μένει ανοικτός.")
    st.stop()

raw_teams = pd.read_csv(team_path)
if raw_teams.empty:
    st.warning("Δεν υπάρχουν validated team-game rows.")
    st.stop()
if 'MARKET_SCHEMA_VERSION' not in raw_teams.columns or not (raw_teams.MARKET_SCHEMA_VERSION == MARKET_SCHEMA_VERSION).all():
    st.error("Το market contract δεν έχει εφαρμοστεί. OREB/DREB/REB/TO must be player sums. Apply the unified upgrade after bootstrap.")
    st.stop()
try:
    teams = clean_games(raw_teams)
except Exception as ex:
    st.error(f"Quality validation failure: {ex}")
    st.stop()
players = pd.read_csv(player_path) if player_path.exists() else pd.DataFrame()
audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
coeff = json.loads(coeff_path.read_text(encoding="utf-8")) if coeff_path.exists() else None
if coeff is not None and coeff.get('market_schema_version') != MARKET_SCHEMA_VERSION:
    st.warning("Stale coefficient file from an older market definition. It cannot be used: retraining is required.")
    coeff = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Validated games", len(teams)//2)
c2.metric("Team-game observations", len(teams))
c3.metric("Player-game observations", len(players))
c4.metric("Data quality flags", len(audit))

page_data, page_matchup, page_coeff, page_audit = st.tabs(["League Baselines", "Matchup Audit", "Coefficients", "Data Audit"])
st.info("**Market settlement rule:** Ομαδικά OREB, DREB, REB και turnovers είναι ΠΑΝΤΑ άθροισμα παικτών. Τα επίσημα team-only events κρατούνται χωριστά για pace/audit.")

with page_data:
    season = st.selectbox("Season start year", sorted(teams.SEASON.unique().astype(int), reverse=True))
    a = teams[teams.SEASON == season].copy()
    exposure = a.POSS_RAW.sum()
    display_stats = ["REB"] + STAT_KEYS
    if exposure <= 0:
        st.warning("Insufficient exposure")
    else:
        desc = pd.DataFrame([{"Metric": "Estimated possessions", "Per team-game": a.POSS_RAW.mean(), "Per 100 possessions": 100},
                             *[{"Metric": s, "Per team-game": a[s].mean(), "Per 100 possessions": a[s].sum()/exposure*100} for s in display_stats]])
        desc['Definition'] = desc['Metric'].map(lambda x: 'PLAYER SUM' if x in ('OREB','DREB','REB','TO') else 'OFFICIAL/PREGAME AUDIT')
        st.dataframe(desc.style.format({"Per team-game": "{:.2f}", "Per 100 possessions": "{:.2f}"}), hide_index=True, use_container_width=True)
        st.info("Possessions are currently box-score estimates (FGA−OREB+TO+0.44FTA), not exact play-by-play counts.")

with page_matchup:
    st.subheader("Side-by-side team matchups: descriptive audit, not fair odds")
    season = st.selectbox("Matchup season", sorted(teams.SEASON.unique().astype(int), reverse=True), key="matchup_season")
    aa = teams[teams.SEASON == season]
    names = sorted(aa.TEAM_ID.unique())
    if len(names) < 2:
        st.warning("Need at least two teams in the selected season")
    else:
        left, right = st.columns(2)
        home = left.selectbox("Home", names)
        away = right.selectbox("Away", [t for t in names if t != home])
        n = st.slider("Latest same-season games (not hardcoded model coefficients)", 3, 15, 8)
        def descriptor(team):
            log = aa[aa.TEAM_ID == team].sort_values("GAME_DATE").tail(n)
            return {"Games":len(log), "Pace estimate": log.POSS_RAW.mean(),
                    **{s:log[s].mean() for s in ["REB"] + STAT_KEYS},
                    **{"Allowed "+s:log["ALLOWED_"+s].mean() for s in ["REB"] + STAT_KEYS}}
        table = pd.DataFrame({home:descriptor(home), away:descriptor(away)})
        st.dataframe(table.style.format(precision=2), use_container_width=True)
        st.caption("Opposition allowed has been reconstructed from the matched opponent side of the same games. This panel deliberately does not turn raw means into picks.")

with page_coeff:
    if not coeff:
        st.warning("No fitted model. Run: python train.py after historical backfill.")
    else:
        st.write("**Holdout definition:**", coeff.get("holdout"))
        st.write("**Pregame pace model (shadow):**", coeff.get("pace"))
        report = []
        for s, value in coeff.get("markets",{}).items():
            report.append({"Market": s, "Status": value.get("status"),
                           "Off coeff": (value.get("coefficients") or [None]*4)[0],
                           "Opponent coeff": (value.get("coefficients") or [None]*4)[1],
                           "Holdout improvement %":value.get("improvement_pct"),
                           "Training team-games":value.get("train_rows")})
        st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)
        st.caption("If BASELINE_WINS, no extra opponent/home coefficient is used; no arbitrary positive adjustment is forced.")

with page_audit:
    st.subheader("Boxscore ingestion issues")
    st.write("**Player credited vs team-only gaps:**", market_audit(raw_teams))
    if audit.empty:
        st.success("No recorded ingestion issues in local backfill.")
    else:
        st.dataframe(audit, use_container_width=True)
    st.write("Last available game date:", str(teams.GAME_DATE.max()))
    st.write("Unique team codes:", len(teams.TEAM_ID.unique()))
    if players.empty:
        st.warning("No player-level records were parsed yet.")
