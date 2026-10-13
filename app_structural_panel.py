"""Read-only Streamlit tab. No training or upstream API fetch on page rerun."""
from __future__ import annotations
import json
from datetime import date,timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
from core.structural_v05 import (verified_games,event_rate_rows,upcoming_pair,
                                  project_two_sides,simulate_two_sides)

@st.cache_data(show_spinner=False,ttl=3600)
def _validated_data(csv_path:str,player_path:str,stamp:float,player_stamp:float):
    raw=pd.read_csv(csv_path,low_memory=False)
    players=pd.read_csv(player_path,low_memory=False)
    return event_rate_rows(verified_games(raw,players))


def render_shadow(base:Path)->None:
    st.warning('EXPERIMENTAL / SHADOW — not fair odds, not injury-adjusted, not production predictions.')
    model_path=base.parent/'shadow_v05'/'structural_v05_report.json'
    if not model_path.exists():
        st.info('Run `python train_structural_v05.py` via the GitHub Actions workflow. The generated report will appear in data/euroleague/shadow_v05/.')
        return
    data_path=base/'team_games.csv'; player_path=base/'player_games.csv'
    if not data_path.exists() or not player_path.exists():
        st.error('Missing persisted team/player box scores')
        return
    report=json.loads(model_path.read_text(encoding='utf-8'))
    if report.get('status')!='SHADOW_NOT_PRICING_READY' or report.get('market_definition')!='players_only_v1':
        st.error('Invalid shadow model/market schema, refusing to run')
        return
    st.caption(f"Historical heldout: {report['train_games']} train / {report['test_games']} test games. Cutoff {report['cutoff_utc']}. No player availability effects.")
    events=_validated_data(str(data_path),str(player_path),data_path.stat().st_mtime,player_path.stat().st_mtime)
    season=int(events.SEASON.max())
    latest=events[events.SEASON.eq(season)]
    display=latest[['TEAM_ID','TEAM_NAME']].drop_duplicates().sort_values('TEAM_NAME')
    names={r.TEAM_ID:r.TEAM_NAME for _,r in display.iterrows()}
    ids=list(names)
    left,right,third=st.columns(3)
    home=left.selectbox('Home team',ids,format_func=lambda t:names[t],key='str_v05_home')
    away=right.selectbox('Away team',[x for x in ids if x!=home],format_func=lambda t:names[t],key='str_v05_away')
    earliest=(events.GAME_DATE.max().date()+timedelta(days=1))
    next_date=third.date_input('Pregame as-of date (UTC)',value=max(date.today(),earliest),min_value=earliest,key='str_v05_date')
    if st.button('Run read-only structural preview',key='str_v05_preview'):
        try:
            home_r,away_r=upcoming_pair(events,season=season,home_id=home,away_id=away,asof_date=next_date)
            preds=project_two_sides(home_r,away_r,report.get('next_game_shadow_coefficients') or report['coefficients'])
            st.metric('Shared projected possessions',round(preds['pace'],1))
            ks=['PTS','FG2A','FG3A','FG2M','FG3M','FTA','FTM','MARKET_OREB','MARKET_DREB','MARKET_REB','MARKET_TO','AST','STL','BLK']
            display_names={'MARKET_OREB':'OREB (credited players)','MARKET_DREB':'DREB (credited players)',
                           'MARKET_REB':'REB (credited players)','MARKET_TO':'TO (credited players)'}
            rows=[{'Metric':display_names.get(k,k),names[home]:round(preds['home'][k],2),
                   names[away]:round(preds['away'][k],2)} for k in ks]
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.caption('Expected means under structural identities only; not a market line, not calibrated over/under probabilities.')
            st.info('No lineup/absence inputs yet. No rest adjustment, no technical-FT rebound PBP classification.')
        except (ValueError,KeyError) as ex:
            st.error(str(ex))
    with st.expander('Chronological test audit'):
        out=[]
        for k,x in report['metrics'].items():
            out.append({'Market':k,'Model RMSE':x['rmse'],'Baseline RMSE':x['baseline_rmse'],
                        'Difference %':x['improvement_pct'],'Bias':x['bias']})
        st.dataframe(pd.DataFrame(out),hide_index=True,use_container_width=True)
        st.caption('One historical holdout, not a performance guarantee. No fair prices issued.')
