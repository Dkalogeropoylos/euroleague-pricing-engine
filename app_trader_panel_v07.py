"""Streamlit read-only trader-like research board. No bookmaker-ready probabilities."""
from __future__ import annotations
import io,json
from datetime import date,timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from core.structural_v06 import verified_games,event_rate_rows,upcoming_pair,project_two_sides,simulate_two_sides
from core.trader_markets_v07 import build_trader_board,threshold_counts,pricing_probabilities,evaluate_external_odds,MARKETS
from core.efficiency_benchmark_v07 import upcoming_efficiency_benchmark
from core.discrimination_v08 import upcoming_with_state, calculate
from matchup_diagnostic_v09 import run as count_only_audit

@st.cache_data(show_spinner=False,ttl=3600)
def load_verified(team_path,player_path,team_mtime,player_mtime):
    return verified_games(pd.read_csv(team_path),pd.read_csv(player_path))

@st.cache_data(show_spinner=False,ttl=3600)
def cached_sim(tpath,ppath,tm,pm,model_stamp,benchmark_stamp,season,home,away,asof,n,seed,model_data,challenger=None):
    data=load_verified(tpath,ppath,tm,pm)
    features=event_rate_rows(data)
    if challenger is None:
        h,a=upcoming_pair(features,season=season,home_id=home,away_id=away,asof_date=asof)
        pred=project_two_sides(h,a,model_data)
        sim=simulate_two_sides(h,a,model_data,n=n,seed=seed)
    else:
        h,a=upcoming_with_state(features,season=season,home_id=home,away_id=away,asof_date=asof)
        pred,sim=calculate(h,a,model_data,challenger,draws=n,seed=seed)
    return pred,sim


def _round_table(df, cols):
    q=df.copy()
    for c in cols:
        if c in q: q[c]=q[c].apply(lambda v:'—' if not np.isfinite(float(v)) else f'{v:.2f}')
    if 'P' in q:q['P']=q.P.map(lambda v:f'{v:.1%}')
    return q


def render_trader_shadow(base:Path):
    st.error('RESEARCH SHADOW — odds are mathematically derived from UNCALIBRATED simulations. NOT PRODUCTION PRICES / NOT CONFIRMED VALUE. No injuries, PBP rebound sequence or overtime.')
    p06=base.parent/'shadow_v06'/'structural_v06_report.json'
    p07=base.parent/'shadow_v07'/'efficiency_benchmark_v07.json'
    tp=base/'team_games.csv';pp=base/'player_games.csv'
    if not all(p.exists() for p in (tp,pp,p06,p07)):
        st.warning('Missing one of historical team/player exports or SHADOW v0.6/v0.7 reports. Run the updated GitHub daily workflow once.')
        return
    report=json.loads(p06.read_text());benchmark=json.loads(p07.read_text())
    v08_file=base.parent/'shadow_v08'/'discrimination_v08_report.json'
    v08=None
    if v08_file.exists():
        v08=json.loads(v08_file.read_text(encoding='utf-8'))
        if v08.get('status')!='SHADOW_NOT_PRICING_READY' or v08.get('market_definition')!='official_inclusive_v2':
            st.error('V0.8 schema/status mismatch. Refusing challenger pricing.');return
    v09_file=base.parent/'shadow_v09'/'calibration_v09_report.json'
    dist_file=base.parent/'shadow_v07'/'distribution_audit_v07.json'
    dist=json.loads(dist_file.read_text()) if dist_file.exists() else None

    if report.get('market_definition')!='official_inclusive_v2' or benchmark.get('data_definition')!='official_inclusive_v2':
        st.error('Settlement mismatch; do not quote prices');return
    d=load_verified(str(tp),str(pp),tp.stat().st_mtime,pp.stat().st_mtime)
    last=int(d.SEASON.max());teams=d[d.SEASON.eq(last)][['TEAM_ID','TEAM_NAME']].drop_duplicates().sort_values('TEAM_NAME')
    name=dict(zip(teams.TEAM_ID,teams.TEAM_NAME));ids=list(name)
    if len(ids)<2:st.warning('No teams for current season');return
    c1,c2,c3=st.columns(3)
    h=c1.selectbox('Home',ids,format_func=lambda z:name[z],key='tv07_home')
    a=c2.selectbox('Away',[v for v in ids if v!=h],format_func=lambda z:name[z],key='tv07_away')
    nextdate=d.GAME_DATE.max().date()+timedelta(days=1)
    asof=c3.date_input('Pregame as-of (UTC)', value=max(date.today(),nextdate),min_value=nextdate,key='tv07_day')
    if v08:
        selected=st.radio('Experimental coefficient version',
            ['v0.8 team-state challenger','v0.6 reference'],
            horizontal=True,index=0,key='tv08_model_choice')
        challenger=v08['next_game_challenger'] if selected.startswith('v0.8') else None
    else:
        st.info('v0.8 report not found; using v0.6 coefficients. Run latest GitHub daily workflow to compare versions.')
        selected='v0.6 reference'
        challenger=None
    st.caption('v0.8 changes pregame team-state coefficients, NOT the uncalibrated Monte Carlo variance. '
               'Differences in projected lines are not proof of reliable odds.')
    sims=st.select_slider('MC draws (more = slower; NOT calibration)',options=[2000,5000,10000,20000],value=5000)
    if st.button('Generate shadow trader board',key='tv07_run',type='primary'):
        with st.spinner('Computing coherent regulation-game scenarios...'):
            model=report.get('next_game_shadow_coefficients') or report['coefficients']
            pred,sim=cached_sim(str(tp),str(pp),tp.stat().st_mtime,pp.stat().st_mtime,
                p06.stat().st_mtime,(v08_file.stat().st_mtime if v08_file.exists() else p07.stat().st_mtime),
                last,h,a,asof,sims,2027,model,challenger)
            cross=upcoming_efficiency_benchmark(d,benchmark,last,h,a,asof,pred['pace'])
            board=build_trader_board(sim,name[h],name[a])
            st.session_state['v07_last']={'match':(h,a,str(asof),selected), 'pred':pred, 'board':board, 'sim':sim,'cross':cross}
    result=st.session_state.get('v07_last')
    if not result or result['match']!=(h,a,str(asof),selected):
        st.info('Select matchup and run the preview. Coefficients are loaded from the GitHub trained shadow artifacts.');return
    pred,board,sim,cross=result['pred'],result['board'],result['sim'],result['cross']
    st.caption('Active mean model: '+selected+'. Market probabilities remain research-only.')
    if v09_file.exists():
        try:
            r09=json.loads(v09_file.read_text())
            if r09.get('market_definition')!='official_inclusive_v2' or r09.get('status')!='SHADOW_NOT_PRICING_READY':
                st.error('v0.9 report has an incompatible market schema/status; refusing the independent audit.')
            else:
                with st.expander('v0.9 independent count benchmark — DOES NOT CHANGE PRICES'):
                    bench=count_only_audit(str(tp),str(v09_file),h,a,str(asof))
                    audit=[]
                    for metric,field in [('Pace','POSS'),('Points','PTS'),('3PA','FG3A'),
                                         ('Steals','STL'),('Turnovers','MARKET_TO'),
                                         ('OREB','MARKET_OREB'),('DREB','MARKET_DREB'),('Rebounds','MARKET_REB')]:
                        for side,team in [('home',h),('away',a)]:
                            if metric=='Pace' and side=='away':continue
                            ref=float(pred['pace'] if metric=='Pace' else pred[side][field])
                            key={'Pace':'POSS','Points':'PTS','3PA':'FG3A',
                                 'Steals':'STL','Turnovers':'TO','OREB':'OREB',
                                 'DREB':'DREB','Rebounds':'REB'}[metric]
                            alt=bench.get(team,{}).get(key,float('nan'))
                            audit.append({'Team':name[team],'Stat':metric,
                                          'Structural mean':round(ref,2),'Independent count mean':alt,
                                          'Difference':round(alt-ref,2)})
                    st.dataframe(pd.DataFrame(audit),hide_index=True,use_container_width=True)
                    st.caption('The independent benchmark estimates counts separately. It does NOT preserve '
                               'all score, attempt and rebound identities, and must NOT replace structural '
                               'means or feed fair/bookmaker odds automatically. Current fair odds still '
                               'refer solely to the selected v0.6/v0.8 structural simulation.')
                    cfg=r09['markets']
                    st.dataframe(pd.DataFrame([{'Statistic':k,'Prior games':v['selected']['prior_games'],
                               'Prev-season share of prior':v['selected']['previous_season_weight_in_prior'],
                               'CV RMSE':v['blocked_cv']['rmse_candidate'],
                               '2026 watch RMSE':v['watch_2026']['rmse_candidate']}
                               for k,v in cfg.items()]),hide_index=True,use_container_width=True)
        except Exception as e:
            st.warning('Unable to open v0.9 count benchmark; historical price model unchanged: '+str(e))
    if v08:
        with st.expander('v0.8 coefficient audit — trained challenger vs v0.6'):
            st.dataframe(pd.DataFrame([
                {'Factor':k,'Selected':v['kind'],'CV RMSE':v['inner_cv_rmse']}
                for k,v in v08['historical_challenger']['targets'].items()
            ]),hide_index=True,use_container_width=True)
            st.caption('Always compare the two versions on different matchups. '
                       'No manual inflation of team variance. Rebound rate models may stay league-based.')
    st.markdown('#### Pregame totals consistency audit')
    cc1,cc2,cc3,cc4=st.columns(4)
    structural=pred['home']['PTS']+pred['away']['PTS']
    cc1.metric('Pregame shared possessions',f'{pred["pace"]:.1f}')
    cc2.metric('Structural game total',f'{structural:.1f}')
    cc3.metric('Pace × adjusted efficiencies',f'{cross["projected_total_points"]:.1f}')
    cc4.metric('Disagreement (struct − benchmark)',f'{structural-cross["projected_total_points"]:+.1f}')
    with st.expander('Efficiency benchmark inputs and model selection'):
        st.write({'League PPP historical':round(cross['league_ppp'],3),'Home':cross['home'],
                  'Away':cross['away'],'Independent model':benchmark['selection'],
                  'Outer team-points RMSE with observed pace (rate-only)':benchmark['outer_team_ppp_model_rmse_pts'],
                  'Outer game-total RMSE with observed pace (rate-only)':benchmark['outer_total_ppp_model_rmse_pts']})
        st.caption('This comparator is NOT automatically blended into structural predictions. The historical evaluation uses observed possessions to isolate PPP accuracy only; upcoming projection uses pregame pace.')
    st.markdown('#### Probability calibration gate')
    if dist:
        if dist.get('production_pricing_gate') is not False:
            st.error('Calibration authorization mismatch — STOP');return
        records=[]
        for metric,info in dist['metrics'].items():
            records.append({'Market':metric,'90% interval actual coverage':f"{info['observed_90pct_coverage']:.1%}",'Outside statistical band': 'YES' if info['nominal_interval_coverage_flag'] else 'No',
                            'Mean forecast error':info['mean_error']})
        with st.expander('Historical 90% coverage — repeated holdout, not new OOS evidence'):
            st.dataframe(pd.DataFrame(records),use_container_width=True,hide_index=True)
        flagged=[name for name,r in dist['metrics'].items() if r['nominal_interval_coverage_flag']]
        if flagged:st.warning('Monte Carlo dispersion FAILED for '+', '.join(flagged)+'. Fair/bookmaker odds below are diagnostic ONLY.')
    else:
        st.warning('Distribution audit missing; book prices have no calibration evidence.')
    st.markdown('#### Trader view — fair and margin-included, BOTH uncalibrated')
    tabs=st.tabs(['Team O/U · 109%','Match Totals · 109%','Most Stat · 3-way 115%','Handicaps · 109%','Odds / value check'])
    for tab,key,cols in zip(tabs[:4],('teams','match','most','handicap'),
           (['Market','Line','Side','P','Fair','Book_109','Mean','MC_SD'],
            ['Market','Line','Side','P','Fair','Book_109','Mean','MC_SD'],
            ['Market','Side','P','Fair','Book_115'],
            ['Handicap_Home','Side','P','Fair','Book_109'])):
        with tab:
            frame=board[key]
            st.dataframe(_round_table(frame[cols],['Fair','Book_109','Book_115']),hide_index=True,use_container_width=True)
            st.download_button(f'Export {key} CSV',data=frame.to_csv(index=False).encode('utf-8'),file_name=f'euroleague_shadow_{key}.csv',mime='text/csv',key='v07_export_'+key)
    with tabs[4]:
        st.caption('Research EV comparison only. A positive shadow EV does NOT establish a playable bet.')
        frame=board['teams']; options=list(range(len(frame)))
        ix=st.selectbox('Select a model market',options,format_func=lambda i:f'{frame.iloc[i].Market} {frame.iloc[i].Side} {frame.iloc[i].Line}',key='v07_pick')
        odd=st.number_input('External bookmaker decimal odds',min_value=1.01,value=1.90,step=.01,key='v07_quote')
        z=evaluate_external_odds(float(frame.iloc[ix].P),odd)
        st.write(f'Shadow model P: {z["Model_p"]:.1%} · External break-even: {z["Break_even_p"]:.1%} · Uncalibrated model EV: {z["Model_EV"]:+.1%}')
        st.warning(z['Status'])
        st.markdown('##### Custom half-point O/U threshold')
        category=st.selectbox('Stat',[(k,name) for k,name,_ in MARKETS],format_func=lambda v:v[1],key='v07_custom_stat')
        who=st.selectbox('Scope',['Home','Away','Match'],key='v07_custom_scope')
        sampled=(sim['home'][category[0]] if who=='Home' else sim['away'][category[0]] if who=='Away'
                 else sim['home'][category[0]]+sim['away'][category[0]])
        default=float(np.floor(np.median(sampled))+.5)
        threshold=st.number_input('Half-point line',value=default,step=1.0,key='v07_custom_line')
        if np.isclose((threshold*2)%2,1):
            pr=threshold_counts(sampled,threshold);fair,bk=pricing_probabilities(pr,1.09)
            st.table(pd.DataFrame({'Side':['Over','Under'],'MC probability':pr,'Fair odds':fair,'Book 109%':bk}))
        else:st.error('Only x.5 thresholds; integer lines have a push and need a separate settlement model.')
    st.caption('Internal 100%/109%/115% overround identities are exact before display rounding. 3-way most points includes regulation tie, not OT. Pending: independent interval/probability calibration, opponent-role adjustments, PBP, absences.')
