"""EuroLeague v1.1 auditable shadow board. Fair and margin-odds are NOT production quotes."""
import json
from datetime import date,timedelta
from pathlib import Path
import numpy as np,pandas as pd
import streamlit as st
from core.matchup_v11 import tidy_data,live_features,projection,bootstrap_simulation,quality_gate,counts_audit,full_current_only_forecast
from core.trader_markets_v07 import build_trader_board,pricing_probabilities,MARKETS
from core.rotation_audit_v11 import current_rotation_audit

@st.cache_data(show_spinner=False,ttl=1800)
def _load_data(file,mtime):return tidy_data(pd.read_csv(file))

@st.cache_data(show_spinner=False,ttl=1800)
def _run_model(file,mtime,report_time,home,away,asof,draws):
 data=_load_data(file,mtime)
 base=Path(file).parent.parent
 r=json.loads((base/'shadow_v11'/'matchup_v11_report.json').read_text())
 pred,h,a=full_current_only_forecast(data,r,int(data.SEASON.max()),home,away,asof)
 sim=bootstrap_simulation(pred,r['matched_holdout_2025_residuals'],n=draws,seed=202611,sd=r['chosen_reconciliation_sd_2024'])
 return pred,sim,h,a

def render_trader_shadow(base:Path):
 st.error('SHADOW RESEARCH ONLY — ALL bookmaker odds on this page are illustrative, not safe for posting. Live injuries/rosters and independent future validation are missing.')
 teamfile=base/'team_games.csv'; model=base.parent/'shadow_v11'/'matchup_v11_report.json'
 cal=base.parent/'shadow_v11'/'distribution_v11_report.json'
 if not teamfile.exists() or not model.exists() or not cal.exists():
  st.warning('Missing current data or v1.1 report. Run updated GitHub Data Refresh daily workflow. Do NOT reuse legacy prices.')
  return
 r=json.loads(model.read_text());audit=json.loads(cal.read_text())
 if r.get('live_contract')!='SAME_SEASON_ONLY' or r.get('market_definition')!='official_inclusive_v2' or audit.get('production_pricing_gate') is not False:
  st.error('Data provenance / settlement / price safety mismatch — BLOCKED');return
 d=_load_data(str(teamfile),teamfile.stat().st_mtime)
 sea=int(d.SEASON.max());teams=d[d.SEASON.eq(sea)][['TEAM_ID','TEAM_NAME']].drop_duplicates().sort_values('TEAM_NAME')
 nam=dict(zip(teams.TEAM_ID,teams.TEAM_NAME));ids=list(nam)
 if len(ids)<2:return
 c1,c2,c3=st.columns(3)
 home=c1.selectbox('Home',ids,format_func=lambda z:nam[z],key='v11_home')
 away=c2.selectbox('Away',[x for x in ids if x!=home],format_func=lambda z:nam[z],key='v11_away')
 cutoff=d.GAME_DATE.max().date()+timedelta(days=1)
 asof=c3.date_input('As-of UTC date (no same-day games)',value=max(date.today(),cutoff),min_value=cutoff,key='v11_asof')
 st.caption('Data last game: '+str(d.GAME_DATE.max())+' UTC. Season-only source inputs. Historical seasons train coefficients and error distributions, never team priors.')
 with st.expander('Optional external market sanity check — never used to fit model'):
  use_external=st.checkbox('I have a verified market for this exact matchup and settlement',key='v11_extcheck')
  ext={}
  if use_external:
   ext['home_margin']=st.number_input('Market projected HOME winning margin (+ home favourite), 40-minute comparable',value=0.,step=.5,key='v11_extmargin')
   ext['total']=st.number_input('Market total points (same 40-minute settlement)',min_value=80.,value=170.,step=.5,key='v11_exttotal')
   st.warning('Bookmaker full-time INCLUDING overtime is not automatically comparable with this regulation-only model.')
 sims=st.select_slider('Paired historical-error bootstrap samples',options=[2000,4000,8000],value=4000,key='v11_draws')
 if st.button('Calculate current-only v1.1 SHADOW',type='primary',key='v11_run'):
  try:
   pred,sim,h,a=_run_model(str(teamfile),teamfile.stat().st_mtime,model.stat().st_mtime,home,away,asof,sims)
   check=quality_gate(pred,h,a,ext if use_external else None,audit)
   board=build_trader_board(sim,nam[home],nam[away])
   st.session_state['v11_result']={'match':(home,away,str(asof),sims),
       'pred':pred,'sim':sim,'audit':counts_audit(pred,h,a),'check':check,'board':board}
  except Exception as ex:
   st.error('Forecast refused: '+str(ex));return
 result=st.session_state.get('v11_result')
 if not result or result['match']!=(home,away,str(asof),sims):
  st.info('Select matchup and run the current-season-only audit.');return
 pr=result['pred'];m=result['audit'];check=result['check'];board=result['board'];sim=result['sim']
 st.error('MARKET RELEASE: '+check['status']+' — prices are NOT approved for traders. No per-player injury/rotation model.')
 for f in check['flags']:st.caption('• '+f)
 o=pr['home']['PTS']+pr['away']['PTS']
 c1,c2,c3,c4=st.columns(4)
 c1.metric('Pace',f'{pr["pace"]:.1f}')
 c2.metric(nam[home]+' points',f'{pr["home"]["PTS"]:.1f}')
 c3.metric(nam[away]+' points',f'{pr["away"]["PTS"]:.1f}')
 c4.metric('40-minute game total',f'{o:.1f}')
 st.markdown('#### Pregame opponent audit — each number uses THIS season only')
 rows=[]
 for side,short,opp in [('home',home,away),('away',away,home)]:
  for stat,key in [('3PA','FG3A'),('REB','MARKET_REB')]:
   rows.append({'Team':nam[short],'Market':stat,'Own season estimate':m[side]['3PA_current_own' if stat=='3PA' else 'REB_current_own'],
                'Opponent allowed':m[side]['3PA_allowed_opponent' if stat=='3PA' else 'REB_allowed_opponent'],
                'Projected':m[side]['3PA_final' if stat=='3PA' else 'REB_final']})
 st.dataframe(pd.DataFrame(rows).round(2),hide_index=True,use_container_width=True)
 with st.expander('Current-season player minutes, shot distribution and opponent position-category audit'):
  pp=base/'player_games.csv'
  if pp.exists():
   playerdf=pd.read_csv(pp)
   for team,opp in ((home,away),(away,home)):
    own,allowed,flags=current_rotation_audit(playerdf,sea,team,opp,asof)
    st.write('**'+nam[team]+'**')
    if not own.empty:st.dataframe(own.round(2),hide_index=True,use_container_width=True)
    if not allowed.empty:
     st.caption('Opponent team '+nam[opp]+' allowed 3PA by player POSITION raw category; descriptive, no role-matched coefficients')
     st.dataframe(allowed.round(2),hide_index=True,use_container_width=True)
    for flag in flags:st.warning(flag)
   st.caption('Historical appearances are not an injury report. Verify CURRENT match availability, rotation, coach changes and minutes before operational use.')
  else:st.warning('Player_games.csv absent: lineup and position audit cannot run; HOLD')
 st.markdown('#### Joint team means — official team rebounds and turnovers')
 marks=[('Points','PTS'),('2PA','FG2A'),('2PM','FG2M'),('3PA','FG3A'),('3PM','FG3M'),('FTA','FTA'),('FTM','FTM'),('OREB','MARKET_OREB'),('DREB','MARKET_DREB'),('REB','MARKET_REB'),('AST','AST'),('TO','MARKET_TO'),('STL','STL'),('BLK','BLK')]
 means=pd.DataFrame([{'Market':name,nam[home]:round(pr['home'][k],2),nam[away]:round(pr['away'][k],2),
                       'Combined':round(pr['home'][k]+pr['away'][k],2)} for name,k in marks])
 st.dataframe(means,hide_index=True,use_container_width=True)
 with st.expander('Statistical proof, score accounting, current-season dates'):
  st.write({'Training':r['training_seasons'],'Parameter choice':r['selection_season'],
            'Historical test':r['untuned_test_season'],'Current season':sea,
            'Current-season watch side-observations':audit['side_observations'],
            'Independent market approval':False})
  st.dataframe(pd.DataFrame([{'Market':k,'Prior games':v['prior'],'Opponent coefficient':v['beta'],
                 '2025 historical RMSE':r['test_2025'].get(k,{}).get('rmse')}
           for k,v in r['params'].items()]).round(3),use_container_width=True,hide_index=True)
  st.dataframe(pd.DataFrame([{'Market':k,'2026 watch 90% interval coverage':v['coverage_90'],'Watch RMSE':v['RMSE']} for k,v in audit['markets'].items()]).round(3),use_container_width=True,hide_index=True)
  st.caption('Points=2×2PM+3×3PM+FTM, REB=OREB+DREB. Possession proxy is reconciled through attempts, FTAs, TO and offensive rebounds. FT rebounding opportunity remains approximate without PBP.')
 st.markdown('#### Illustrative fair odds versus 109% / 115% — ALWAYS HOLD')
 tabs=st.tabs(['Team O/U 109%','Game combined 109%','Most stat 3-way 115%','Handicap 109%'])
 for tab,k,cols in zip(tabs,('teams','match','most','handicap'),
   (['Market','Line','Side','P','Fair','Book_109','Mean','MC_SD'],
    ['Market','Line','Side','P','Fair','Book_109','Mean','MC_SD'],
    ['Market','Side','P','Fair','Book_115'],
    ['Handicap_Home','Side','P','Fair','Book_109'])):
  with tab:
   q=board[k][cols].copy()
   for f in ('Fair','Book_109','Book_115'):
    if f in q:q[f]=q[f].apply(lambda x:f'{x:.2f}' if np.isfinite(x) else 'HOLD')
   q['Release']='HOLD (uncalibrated / no injury verification)'
   st.dataframe(q,use_container_width=True,hide_index=True)
   st.download_button('Export '+k+' SHADOW CSV',data=board[k].assign(Release='HOLD').to_csv(index=False).encode('utf-8'),
    file_name='EUROLEAGUE_V11_SHADOW_'+k+'.csv',mime='text/csv',key='v11_dl_'+k)
 st.caption('NEVER copy 109%/115% outputs directly to a live bookmaker feed. All markets must pass injury, venue, date and independent calibration checks first.')
