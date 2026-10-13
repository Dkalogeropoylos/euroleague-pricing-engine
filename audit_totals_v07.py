"""Paired pregame holdout evaluation for v0.6 vs independent efficiency audit.

No refitting on heldout games. Both forecasts share the SAME *predicted* pace.
The 2026-01 onward evaluation window was already repeatedly analyzed;
these figures are diagnostics, not new untouched out-of-sample proof.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from core.structural_v06 import verified_games,event_rate_rows,chronological_design,project_two_sides
from core.efficiency_benchmark_v07 import chrono_ppp_design

def audit(data:pd.DataFrame, v06:dict,v07:dict)->dict:
    event=event_rate_rows(data)
    struct=chronological_design(event)
    eff=chrono_ppp_design(data,v07['prior_games'])
    ref=struct[['GAME_ID','TEAM_ID','HOME_AWAY','DAY']].merge(
        eff[['GAME_ID','TEAM_ID','BASE','X_OFF','X_DEF']],on=['GAME_ID','TEAM_ID'],how='inner',validate='one_to_one')
    spec=struct.merge(ref.drop(columns=['HOME_AWAY','DAY']),on=['GAME_ID','TEAM_ID'],validate='one_to_one')
    cutoff=pd.Timestamp(v07['outer_cutoff_day_utc'])
    spec=spec[spec.DAY>=cutoff]
    rec=[]
    model=v06['coefficients'] # Historical holdout uses the OUTER-fitted version
    m=v07['model'];c=np.asarray(m['coefficients'],float)
    for gid,g in spec.groupby('GAME_ID'):
        if len(g)!=2:continue
        h=g[g.HOME_AWAY.eq('H')].iloc[0];a=g[g.HOME_AWAY.eq('A')].iloc[0]
        p=project_two_sides(h,a,model)
        pace=p['pace']
        pp=[]
        for s in (h,a):
            x=np.array([s.X_OFF,s.X_DEF,s.HOME],float)
            pp.append(float(s.BASE*np.exp(np.clip(x@c+m['intercept'],-.30,.30))))
        actual=float(h.ACTUAL_PTS+a.ACTUAL_PTS)
        rec.append({'game':gid,'actual':actual,'model':pace*sum(pp),'structural':p['home']['PTS']+p['away']['PTS'],
                 'league':pace*(h.BASE+a.BASE),'pace_pred':pace,'pace_actual':float(h.PACE_TARGET)})
    x=pd.DataFrame(rec)
    if len(x)<50:raise RuntimeError('Too few paired pregame holdout games')
    out={'status':'SHADOW_PREGAME_TOTALS_AUDIT','games':len(x),'cutoff':str(cutoff),
         'method':'v06 pregame predicted possessions x v07 trained PPP, compared with v06 structural and league rate',
         'model_reused_holdout':True}
    for col in ('model','structural','league'):
        err=x[col]-x.actual
        out[col]={'rmse':float(np.sqrt(np.mean(err**2))),
                  'mae':float(np.mean(np.abs(err))),'bias':float(err.mean())}
    out['pace']={'rmse':float(np.sqrt(np.mean((x.pace_pred-x.pace_actual)**2))),
                  'bias':float(np.mean(x.pace_pred-x.pace_actual))}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('data/euroleague/exports'))
    ap.add_argument('--shadow',type=Path,default=Path('data/euroleague'))
    z=ap.parse_args()
    data=verified_games(pd.read_csv(z.data/'team_games.csv'),pd.read_csv(z.data/'player_games.csv'))
    v06=json.loads((z.shadow/'shadow_v06/structural_v06_report.json').read_text())
    v07=json.loads((z.shadow/'shadow_v07/efficiency_benchmark_v07.json').read_text())
    report=audit(data,v06,v07)
    outfile=z.shadow/'shadow_v07/pregame_totals_audit_v07.json'
    outfile.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('Pregame audit:',report)
if __name__=='__main__':main()
