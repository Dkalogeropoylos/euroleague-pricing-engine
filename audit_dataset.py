"""EuroLeague Pricing Engine | non-destructive historical dataset preflight.

Run: python audit_dataset.py --team 'team_games.csv' --player 'player_games.csv' --out audit.json
No network, no model fitting, no write to original inputs.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

TEAM_FIELDS = ('PTS','FG2M','FG2A','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TO','PF','FD')
EXACT_PLAYER_SUM = ('PTS','FG2M','FG2A','FG3M','FG3A','FTM','FTA','AST','STL','BLK','PF','FD')
PLAYER_SUM_ALLOW_TEAM = ('OREB','DREB','REB','TO')
MIRROR_FIELDS = ('PTS','FG2M','FG2A','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TO','PF','FD')
REQ_TEAM = set(TEAM_FIELDS) | {'SEASON','GAME_ID','GAME_DATE','TEAM_ID','OPP_ID','HOME_AWAY','POSS_RAW','TEAM_MINUTES','OT_FLAG'}
REQ_PLAYER = set(TEAM_FIELDS) | {'SEASON','GAME_ID','GAME_DATE','TEAM_ID','PLAYER_ID','MINUTES','POSITION'}


def audit(team: pd.DataFrame, player: pd.DataFrame, min_seasons: int = 3, min_games: int = 500) -> dict:
    """Build an audit without using future games or claiming model predictive validity."""
    fatal: list[str] = []
    warnings: list[str] = []
    summary = {'team_rows': int(len(team)), 'player_rows': int(len(player))}
    missing_t, missing_p = sorted(REQ_TEAM - set(team)), sorted(REQ_PLAYER - set(player))
    if missing_t: fatal.append(f'TEAM_MISSING_COLUMNS: {missing_t}')
    if missing_p: fatal.append(f'PLAYER_MISSING_COLUMNS: {missing_p}')
    if missing_t or missing_p:
        return {'summary': summary, 'fatal': fatal, 'warnings': warnings, 'ready_for_training': False}
    t,p=team.copy(),player.copy()
    for frame, cols in ((t, TEAM_FIELDS + ('POSS_RAW','TEAM_MINUTES')),
                        (p, TEAM_FIELDS + ('MINUTES',))):
        for c in cols: frame[c]=pd.to_numeric(frame[c],errors='coerce')
    t['parsed_date']=pd.to_datetime(t.GAME_DATE,utc=True,errors='coerce')
    p['parsed_date']=pd.to_datetime(p.GAME_DATE,utc=True,errors='coerce')
    if t.duplicated(['GAME_ID','TEAM_ID']).any(): fatal.append('DUPLICATE_GAME_TEAM')
    if p.duplicated(['GAME_ID','PLAYER_ID']).any(): fatal.append('DUPLICATE_GAME_PLAYER')
    if t.parsed_date.isna().any() or p.parsed_date.isna().any(): fatal.append('INVALID_GAME_DATES')
    if t[list(TEAM_FIELDS)].isna().any().any(): fatal.append('MISSING_TEAM_BOX_SCORE_STATS')
    if p[list(TEAM_FIELDS)].isna().any().any(): fatal.append('MISSING_PLAYER_BOX_SCORE_STATS')
    if (t[list(TEAM_FIELDS)] < 0).any().any() or (p[list(TEAM_FIELDS)] < 0).any().any():
        fatal.append('NEGATIVE_STAT_COUNTS')
    if p.MINUTES.isna().any() or (p.MINUTES<0).any(): fatal.append('BAD_PLAYER_MINUTES')
    games = t.groupby('GAME_ID',dropna=False)
    bad_pairs = int((games.size()!=2).sum())
    if bad_pairs: fatal.append(f'INCOMPLETE_GAME_PAIRS={bad_pairs}')
    valid_pairs = games.filter(lambda g: len(g)==2)
    if not valid_pairs.empty:
        mismatch=[]
        for game_id, group in valid_pairs.groupby('GAME_ID'):
            r0,r1=group.iloc[0],group.iloc[1]
            if (str(r0.TEAM_ID)!=str(r1.OPP_ID) or str(r1.TEAM_ID)!=str(r0.OPP_ID)
               or {str(r0.HOME_AWAY),str(r1.HOME_AWAY)}!={'H','A'}
               or r0.SEASON!=r1.SEASON or r0.GAME_DATE!=r1.GAME_DATE):
                mismatch.append(game_id)
        if mismatch: fatal.append(f'GAME_PAIR_IDENTITY_MISMATCH={len(mismatch)}')
    team_ids = set(t.GAME_ID.astype(str))
    player_ids = set(p.GAME_ID.astype(str))
    if team_ids!=player_ids: fatal.append(f'MISSING_GAME_IN_OTHER_TABLE: team_only={len(team_ids-player_ids)} player_only={len(player_ids-team_ids)}')
    if (p.groupby(['GAME_ID','TEAM_ID']).size().index.difference(t.set_index(['GAME_ID','TEAM_ID']).index).size)>0:
        fatal.append('PLAYER_TEAM_NOT_IN_GAME')
    for frame, label in ((t,'TEAM'),(p,'PLAYER')):
        err=frame.PTS-(2*frame.FG2M+3*frame.FG3M+frame.FTM)
        if (err.abs()>.01).any(): fatal.append(f'{label}_SCORING_IDENTITY_BROKEN={int((err.abs()>.01).sum())}')
        for attempts,makes in [('FG2A','FG2M'),('FG3A','FG3M'),('FTA','FTM')]:
            if (frame[attempts]+.001<frame[makes]).any():fatal.append(f'{label}_{attempts}_LESS_THAN_{makes}')
        if (frame.REB-frame.OREB-frame.DREB).abs().gt(.01).any():
            fatal.append(f'{label}_REB_IDENTITY_BROKEN')
    sums=p.groupby(['GAME_ID','TEAM_ID']).agg({c:'sum' for c in TEAM_FIELDS + ('MINUTES',)})
    sums=sums.rename(columns={c:c+'_PLAYER' for c in sums.columns})
    aligned=t.set_index(['GAME_ID','TEAM_ID']).join(sums,how='left')
    for c in EXACT_PLAYER_SUM:
        missing=aligned[c+'_PLAYER'].isna()
        bad=(aligned[c]-aligned[c+'_PLAYER']).abs().gt(.01)
        if missing.any() or bad.any():fatal.append(f'PLAYER_TEAM_{c}_MISMATCH={int((missing|bad).sum())}')
    residuals={}
    for c in PLAYER_SUM_ALLOW_TEAM:
        delta=aligned[c]-aligned[c+'_PLAYER']
        residuals[c]={'mean':float(delta.mean()),'max':float(delta.max()),'nonzero_team_rows':int((delta>.01).sum())}
        if delta.isna().any() or (delta<-.01).any():
            fatal.append(f'PLAYER_SUM_EXCEEDS_TEAM_{c}')
    mins=aligned['MINUTES_PLAYER']
    if (mins-aligned.TEAM_MINUTES).abs().gt(.01).any(): fatal.append('TEAM_MINUTES_NOT_EQUAL_PLAYER_MINUTES')
    # OT game must provide >=225 player-minutes per team for 5-minute overtime.
    reg=t.OT_FLAG.astype(str).str.lower().isin(['false','0','no'])
    if reg.any() and (aligned.loc[reg.to_numpy(),'MINUTES_PLAYER']-200).abs().gt(1.0).any():
        fatal.append('REGULATION_PLAYER_MINUTES_NOT_200')
    pos_counts=p.POSITION.value_counts(dropna=False).to_dict()
    # EuroLeague feed here codes guard/forward/center in 1,2,3; DO NOT assume five exact positions.
    if not set(p.POSITION.astype(str).unique()).issubset({'1','2','3'}):
        warnings.append('POSITION_CODES_OUTSIDE_1_2_3: inspect source mapping; never infer 5-position defense automatically')
    pair_poss=t.groupby('GAME_ID').POSS_RAW.agg(lambda x:float(max(x)-min(x)) if len(x)==2 else float('nan'))
    poss_large=int((pair_poss>3).sum())
    if poss_large:
        warnings.append(f'POSS_ESTIMATE_SIDES_DISAGREE_GT_3_IN_{poss_large}_GAMES: pregame pace must be modeled separately')
    # Check existing OPP_X fields actually mirror opposing team, not stale.
    mirror_bad = {}
    for f in MIRROR_FIELDS:
        c='OPP_'+f
        if c in t.columns:
            comp=[]
            for _,grp in valid_pairs.groupby('GAME_ID'):
                a,b=grp.iloc[0],grp.iloc[1]
                comp.extend([pd.isna(a[c]) or abs(float(a[c])-float(b[f]))>0.01,
                             pd.isna(b[c]) or abs(float(b[c])-float(a[f]))>0.01])
            mirror_bad[f]=sum(comp)
    if any(mirror_bad.values()):fatal.append('OPP_STATS_NOT_MIRRORED: '+str({k:v for k,v in mirror_bad.items() if v}))
    seasons={str(int(k)):int(v) for k,v in t.groupby('SEASON').GAME_ID.nunique().items()}
    summary.update({'games':int(t.GAME_ID.nunique()),'seasons':seasons,
      'teams':int(t.TEAM_ID.nunique()),'players':int(p.PLAYER_ID.nunique()),
      'date_min_utc':str(t.parsed_date.min()),'date_max_utc':str(t.parsed_date.max()),
      'exact_stat_checks':list(EXACT_PLAYER_SUM),'team_only_residuals':residuals,
      'position_codes':{str(k):int(v) for k,v in pos_counts.items()},
      'estimated_possessions_per_team_game_mean':float(t.POSS_RAW.mean()),
      'estimated_possessions_pair_discrepancy_max':float(pair_poss.max()),
      'estimated_possessions_pair_discrepancy_gt_3_games':poss_large,
      'missing_optional_team_MINUTES':int(t.MINUTES.isna().sum()) if 'MINUTES' in t else None})
    ready=(not fatal and len(seasons)>=min_seasons and summary['games']>=min_games)
    if not ready and not fatal:
        warnings.append(f'HISTORICAL_COVERAGE_NOT_READY: {len(seasons)} seasons/{summary["games"]} games; model-training gate {min_seasons} seasons and {min_games} games')
    return {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'summary':summary,'fatal':fatal,'warnings':warnings,
            'data_integrity_passed':not fatal,
            'ready_for_training':ready,
            'note':'Training readiness is a data-coverage gate, not proof of predictive value or market-calibrated odds.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--team',required=True)
    parser.add_argument('--player',required=True)
    parser.add_argument('--out',default='data_quality_report.json')
    args=parser.parse_args()
    report=audit(pd.read_csv(args.team,low_memory=False), pd.read_csv(args.player,low_memory=False))
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('Integrity:', 'PASS' if report.get('data_integrity_passed') else 'FAIL')
    print('Training data gate:', 'READY' if report.get('ready_for_training') else 'NOT READY')
    print('Historical seasons:',report['summary'].get('seasons'))
    print('Warnings:',report.get('warnings'))
    print('Failures:',report.get('fatal'))
    if report.get('fatal'): raise SystemExit(1)

if __name__=='__main__':
    main()
