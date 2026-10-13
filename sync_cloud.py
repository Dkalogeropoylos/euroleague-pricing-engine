"""Incremental GitHub Actions ingestion and training. No backfill in Streamlit web requests.

External API ingestion here is intentionally rate-limited and strict. Only validated
team/player summaries are checked into Git; raw responses are ephemeral in Actions.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from providers.euroleague import EuroLeagueProvider, scalar
from league_coefficients import clean_games, estimate_coefficients

EXPORT_DIR = Path('data/euroleague/exports')


def current_season_start() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def old_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_csv(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.tmp')
    df.to_csv(temp, index=False)
    temp.replace(target)


def combine_games(old: pd.DataFrame, incoming: list[dict], keys: list[str]) -> pd.DataFrame:
    fresh = pd.DataFrame(incoming)
    if old.empty:
        merged = fresh
    elif fresh.empty:
        merged = old.copy()
    else:
        merged = pd.concat([old, fresh], ignore_index=True)
    if merged.empty:
        return merged
    if any(k not in merged.columns for k in keys):
        raise ValueError('Missing required canonical identifier fields in merged exported rows')
    # Reject collisions rather than arbitrarily choosing a game version.
    if merged.duplicated(keys).any():
        dupes = merged.loc[merged.duplicated(keys, keep=False), keys].head(8)
        raise ValueError(f'Duplicate export identifiers: {dupes.to_dict(orient="records")}')
    return merged.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)


def sync(seasons: list[int], export_dir: Path = EXPORT_DIR, max_new_games: int | None = None,
         provider: EuroLeagueProvider | None = None) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    team_path = export_dir / 'team_games.csv'
    player_path = export_dir / 'player_games.csv'
    audit_path = export_dir / 'data_audit.csv'
    teams_old = old_csv(team_path)
    players_old = old_csv(player_path)
    audits_old = old_csv(audit_path)
    existing_ids = set(teams_old.GAME_ID.astype(str)) if 'GAME_ID' in teams_old else set()
    provider = provider or EuroLeagueProvider(root='data/euroleague', competition='E', minimum_interval=7.0)
    incoming_team: list[dict] = []
    incoming_player: list[dict] = []
    issues: list[dict] = []
    total_new_games = 0
    for year in sorted(set(seasons)):
        schedule_path = provider.root / provider.competition / f'E{year}' / 'schedule_0.json'
        # Schedule is refetched each Actions run. Historical game boxscore changes are
        # NOT auto reprocessed: revisions require controlled manual rebuild.
        schedule_path.unlink(missing_ok=True)
        try:
            games = provider.games(year)
        except Exception as exc:
            raise RuntimeError(f'Could not fetch official schedule for E{year}; not committing partial data') from exc
        for game in sorted(games, key=lambda x: str(x.get('date') or x.get('localDate') or '')):
            if not provider._ready(game):
                continue
            phase = scalar(game.get('phaseType', 'RS')).upper()
            if phase not in {'RS', 'REGULAR SEASON', 'REGULAR_SEASON'}:
                continue
            gid = f'E{year}_{int(game["gameCode"])}'
            if gid in existing_ids:
                continue
            if max_new_games is not None and total_new_games >= max_new_games:
                break
            try:
                trows, prows, errors = provider.normalize_game(
                    game, provider.game_stats(year, int(game['gameCode'])), 'E', year
                )
                if errors:
                    issues.extend({'GAME_ID': gid, 'ISSUE': str(err)} for err in errors)
                    continue
                if len(trows) != 2 or len(prows) < 10:
                    issues.append({'GAME_ID': gid, 'ISSUE': 'Incomplete parsed boxscore/team/player rows'})
                    continue
                if len({str(t['TEAM_ID']) for t in trows}) != 2:
                    issues.append({'GAME_ID': gid, 'ISSUE': 'Two unique teams required'})
                    continue
                incoming_team.extend(trows)
                incoming_player.extend(prows)
                existing_ids.add(gid)
                total_new_games += 1
            except Exception as exc:
                issues.append({'GAME_ID': gid, 'ISSUE': f'{type(exc).__name__}: {exc}'})
        print(f'E{year}: added {total_new_games} games so far', flush=True)

    # Refuse to commit data from a malformed historical file or a broken parser.
    teams_all = combine_games(teams_old, incoming_team, ['GAME_ID', 'TEAM_ID'])
    players_all = combine_games(players_old, incoming_player, ['GAME_ID', 'PLAYER_ID'])
    if not teams_all.empty:
        clean_games(teams_all)  # checks two-sided, unique identities, date and possessions
    audits_new = pd.DataFrame(issues, columns=['GAME_ID', 'ISSUE'])
    audits_all = pd.concat([audits_old, audits_new], ignore_index=True).drop_duplicates() if not audits_old.empty else audits_new

    # Fit BEFORE overwriting any exported artifact. A good data refresh must not
    # discard the prior coefficient file merely because the data sample is small.
    fitted = None
    if len(teams_all) >= 100 and total_new_games > 0:
        try:
            fitted = estimate_coefficients(teams_all)
        except ValueError as exc:
            print('Calibration deferred:', exc, flush=True)

    if total_new_games > 0:
        atomic_csv(teams_all, team_path)
        atomic_csv(players_all, player_path)
        if fitted is not None:
            coefficient_path = export_dir / 'league_coefficients.json'
            temp = coefficient_path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(fitted, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(coefficient_path)
        info_path = export_dir / 'manifest.json'
        temp = info_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps({
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'games': int(len(teams_all) // 2),
            'new_games': total_new_games,
            'seasons': sorted({int(x) for x in teams_all.SEASON.dropna().unique()}),
            'coefficient_stage': 'shadow_only_not_pricing_ready',
        }, indent=2), encoding='utf-8')
        temp.replace(info_path)
    if issues:
        atomic_csv(audits_all, audit_path)
    if total_new_games == 0:
        print('No new validated games. Existing CSV/model unchanged.', flush=True)
    return {'new_games': total_new_games, 'issues': len(issues), 'all_games': int(len(teams_all) // 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['daily', 'bootstrap'], default='daily')
    parser.add_argument('--seasons', type=int, nargs='*')
    parser.add_argument('--max-new-games', type=int, default=None)
    args = parser.parse_args()
    seasons = args.seasons or (
        [current_season_start()] if args.mode == 'daily' else
        list(range(current_season_start()-3, current_season_start()+1))
    )
    print(sync(seasons=seasons, max_new_games=args.max_new_games))


if __name__ == '__main__':
    main()
