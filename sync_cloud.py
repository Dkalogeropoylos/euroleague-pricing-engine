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
from market_truth import attach_market_truth, market_audit, MARKET_SCHEMA_VERSION

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
    # Existing exports may come from an older parser that did not understand
    # EuroLeague's blocksFavour field. Reparse affected games rather than
    # preserving missing blocks forever. Correct data is never overwritten.
    repair_ids: set[str] = set()
    for previous in (teams_old, players_old):
        if not previous.empty and 'GAME_ID' in previous.columns:
            if 'BLK' not in previous.columns:
                repair_ids.update(previous.GAME_ID.astype(str).tolist())
            else:
                missing = pd.to_numeric(previous.BLK, errors='coerce').isna()
                repair_ids.update(previous.loc[missing, 'GAME_ID'].astype(str).tolist())
    provider = provider or EuroLeagueProvider(root='data/euroleague', competition='E', minimum_interval=7.0)
    incoming_team: list[dict] = []
    incoming_player: list[dict] = []
    issues: list[dict] = []
    total_new_games = 0
    repaired_ids: set[str] = set()
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
            if gid in existing_ids and gid not in repair_ids:
                continue
            if max_new_games is not None and total_new_games + len(repaired_ids) >= max_new_games:
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
                # Blocks missing is not zero blocks. Ingestion must not silently
                # backfill unknown stats with zeros or claim them as complete.
                if any(t.get('BLK') is None for t in trows) or any(p.get('BLK') is None for p in prows):
                    issues.append({'GAME_ID': gid, 'ISSUE': 'BLK missing from upstream game boxscore'})
                    continue
                incoming_team.extend(trows)
                incoming_player.extend(prows)
                if gid in repair_ids:
                    repaired_ids.add(gid)
                else:
                    existing_ids.add(gid)
                    total_new_games += 1
            except Exception as exc:
                issues.append({'GAME_ID': gid, 'ISSUE': f'{type(exc).__name__}: {exc}'})
        print(f'E{year}: added {total_new_games} new games; repaired {len(repaired_ids)} old games', flush=True)

    # Refuse to commit data from a malformed historical file or a broken parser.
    if repaired_ids:
        teams_old = teams_old.loc[~teams_old.GAME_ID.astype(str).isin(repaired_ids)].copy()
        players_old = players_old.loc[~players_old.GAME_ID.astype(str).isin(repaired_ids)].copy()
    teams_all = combine_games(teams_old, incoming_team, ['GAME_ID', 'TEAM_ID'])
    players_all = combine_games(players_old, incoming_player, ['GAME_ID', 'PLAYER_ID'])
    # Always reconcile credited player totals AND set official-inclusive markets for all seasons.
    # This also migrates already-committed exports without fetching games again.
    migration = (not teams_old.empty and
                 ('MARKET_SCHEMA_VERSION' not in teams_old.columns or
                  not (teams_old['MARKET_SCHEMA_VERSION'].astype(str) == MARKET_SCHEMA_VERSION).all()))
    contract_audit = {}
    if not teams_all.empty:
        teams_all = attach_market_truth(teams_all, players_all)  # fail-closed validation
        contract_audit = market_audit(teams_all)
        clean_games(teams_all)  # validates two-sided, IDs, date, market columns and pace
    audits_new = pd.DataFrame(issues, columns=['GAME_ID', 'ISSUE'])
    audits_all = pd.concat([audits_old, audits_new], ignore_index=True).drop_duplicates() if not audits_old.empty else audits_new

    # Fit the NEW target schema BEFORE overwriting exported artifacts. Old
    # players_only coefficients must never masquerade as official-inclusive.
    fitted = None
    # Training gate: >=3 distinct seasons, >=500 validated games. This is
    # NOT a calibrated pricing certificate; models remain shadow-only.
    enough_history = (len(teams_all) >= 1000 and teams_all.SEASON.nunique() >= 3)
    coefficient_path = export_dir / "league_coefficients.json"
    old_coeff_schema = None
    if coefficient_path.exists():
        try:
            old_coeff_schema = json.loads(coefficient_path.read_text()).get("market_schema_version")
        except (ValueError, OSError):
            old_coeff_schema = "corrupt"
    stale_coefficients = bool(enough_history and old_coeff_schema != MARKET_SCHEMA_VERSION)
    if enough_history and (total_new_games > 0 or repaired_ids or migration or stale_coefficients):
        try:
            fitted = estimate_coefficients(teams_all)
        except ValueError as exc:
            if migration or stale_coefficients:
                raise RuntimeError('Official-inclusive coefficient retraining failed: refusing partial migration') from exc
            print('Calibration deferred:', exc, flush=True)

    if total_new_games > 0 or repaired_ids or migration or stale_coefficients:
        atomic_csv(teams_all, team_path)
        atomic_csv(players_all, player_path)
        if fitted is None and (migration or stale_coefficients) and enough_history:
            raise RuntimeError('Updated settlement contract without updated coefficients: refused')
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
            'repaired_games': len(repaired_ids),
            'seasons': sorted({int(x) for x in teams_all.SEASON.dropna().unique()}),
            'coefficient_stage': 'shadow_only_not_pricing_ready',
            'market_schema_version': MARKET_SCHEMA_VERSION,
            'market_audit': contract_audit,
            'training_gate': {'min_seasons': 3, 'min_games': 500, 'met': bool(enough_history)},
        }, indent=2), encoding='utf-8')
        temp.replace(info_path)
    if issues:
        atomic_csv(audits_all, audit_path)
    if total_new_games == 0 and not repaired_ids and not migration and not stale_coefficients:
        print('No new validated games. Existing CSV/model unchanged.', flush=True)
    return {'new_games': total_new_games, 'repaired_games': len(repaired_ids), 'migrated_market_schema': bool(migration), 'issues': len(issues), 'all_games': int(len(teams_all) // 2)}


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
