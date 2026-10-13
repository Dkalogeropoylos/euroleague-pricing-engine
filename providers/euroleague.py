"""EuroLeague public-data adapter, independent of the production WNBA engine.

Sources: https://api-live.euroleague.net/v2 (competition schedule & game stats)
         https://live.euroleague.net/api (later optional PBP/shots)
Not affiliated with, endorsed by or guaranteed by Euroleague Basketball.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE = "https://api-live.euroleague.net"
NUMERIC = {
    "PTS": ("points",),
    "FG2M": ("fieldGoalsMade2", "twoPointsMade"),
    "FG2A": ("fieldGoalsAttempted2", "twoPointsAttempted"),
    "FG3M": ("fieldGoalsMade3", "threePointsMade"),
    "FG3A": ("fieldGoalsAttempted3", "threePointsAttempted"),
    "FTM": ("freeThrowsMade",),
    "FTA": ("freeThrowsAttempted",),
    "OREB": ("offensiveRebounds", "reboundsOffensive"),
    "DREB": ("defensiveRebounds", "reboundsDefensive"),
    "REB": ("totalRebounds", "reboundsTotal", "rebounds"),
    "AST": ("assistances", "assists"),
    "STL": ("steals",),
    "BLK": ("blocksFavour", "blocks", "blocksMade"),
    "TO": ("turnovers",),
    "PF": ("foulsCommited", "foulsCommitted", "personalFouls"),
    "FD": ("foulsReceived", "foulsDrawn"),
    "PIR": ("valuation", "pir"),
}


def scalar(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, dict):
        for key in ("code", "name", "fullName", "clubPermanentName", "editorialName", "abbreviatedName", "id"):
            if value.get(key) is not None:
                return str(value[key]).strip()
        return default
    return str(value).strip()


def number(obj: dict, fields: tuple[str, ...], default: float | None = None) -> float | None:
    for key in fields:
        val = obj.get(key)
        if val is not None and val != "":
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return default


def minutes(raw: Any) -> float | None:
    """The v2 feed uses seconds, but older feeds may return MM:SS."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, str) and ":" in raw:
        try:
            m, s = raw.split(":", 1)
            return int(m) + int(s) / 60
        except ValueError:
            return None
    try:
        return float(raw) / 60
    except (TypeError, ValueError):
        return None


def team_identity(side: Any) -> tuple[str, str]:
    if not isinstance(side, dict):
        return scalar(side), scalar(side)
    club = side.get("club", side.get("team", side))
    if isinstance(club, dict):
        code = scalar(club.get("code", club.get("abbreviatedName")))
        name = scalar(club.get("name", club.get("clubPermanentName", club.get("editorialName"))))
        return (code or name, name or code)
    return scalar(club), scalar(club)


def player_identity(value: Any) -> tuple[str, str]:
    """Never merge EuroLeague ids with NBA/WNBA ids."""
    if not isinstance(value, dict):
        s = scalar(value)
        return s, s
    person = value.get("person", value)
    if isinstance(person, dict):
        return scalar(person.get("code", person.get("id"))), scalar(person.get("name", person.get("fullName")))
    return scalar(person), scalar(value.get("name", person))


def numeric_row(raw: dict) -> dict:
    out = {dest: number(raw, keys) for dest, keys in NUMERIC.items()}
    if out["REB"] is None and out["OREB"] is not None and out["DREB"] is not None:
        out["REB"] = out["OREB"] + out["DREB"]
    out["FGA"] = (out["FG2A"] + out["FG3A"]) if out["FG2A"] is not None and out["FG3A"] is not None else None
    out["FGM"] = (out["FG2M"] + out["FG3M"]) if out["FG2M"] is not None and out["FG3M"] is not None else None
    return out


class EuroLeagueProvider:
    def __init__(self, root: str | Path = "data/euroleague", competition: str = "E", minimum_interval: float = 7.0, timeout: float = 25.0):
        if competition not in {"E", "U"}:
            raise ValueError("Competition must be E (EuroLeague) or U (EuroCup)")
        self.root = Path(root)
        self.competition = competition
        self.minimum_interval = float(minimum_interval)
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BasketballAnalyticsResearch/0.1 (cache-first)", "Accept": "application/json"})
        self._last_request = -1e10

    def season_code(self, season: int) -> str:
        if not 2000 <= int(season) <= 2100:
            raise ValueError("Season means starting calendar year, e.g. 2025 = E2025")
        return f"{self.competition}{int(season)}"

    def _fetch(self, endpoint: str, cache_name: str, params: dict | None = None) -> Any:
        path = self.root / self.competition / cache_name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        path.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + endpoint
        for attempt in range(4):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    if attempt == 3:
                        resp.raise_for_status()
                    ra = resp.headers.get("Retry-After", "")
                    try:
                        pause = max(float(ra), 3.0)
                    except ValueError:
                        pause = min(300, 10 * 2**attempt)
                    time.sleep(pause)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                if not payload:
                    raise ValueError(f"Empty payload {url}")
                # Atomic cache ensures interrupted jobs remain resumable.
                temp = path.with_suffix(path.suffix + ".tmp")
                temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                os.replace(temp, path)
                return payload
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(min(60, 3 * 2**attempt))
        raise RuntimeError("Unexpected HTTP retry exit")

    def games(self, season: int) -> list[dict]:
        code = self.season_code(season)
        # Most seasons fit in one response; paginate defensively.
        result = []
        seen = set()
        page_size = 500
        for offset in range(0, 5000, page_size):
            data = self._fetch(
                f"/v2/competitions/{self.competition}/seasons/{code}/games",
                f"{code}/schedule_{offset}.json", {"limit": page_size, "offset": offset},
            )
            rows = data.get("data", []) if isinstance(data, dict) else data
            if not isinstance(rows, list):
                raise ValueError("Unexpected schedule response; check API version")
            for row in rows:
                game_id = row.get("gameCode")
                if game_id is None:
                    raise ValueError("Schedule row lacks gameCode")
                if game_id not in seen:
                    seen.add(game_id)
                    result.append(row)
            total = data.get("total") if isinstance(data, dict) else None
            if not rows or (total is not None and len(seen) >= int(total)) or len(rows) < page_size:
                break
        return result

    def game_stats(self, season: int, game_code: int) -> dict:
        code = self.season_code(season)
        stats = self._fetch(
            f"/v2/competitions/{self.competition}/seasons/{code}/games/{int(game_code)}/stats",
            f"{code}/stats/{int(game_code)}.json",
        )
        if not isinstance(stats, dict) or "local" not in stats or "road" not in stats:
            raise ValueError(f"Game {game_code}: unexpected stats structure")
        return stats

    @staticmethod
    def _ready(game: dict) -> bool:
        if game.get("played") is True:
            return True
        # Do not guess based on game date. Never ingest future fixtures as 0-0 results.
        return False

    @staticmethod
    def normalize_game(game: dict, payload: dict, competition: str, season: int) -> tuple[list[dict], list[dict], list[str]]:
        errors = []
        gc = int(game["gameCode"])
        game_date = game.get("date") or game.get("localDate")
        date = pd.to_datetime(game_date, utc=True, errors="coerce")
        if pd.isna(date):
            errors.append(f"game={gc}: missing/bad game date")
        date_fmt = date.isoformat() if pd.notna(date) else ""
        local_id, local_name = team_identity(game.get("local", {}))
        road_id, road_name = team_identity(game.get("road", {}))
        if not local_id or not road_id or local_id == road_id:
            errors.append(f"game={gc}: invalid team identities")
        teams, players = [], []
        for side, own, opposing, name, opposing_name, loc in (
            ("local", local_id, road_id, local_name, road_name, "H"),
            ("road", road_id, local_id, road_name, local_name, "A"),
        ):
            p_side = payload[side]
            stat = p_side.get("total") or p_side.get("team") or {}
            if not isinstance(stat, dict):
                stat = {}
            totals = numeric_row(stat)
            entry = dict(SEASON=int(season), COMPETITION=competition,
                         GAME_ID=f"{competition}{season}_{gc}", GAME_CODE=gc, GAME_DATE=date_fmt,
                         TEAM_ID=own, TEAM_NAME=name, OPP_ID=opposing, OPP_NAME=opposing_name,
                         HOME_AWAY=loc, PHASE=scalar(game.get("phaseType")),
                         IS_NEUTRAL=bool(game.get("isNeutralVenue", False)), **totals)
            entry["MINUTES"] = None
            teams.append(entry)
            if any(totals.get(x) is None for x in ("FG2M", "FG2A", "FG3M", "FG3A", "FTM", "FTA", "OREB", "TO", "PTS")):
                errors.append(f"game={gc} {side}: missing required team stats")
            elif totals["FG2A"] < totals.get("FG2M", 0) or totals["FG3A"] < totals.get("FG3M", 0) or totals["FTA"] < totals.get("FTM", 0):
                errors.append(f"game={gc} {side}: attempts < makes")
            elif abs(2 * totals["FG2M"] + 3 * totals["FG3M"] + totals["FTM"] - totals["PTS"]) > 1.01:
                errors.append(f"game={gc} {side}: team score vs shooting mismatch")
            for p in p_side.get("players", []):
                person = p.get("player", {})
                player_id, player_name = player_identity(person)
                ps = p.get("stats") or {}
                if not ps:
                    continue
                pm = minutes(ps.get("timePlayed"))
                if pm is None or pm < 0:
                    errors.append(f"game={gc} player={player_id}: bad minutes")
                row = dict(entry)
                row.update(numeric_row(ps))
                row.update(PLAYER_ID=f"{competition}:{player_id}", PLAYER_NAME=player_name,
                           POSITION=scalar(person.get("position") if isinstance(person, dict) else None),
                           MINUTES=pm)
                players.append(row)
        if len(teams) == 2 and all(t["PTS"] is not None for t in teams):
            for side, idx in (("local", 0), ("road", 1)):
                score = game.get(side, {}).get("score")
                if isinstance(score, dict):
                    score = score.get("total", score.get("points"))
                if score is not None and abs(float(score) - teams[idx]["PTS"]) > 1.01:
                    errors.append(f"game={gc}: official final score != stats for {side}")
            for entry, opponent in ((teams[0], teams[1]), (teams[1], teams[0])):
                entry["OPP_PTS"] = opponent["PTS"]
                for k in NUMERIC:
                    entry["OPP_" + k] = opponent[k]
                if all(entry[k] is not None for k in ("FGA", "OREB", "TO", "FTA")):
                    entry["POSS_RAW"] = entry["FGA"] - entry["OREB"] + entry["TO"] + 0.44 * entry["FTA"]
                else:
                    entry["POSS_RAW"] = None
            # Only if official boxscore sum of player minutes is sane.
            for idx, side in enumerate(("local", "road")):
                ids = [p for p in players if p["TEAM_ID"] == teams[idx]["TEAM_ID"]]
                mins = sum(p["MINUTES"] for p in ids if p["MINUTES"] is not None)
                teams[idx]["TEAM_MINUTES"] = mins
                teams[idx]["OT_FLAG"] = mins > 203.0
                if mins and abs(mins - 200) > 8 and not (203 <= mins <= 280):
                    errors.append(f"game={gc} {side}: player minutes sum atypical ({mins:.1f})")
        return teams, players, errors

    def load_season(self, season: int, max_games: int | None = None, strict: bool = True, regular_only: bool = True) -> dict[str, pd.DataFrame]:
        games = sorted(self.games(season), key=lambda g: str(g.get("date", "")))
        if regular_only:
            # API phase values can be 'RS' or a dict. Empty means not proved: retain and audit.
            games = [g for g in games if scalar(g.get("phaseType", "RS")).upper() in {"RS", "REGULAR SEASON", "REGULAR_SEASON"}]
        games = [g for g in games if self._ready(g)]
        if max_games is not None:
            games = games[:int(max_games)]
        out_t, out_p, audit = [], [], []
        for game in games:
            gc = int(game["gameCode"])
            try:
                teams, players, errs = self.normalize_game(game, self.game_stats(season, gc), self.competition, season)
                audit.extend({"GAME_ID": f"{self.competition}{season}_{gc}", "ISSUE": e} for e in errs)
                if errs and strict:
                    continue
                out_t.extend(teams)
                out_p.extend(players)
            except (ValueError, TypeError, KeyError, requests.RequestException) as exc:
                audit.append({"GAME_ID": f"{self.competition}{season}_{gc}", "ISSUE": str(exc)})
        team_df = pd.DataFrame(out_t)
        player_df = pd.DataFrame(out_p)
        audit_df = pd.DataFrame(audit, columns=["GAME_ID", "ISSUE"])
        if not team_df.empty:
            # Exact mirrored 2-row games, uniqueness checks.
            duplicated = team_df.duplicated(["GAME_ID", "TEAM_ID"]).any()
            count_bad = (team_df.groupby("GAME_ID").size() != 2).any()
            if duplicated or count_bad:
                raise ValueError("Fatal: malformed or duplicated game/team rows")
        return {"team": team_df, "player": player_df, "audit": audit_df,
                "metadata": {"competition": self.competition, "season": season,
                             "loaded_games": len(team_df) // 2, "scheduled_played": len(games),
                             "regular_only": regular_only, "strict": strict}}
