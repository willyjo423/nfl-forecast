"""Game results for the NBA and NHL tabs.

NBA: ESPN's public scoreboard, one call per date - scores, and the spread and
total ESPN shows when it has them. NHL: the NHL's own API - scores per date,
and every goalie's game log (who started, shots, goals against) from the
stats endpoint, a few dozen calls a season.

Finished seasons are cached in data/cache/ and never refetched.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("sports.fetch")
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
UA = {"User-Agent": "Mozilla/5.0 (forecast research)"}

NBA_CANON = {"GS": "GSW", "SA": "SAS", "NO": "NOP", "NOR": "NOP", "NY": "NYK", "UTAH": "UTA",
             "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "NJN": "BKN", "CHO": "CHA", "CHH": "CHA"}
NHL_CANON = {"ARI": "UTA", "PHX": "UTA"}      # the franchise moved; ratings carry over


def nba_team(c: str) -> str:
    c = str(c or "").upper().strip()
    return NBA_CANON.get(c, c)


def _get(url, params=None, tries=4):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as exc:                               # noqa: BLE001
            last = str(exc)[:80]
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url} {params}: {last}")


def season_of(day: dt.date) -> int:
    """Start year of the season a date belongs to (Oct-Jun)."""
    return day.year if day.month >= 8 else day.year - 1


def _days(season: int, until: dt.date | None = None):
    d, end = dt.date(season, 9, 25), dt.date(season + 1, 7, 5)
    if until:
        end = min(end, until)
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


# --------------------------------------------------------------------- NBA
NBA_SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


def _nba_odds(comp: dict, home: str, away: str):
    o = (comp.get("odds") or [{}])[0] if comp.get("odds") else {}
    total = o.get("overUnder")
    spread = None
    det = str(o.get("details") or "")
    m = re.match(r"([A-Z]+)\s+([-+]?\d+(?:\.\d+)?)", det)
    if m:
        team, pts = nba_team(m.group(1)), float(m.group(2))
        # "BOS -5.5": BOS favoured by 5.5 -> home margin line
        spread = -pts if team == home else pts if team == away else None
    elif det.upper() == "EVEN":
        spread = 0.0
    return spread, (float(total) if total not in (None, "") else None)


def nba_day(day: dt.date) -> list[dict]:
    sb = _get(NBA_SB, {"dates": day.strftime("%Y%m%d"), "limit": 50})
    rows = []
    for ev in sb.get("events") or []:
        stype = (ev.get("season") or {}).get("type")
        if stype not in (2, 3, 5):                       # regular, playoffs, play-in
            continue
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if "home" not in teams or "away" not in teams:
            continue
        h, a = teams["home"], teams["away"]
        home, away = nba_team(h["team"]["abbreviation"]), nba_team(a["team"]["abbreviation"])
        done = bool(((ev.get("status") or {}).get("type") or {}).get("completed"))
        spread, total = _nba_odds(comp, home, away)
        rows.append({"game_id": str(ev["id"]), "date": day.isoformat(), "start_utc": ev.get("date"),
                     "home": home, "away": away, "neutral": bool(comp.get("neutralSite")),
                     "home_pts": float(h.get("score")) if done else None,
                     "away_pts": float(a.get("score")) if done else None,
                     "done": done, "playoff": stype == 3,
                     "line_margin": spread, "line_total": total})
    return rows


# --------------------------------------------------------------------- NHL
NHL_SCORE = "https://api-web.nhle.com/v1/score/{d}"
NHL_GOALIE = "https://api.nhle.com/stats/rest/en/goalie/summary"


def nhl_day(day: dt.date) -> list[dict]:
    sc = _get(NHL_SCORE.format(d=day.isoformat()))
    rows = []
    for g in sc.get("games") or []:
        if g.get("gameType") not in (2, 3):
            continue
        st = g.get("gameState")
        done = st in ("OFF", "FINAL")
        hs, as_ = (g.get("homeTeam") or {}), (g.get("awayTeam") or {})
        canon = lambda x: NHL_CANON.get(x, x)                  # noqa: E731
        per = ((g.get("periodDescriptor") or {}).get("periodType")) if done else None
        rows.append({"game_id": str(g["id"]), "date": day.isoformat(), "start_utc": g.get("startTimeUTC"),
                     "home": canon(hs.get("abbrev")), "away": canon(as_.get("abbrev")),
                     "neutral": bool(g.get("neutralSite")),
                     "home_pts": float(hs.get("score")) if done and hs.get("score") is not None else None,
                     "away_pts": float(as_.get("score")) if done and as_.get("score") is not None else None,
                     "done": done, "playoff": g.get("gameType") == 3, "ended_in": per,
                     "line_margin": None, "line_total": None})
    return rows


def nhl_goalies(season: int) -> pd.DataFrame:
    """Every goalie appearance of a season: game, team, starter, shots, goals."""
    sid = f"{season}{season + 1}"
    out, start = [], 0
    while True:
        js = _get(NHL_GOALIE, {"isAggregate": "false", "isGame": "true", "start": start,
                               "limit": 100, "sort": "gameDate",
                               "cayenneExp": f"seasonId={sid} and gameTypeId>=2"})
        data = js.get("data") or []
        for r in data:
            out.append({"game_id": str(r.get("gameId")), "date": str(r.get("gameDate"))[:10],
                        "team": NHL_CANON.get(r.get("teamAbbrev"), r.get("teamAbbrev")),
                        "goalie_id": r.get("playerId"), "goalie": r.get("goalieFullName"),
                        "started": int(r.get("gamesStarted") or 0),
                        "shots": float(r.get("shotsAgainst") or 0),
                        "goals": float(r.get("goalsAgainst") or 0)})
        start += len(data)
        if not data or start >= int(js.get("total") or 0):
            break
        time.sleep(0.3)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ cache
def season_games(sport: str, season: int, today: dt.date) -> pd.DataFrame:
    """A season's games (finished seasons cached; the current one refetched
    from the cache's last finished day)."""
    path = CACHE / f"{sport}_games_{season}.csv"
    finished = dt.date(season + 1, 7, 5) < today
    have = pd.read_csv(path, dtype={"game_id": str}) if path.exists() else pd.DataFrame()
    if finished and len(have):
        return have
    fn = nba_day if sport == "nba" else nhl_day
    keep = have[have["done"] == True] if len(have) else have          # noqa: E712
    last_done = (dt.date.fromisoformat(keep["date"].max()) - dt.timedelta(days=2)) if len(keep) else None
    rows = keep[keep["date"] < (last_done.isoformat() if last_done else "0")].to_dict("records") if len(keep) else []
    for day in _days(season, until=today + dt.timedelta(days=2)):
        if last_done and day < last_done:
            continue
        try:
            rows += fn(day)
        except Exception as exc:                               # noqa: BLE001
            log.warning("%s %s failed: %s", sport, day, exc)
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates("game_id", keep="last").sort_values(["date", "game_id"])
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    log.info("%s %s: %d games", sport, season, len(df))
    return df


def season_goalies(season: int, today: dt.date) -> pd.DataFrame:
    path = CACHE / f"nhl_goalies_{season}.csv"
    finished = dt.date(season + 1, 7, 5) < today
    if finished and path.exists():
        return pd.read_csv(path, dtype={"game_id": str})
    df = nhl_goalies(season)
    if len(df):
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df
