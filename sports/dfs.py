"""What the DFS site knows about tonight, as game features.

NBA - team strength from the DFS player projections (DFS-ALLSPORTS):
  history: data/nba_team_proj.csv there (walk-forward projections 2017-2021
  from DraftKings' own player data, plus every board published this season);
  tonight: the live boards on the DFS site, so a late scratch moves the forecast.
  dfs_avail = projected DK points of the top 8 players expected to play;
  dfs_missing = the same for the whole roster minus that (who is out).

NHL - the starting goalie: the DFS board carries each goalie's start
probability (DailyFaceoff confirmations), and history has who actually started.
"""
from __future__ import annotations

import datetime as dt
import io
import logging

import numpy as np
import pandas as pd
import requests

from sports.fetch import CACHE, NHL_CANON, nba_team

log = logging.getLogger("sports.dfs")
RAW = "https://raw.githubusercontent.com/willyjo423/DFS-ALLSPORTS/main/"
PAGES = "https://willyjo423.github.io/DFS-ALLSPORTS/"


def _csv(url: str, cache_name: str) -> pd.DataFrame:
    path = CACHE / cache_name
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    except Exception as exc:                                   # noqa: BLE001
        log.warning("%s unavailable (%s)", url, str(exc)[:80])
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def live_boards(sport: str) -> list[dict]:
    """Tonight's boards from the DFS site (biggest first)."""
    try:
        man = requests.get(PAGES + "data/manifest.json", timeout=30).json()
    except Exception as exc:                                   # noqa: BLE001
        log.warning("DFS manifest unavailable (%s)", str(exc)[:80])
        return []
    out = []
    for s in man.get("slates") or []:
        if s.get("sport") != sport or s.get("kind") != "classic":
            continue
        try:
            out.append(requests.get(PAGES + s["file"], timeout=60).json())
        except Exception:                                      # noqa: BLE001
            continue
    return sorted(out, key=lambda b: -len(b.get("players") or []))


# --------------------------------------------------------------------- NBA
TOP = 8


def _team_from_board(ps: list[dict]) -> dict:
    proj = np.array([float(p.get("mean") or p.get("proj") or 0) for p in ps])
    out = np.array([str(p.get("status", "")).lower() in ("out", "o", "inactive") for p in ps])
    # an out player's projection is zeroed on the board; value him from salary
    full_val = np.where(out, np.array([float(p.get("salary") or 0) for p in ps]) / 1000 * 5.0, proj)
    avail = np.sort(proj[~out])[::-1][:TOP].sum()
    full = np.sort(full_val)[::-1][:TOP].sum()
    return {"dfs_avail": float(avail), "dfs_missing": float(max(full - avail, 0.0)),
            "dfs_out": [p["name"] for p, o in zip(ps, out) if o and float(p.get("salary") or 0) >= 5000]}


def nba_history() -> pd.DataFrame:
    df = _csv(RAW + "data/nba_team_proj.csv", "nba_team_proj.csv")
    if df.empty:
        return df
    df["team"] = df["team"].map(nba_team)
    df.loc[df["dfs_avail"] < 100, "dfs_avail"] = np.nan          # opening-day artefacts
    return df[["date", "team", "dfs_avail", "dfs_missing"]]


def nba_tonight() -> dict:
    """{team: {dfs_avail, dfs_missing, dfs_out}} from the live boards."""
    res = {}
    for b in live_boards("nba"):
        teams: dict = {}
        for p in b.get("players") or []:
            teams.setdefault(nba_team(p.get("team")), []).append(p)
        for t, ps in teams.items():
            if t not in res:
                res[t] = _team_from_board(ps)
    return res


# --------------------------------------------------------------------- NHL
SK_TOP = 18


def nhl_skater_history() -> pd.DataFrame:
    df = _csv(RAW + "data/nhl_team_proj.csv", "nhl_team_proj.csv")
    if df.empty:
        return df
    df["team"] = df["team"].map(lambda t: NHL_CANON.get(t, t))
    return df[["date", "team", "sk_avail"]]


def nhl_skaters_tonight() -> dict:
    """{team: {sk_avail, sk_out}} from the live boards: projected DK points of
    the top 18 skaters dressing."""
    res = {}
    for b in live_boards("nhl"):
        teams: dict = {}
        for p in b.get("players") or []:
            if p.get("pos") == "G":
                continue
            teams.setdefault(NHL_CANON.get(p.get("team"), p.get("team")), []).append(p)
        for t, ps in teams.items():
            if t in res:
                continue
            out = [str(p.get("status", "")).lower() in ("out", "o", "ir", "inactive") for p in ps]
            proj = np.array([float(p.get("mean") or p.get("proj") or p.get("med") or 0)
                             for p, o in zip(ps, out) if not o])
            res[t] = {"sk_avail": float(np.sort(proj)[::-1][:SK_TOP].sum()),
                      "sk_out": [p["name"] for p, o in zip(ps, out) if o and float(p.get("salary") or 0) >= 4500]}
    return res


def nhl_tonight() -> dict:
    """{team: [(goalie name, start probability, confirmed?)]} from the boards."""
    res: dict = {}
    for b in live_boards("nhl"):
        for p in b.get("players") or []:
            if p.get("pos") != "G":
                continue
            t = NHL_CANON.get(p.get("team"), p.get("team"))
            lst = res.setdefault(t, {})
            if p["name"] not in lst:
                lst[p["name"]] = (float(p.get("pp") or 0), str(p.get("gst") or ""))
    return {t: sorted([(n, pp, g) for n, (pp, g) in v.items()], key=lambda x: -x[1])
            for t, v in res.items()}


# ------------------------------------------------------------ book lines
NHL_NAMES = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL", "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM", "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL", "Montréal Canadiens": "MTL", "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD", "New York Islanders": "NYI", "New York Rangers": "NYR",
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA", "St Louis Blues": "STL", "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR", "Utah Mammoth": "UTA",
    "Utah Hockey Club": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG"}


def nhl_book(dates: list[str]) -> dict:
    """{(home, away): {ml_home, ml_away, line_total}} from the DFS site's odds files."""
    out = {}
    for d in dates:
        try:
            js = requests.get(PAGES + f"data/odds/nhl_{d}.json", timeout=30).json()
        except Exception:                                      # noqa: BLE001
            continue
        for g in js.get("games") or []:
            h, a = NHL_NAMES.get(g.get("home_team")), NHL_NAMES.get(g.get("away_team"))
            if not h or not a:
                continue
            rec = {}
            for bk in g.get("bookmakers") or []:
                for mk in bk.get("markets") or []:
                    if mk.get("key") == "h2h":
                        for o in mk.get("outcomes") or []:
                            t = NHL_NAMES.get(o.get("name"))
                            if t == h:
                                rec["ml_home"] = float(o["price"])
                            elif t == a:
                                rec["ml_away"] = float(o["price"])
                    if mk.get("key") == "totals":
                        pts = [o.get("point") for o in mk.get("outcomes") or [] if o.get("point") is not None]
                        if pts:
                            rec["line_total"] = float(pts[0])
                break
            if rec:
                out[(h, a)] = rec
    return out


def nba_book(dates: list[str]) -> dict:
    """{(home, away): {line_margin, line_total}} (home margin, + = home favoured)."""
    out = {}
    for d in dates:
        try:
            js = requests.get(PAGES + f"data/odds/nba_{d}.json", timeout=30).json()
        except Exception:                                      # noqa: BLE001
            continue
        for t, v in (js.get("lines") or {}).items():
            if not v.get("home"):
                continue
            h, a = nba_team(t), nba_team(v.get("opp"))
            out[(h, a)] = {"line_margin": None if v.get("spread") is None else -float(v["spread"]),
                           "line_total": v.get("total")}
    return out
