"""How the model did against the betting line, game by game, this season.

Uses the walk-forward forecasts (each game predicted by a model fitted only on
earlier seasons, with that night's ratings and DFS layer) against the lines
the DFS site saved:
  NBA - spread and total (DFS-ALLSPORTS data/nba_games.csv.gz from ESPN, plus
        the daily odds files);
  NHL - moneyline and total (the daily odds files; saved since Nov 18).
For every game the model "leans" to one side of the line; this counts how
often that side won, split by how far the model was from the line.
"""
from __future__ import annotations

import datetime as dt
import io
import logging

import numpy as np
import pandas as pd
import requests

from sports.dfs import NHL_NAMES, PAGES, RAW
from sports.fetch import CACHE, nba_team

log = logging.getLogger("sports.vs_line")


def _odds_files(sport: str, start: dt.date, end: dt.date) -> dict:
    path = CACHE / f"{sport}_odds_files.json"
    import json
    have = json.loads(path.read_text()) if path.exists() else {}
    d = start
    while d <= end:
        k = d.isoformat()
        if k not in have or k >= (end - dt.timedelta(days=1)).isoformat():
            try:
                r = requests.get(PAGES + f"data/odds/{sport}_{k}.json", timeout=20)
                have[k] = r.json() if r.status_code == 200 else None
            except Exception:                                  # noqa: BLE001
                have[k] = have.get(k)
        d += dt.timedelta(days=1)
    path.write_text(json.dumps(have))
    return {k: v for k, v in have.items() if v}


def nba_lines(start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = []
    try:
        r = requests.get(RAW + "data/nba_games.csv.gz", timeout=120)
        r.raise_for_status()
        g = pd.read_csv(io.BytesIO(r.content), compression="gzip",
                        usecols=["game_id", "date", "team", "opp", "home", "spread", "total"])
        g = g[g["home"] == 1].drop_duplicates("game_id")
        for x in g.itertuples(index=False):
            if pd.notna(x.spread):
                rows.append({"date": x.date, "home": nba_team(x.team), "away": nba_team(x.opp),
                             "lm": -float(x.spread), "lt": x.total})
    except Exception as exc:                                   # noqa: BLE001
        log.warning("nba_games.csv.gz unavailable (%s)", str(exc)[:80])
    for day, js in _odds_files("nba", start, end).items():
        for t, v in (js.get("lines") or {}).items():
            if v.get("home") and v.get("spread") is not None:
                rows.append({"date": day, "home": nba_team(t), "away": nba_team(v.get("opp")),
                             "lm": -float(v["spread"]), "lt": v.get("total")})
    return pd.DataFrame(rows).drop_duplicates(["date", "home"], keep="first")


def nhl_lines(start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = []
    for day, js in _odds_files("nhl", start, end).items():
        for g in js.get("games") or []:
            h, a = NHL_NAMES.get(g.get("home_team")), NHL_NAMES.get(g.get("away_team"))
            if not h or not a:
                continue
            rec = {"home": h, "away": a,
                   "date": (pd.Timestamp(g.get("commence_time")).tz_convert("America/New_York").date().isoformat()
                            if g.get("commence_time") else day)}
            for bk in (g.get("bookmakers") or [])[:1]:
                for mk in bk.get("markets") or []:
                    if mk.get("key") == "h2h":
                        for o in mk.get("outcomes") or []:
                            t = NHL_NAMES.get(o.get("name"))
                            rec["mlh" if t == h else "mla"] = float(o["price"])
                    if mk.get("key") == "totals":
                        pts = [o.get("point") for o in mk.get("outcomes") or [] if o.get("point") is not None]
                        if pts:
                            rec["lt"] = float(pts[0])
            rows.append(rec)
    return pd.DataFrame(rows).drop_duplicates(["date", "home"], keep="last") if rows else pd.DataFrame()


def _implied(o):
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def _bucket_rows(gap: np.ndarray, won: np.ndarray, push: np.ndarray, edges, unit) -> list[dict]:
    out = []
    for lo, hi in edges:
        m = (np.abs(gap) >= lo) & (np.abs(gap) < hi) & ~push
        n = int(m.sum())
        if n == 0:
            continue
        w = int(won[m].sum())
        out.append({"gap": f"{lo:g}–{hi:g} {unit}" if hi < 99 else f"{lo:g}+ {unit}",
                    "lo": lo, "hi": hi, "n": n, "won": w, "lost": n - w, "pct": round(w / n, 3)})
    m = ~push
    n = int(m.sum())
    if n:
        w = int(won[m].sum())
        out.append({"gap": "all games", "n": n, "won": w, "lost": n - w, "pct": round(w / n, 3)})
    return out


def report(sport: str, oos: pd.DataFrame, season: int, today: dt.date) -> dict:
    o = oos[oos["season"] == season].copy()
    if o.empty:
        return {}
    start = dt.date(season, 9, 25)
    res = {"season": season, "games_forecast": int(len(o))}
    if sport == "nba":
        L = nba_lines(start, today)
        if L.empty:
            return res
        j = o.merge(L, on=["date", "home", "away"], how="inner")
        res["games_with_line"] = int(len(j))
        if len(j):
            gap = (j["pred_margin"] - j["lm"]).to_numpy()
            res_m = (j["margin"] - j["lm"]).to_numpy()
            push = np.isclose(res_m, 0) | np.isclose(gap, 0)
            won = np.sign(res_m) == np.sign(gap)
            res["spread"] = _bucket_rows(gap, won, push, [(0, 2), (2, 4), (4, 6), (6, 99)], "pts")
            jt = j.dropna(subset=["lt"])
            g2 = (jt["pred_total"] - jt["lt"]).to_numpy()
            r2 = (jt["total"] - jt["lt"]).to_numpy()
            p2 = np.isclose(r2, 0) | np.isclose(g2, 0)
            res["total"] = _bucket_rows(g2, np.sign(r2) == np.sign(g2), p2,
                                        [(0, 3), (3, 6), (6, 10), (10, 99)], "pts")
            res["avg_abs_gap_spread"] = round(float(np.abs(gap).mean()), 2)
    else:
        L = nhl_lines(start, today)
        if L.empty:
            return res
        j = o.merge(L, on=["date", "home", "away"], how="inner")
        res["games_with_line"] = int(len(j))
        jm = j.dropna(subset=["mlh", "mla"])
        if len(jm):
            ih, ia = _implied(jm["mlh"].to_numpy()), _implied(jm["mla"].to_numpy())
            mh = ih / (ih + ia)
            edge = jm["home_win_prob"].to_numpy() - mh
            home_won = (jm["margin"] > 0).to_numpy()
            won = np.where(edge > 0, home_won, ~home_won)
            # units: 1 unit on the leaned side at the book's price
            price = np.where(edge > 0, jm["mlh"], jm["mla"])
            payout = np.where(price > 0, price / 100, 100 / -price)
            units = np.where(won, payout, -1.0)
            res["moneyline"] = _bucket_rows(edge * 100, won, np.zeros(len(won), bool),
                                            [(0, 3), (3, 6), (6, 10), (10, 99)], "% win-prob")
            res["moneyline_units"] = round(float(units.sum()), 2)
        jt = j.dropna(subset=["lt"])
        if len(jt):
            g2 = (jt["pred_total"] - jt["lt"]).to_numpy()
            r2 = (jt["total"] - jt["lt"]).to_numpy()
            p2 = np.isclose(r2, 0) | np.isclose(g2, 0)
            res["total"] = _bucket_rows(g2, np.sign(r2) == np.sign(g2), p2,
                                        [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 99)], "goals")
    return res


BREAKEVEN = 0.524
MIN_GAMES = 50


def flag_table(*reports: dict) -> dict:
    """Combine seasons: per market and gap range, which side has hit and how
    often. A range is flagged when one side beat break-even (52.4%) over at
    least 50 games - the model's side, or the market's side when the model's
    lean lost often enough."""
    out = {}
    for key in ("spread", "total", "moneyline"):
        acc: dict = {}
        for r in reports:
            for row in (r or {}).get(key) or []:
                if "lo" not in row:
                    continue
                a = acc.setdefault((row["lo"], row["hi"]), {"won": 0, "lost": 0, "gap": row["gap"]})
                a["won"] += row["won"]
                a["lost"] += row["lost"]
        rows = []
        for (lo, hi), a in sorted(acc.items()):
            n = a["won"] + a["lost"]
            p = a["won"] / n if n else 0.5
            side = "model" if p >= BREAKEVEN else "market" if (1 - p) >= BREAKEVEN else None
            rows.append({"lo": lo, "hi": hi, "gap": a["gap"], "n": n, "model_pct": round(p, 3),
                         "flag": side if n >= MIN_GAMES else None,
                         "hit": round(p if side == "model" else 1 - p, 3) if side else None})
        if rows:
            out[key] = rows
    return out


def flag_for(table: dict, key: str, gap: float) -> dict | None:
    for r in table.get(key) or []:
        if r["lo"] <= abs(gap) < r["hi"]:
            return r if r.get("flag") else None
    return None
