"""Build, test and publish the NBA or NHL forecasts.

    python -m sports.run --sport nba
    python -m sports.run --sport nhl

Fetches (cached) results, rates teams, adds the DFS layer (NBA: team
strength from the DFS player projections; NHL: the starting goalie from the
DFS confirmations), fits the model with it, forecasts today's and tomorrow's
games, and writes docs/<sport>.json, docs/<sport>.html
and data/<sport>_metrics.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sports import dfs as DFS
from sports import features as FE
from sports import fetch as F
from sports import model as M
from sports import render as R

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")
log = logging.getLogger("sports.run")
FIRST = {"nba": 2016, "nhl": 2016}


def load_games(sport: str, today: dt.date) -> pd.DataFrame:
    frames = []
    for s in range(FIRST[sport], F.season_of(today) + 1):
        df = F.season_games(sport, s, today)
        if len(df):
            df["season"] = s
            frames.append(df)
    g = pd.concat(frames, ignore_index=True)
    g["done"] = g["done"].astype(str).str.lower().isin(["true", "1"])
    g["neutral"] = g["neutral"].astype(str).str.lower().isin(["true", "1"])
    g["game_id"] = g["game_id"].astype(str)
    return g.drop_duplicates("game_id", keep="last")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=("nba", "nhl"), required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sport = a.sport
    now = dt.datetime.now(ET)
    today = now.date()

    games = load_games(sport, today)
    print(f"{sport}: {len(games):,} games, {games['done'].sum():,} finished", flush=True)
    g = M.team_ratings(games, sport)
    g = M.add_rest(g)

    if sport == "nba":
        tonight = DFS.nba_tonight()
        g = FE.nba_dfs(g, DFS.nba_history(), tonight)
        extra_m, extra_t, flag = FE.NBA_DFS_M, FE.NBA_DFS_T, None
        label = "DFS team strength (who is playing, from the DFS player projections)"
    else:
        tonight = DFS.nhl_tonight()
        logs = pd.concat([F.season_goalies(s, today) for s in range(FIRST[sport], F.season_of(today) + 1)],
                         ignore_index=True)
        g = FE.nhl_goalies(g, logs, tonight)
        sk_tonight = DFS.nhl_skaters_tonight()
        g = FE.nhl_skaters(g, DFS.nhl_skater_history(), sk_tonight)
        tonight = {"goalies": tonight, "skaters": sk_tonight}
        extra_m, extra_t, flag = FE.NHL_G_M, FE.NHL_G_T, None
        label = "DFS skater strength + starting goalie (DFS confirmations + his record)"

    # ---- the DFS layer is always part of the model
    chosen = M.walk_forward(g, extra_m, extra_t)
    metrics = M.evaluate(chosen)
    metrics["dfs_layer"] = label
    print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=1))

    # ---- final model and the slate
    fit = M.fit_final(g, extra_m, extra_t)
    metrics["coefficients"] = {"margin": {k: float(v) for k, v in fit["coef_m"].items()},
                               "total": {k: float(v) for k, v in fit["coef_t"].items()}}
    up = g[~g["done"] & (g["date"] >= today.isoformat())
           & (g["date"] <= (today + dt.timedelta(days=1)).isoformat()) & g["rate_margin"].notna()]
    pred = M.predict(fit, up) if len(up) else up
    payload = R.payload(sport, pred, metrics, tonight, now)
    (ROOT / "docs" / f"{sport}.json").write_text(json.dumps(payload, indent=1, default=str))
    (ROOT / "docs" / f"{sport}.html").write_text(R.page(sport, payload))
    (ROOT / "data" / f"{sport}_metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    print(f"wrote docs/{sport}.html with {len(payload['games'])} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
