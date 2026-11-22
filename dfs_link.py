"""Player-level information from the NFL-DFS model, as game features.

NFL-DFS projects every skill player from his own production, walk-forward
(fit only on weeks before the game) and WITHOUT betting lines - so this does
not smuggle the market into a model that deliberately leaves it out. Summed to
the team, it says something the team ratings cannot: who is actually playing
and how good that group is, this week. A WR1 and RB1 out show up here before
any margin has been played that would reveal it - the same shape as the
quarterback layer, extended to the rest of the offence.

Source: NFL-DFS data/team_proj.csv (team_proj_export.py there).
Columns (all from that week's projections):
  home/away_dfs_core    projected DK points of QB1 + top-2 RB + top-3 WR + TE1
  dfs_core_diff/sum
  home/away_dfs_change  core minus the team's own average core over its last
                        6 games - the injury/absence signal
  dfs_change_diff

It is a feature GROUP like the others: measured on its own and on top of the
current model in build.py, and only added to the production feature set if it
earns its place.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger("dfs_link")
URL = "https://raw.githubusercontent.com/willyjo423/NFL-DFS/main/data/team_proj.csv"
CACHE = config.DATA / "cache" / "team_proj.csv"

DFS_COLUMNS = ["home_dfs_core", "away_dfs_core", "dfs_core_diff", "dfs_core_sum",
               "home_dfs_change", "away_dfs_change", "dfs_change_diff"]


def load(refresh: bool = True) -> pd.DataFrame:
    """History with the trailing-average change column."""
    df = None
    if refresh:
        try:
            import requests
            r = requests.get(URL, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE, index=False)
        except Exception as exc:                               # noqa: BLE001
            log.warning("team projections unavailable (%s)", str(exc)[:80])
    if df is None and CACHE.exists():
        df = pd.read_csv(CACHE)
    if df is None or df.empty:
        return pd.DataFrame(columns=["season", "week", "team", "dfs_core"])
    df = df.sort_values(["team", "season", "week"]).copy()
    # trailing average of the team's own core, previous 6 games (across
    # season boundaries), never including this week
    df["dfs_base"] = (df.groupby("team")["dfs_core"]
                      .transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean()))
    df["dfs_change"] = df["dfs_core"] - df["dfs_base"]
    return df[["season", "week", "team", "dfs_core", "dfs_change"]]


PAGES = "https://willyjo423.github.io/NFL-DFS/"
CORE = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
DK_TO_NFLVERSE = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV"}


def live_week() -> dict:
    """{team: core} for this week from the live NFL-DFS boards (the biggest
    classic board a team appears on)."""
    import requests
    try:
        man = requests.get(PAGES + "data/manifest.json", timeout=30).json()
    except Exception as exc:                                   # noqa: BLE001
        log.warning("NFL-DFS manifest unavailable (%s)", str(exc)[:80])
        return {}
    boards = []
    for sl in man.get("slates") or []:
        if sl.get("kind") != "classic":
            continue
        try:
            boards.append(requests.get(PAGES + sl["file"], timeout=60).json())
        except Exception:                                      # noqa: BLE001
            continue
    res: dict = {}
    for b in sorted(boards, key=lambda b: -len(b.get("players") or [])):
        teams: dict = {}
        for p in b.get("players") or []:
            if str(p.get("status", "")).lower() in ("out", "o", "ir", "inactive"):
                continue
            t = DK_TO_NFLVERSE.get(p.get("team"), p.get("team"))
            teams.setdefault(t, []).append(p)
        for t, ps in teams.items():
            if t in res:
                continue
            core = 0.0
            for pos, n in CORE.items():
                v = sorted((float(p.get("mean") or p.get("proj") or 0) for p in ps
                            if (p.get("pos") or p.get("position")) == pos), reverse=True)
                core += sum(v[:n])
            res[t] = core
    return res


def attach(feat: pd.DataFrame, proj: pd.DataFrame | None = None, live: dict | None = None) -> pd.DataFrame:
    proj = load() if proj is None else proj
    if live:
        # this week's games from the live boards, appended after the history so
        # the "change vs usual" compares against the team's own recent games
        add = []
        for r in feat.itertuples(index=False):
            for t in (r.home_team, r.away_team):
                if t in live:
                    add.append({"season": r.season, "week": r.week, "team": t, "dfs_core": live[t]})
        if add:
            base = proj[["season", "week", "team", "dfs_core"]] if len(proj) else pd.DataFrame(
                columns=["season", "week", "team", "dfs_core"])
            allp = pd.concat([base, pd.DataFrame(add)]).drop_duplicates(["season", "week", "team"], keep="last")
            allp = allp.sort_values(["team", "season", "week"])
            allp["dfs_base"] = allp.groupby("team")["dfs_core"].transform(
                lambda s: s.shift(1).rolling(6, min_periods=2).mean())
            allp["dfs_change"] = allp["dfs_core"] - allp["dfs_base"]
            proj = allp[["season", "week", "team", "dfs_core", "dfs_change"]]
    out = feat.copy()
    if proj.empty:
        for c in DFS_COLUMNS:
            out[c] = np.nan
        return out
    p = proj.set_index(["season", "week", "team"])
    for side in ("home", "away"):
        key = pd.MultiIndex.from_arrays([out["season"], out["week"], out[f"{side}_team"]])
        got = p.reindex(key)
        out[f"{side}_dfs_core"] = got["dfs_core"].to_numpy()
        out[f"{side}_dfs_change"] = got["dfs_change"].to_numpy()
    out["dfs_core_diff"] = out["home_dfs_core"] - out["away_dfs_core"]
    out["dfs_core_sum"] = out["home_dfs_core"] + out["away_dfs_core"]
    out["dfs_change_diff"] = out["home_dfs_change"] - out["away_dfs_change"]
    cov = out["home_dfs_core"].notna().mean()
    log.info("DFS team projections joined to %.0f%% of games", 100 * cov)
    return out
