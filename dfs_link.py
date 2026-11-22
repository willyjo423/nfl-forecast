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


def attach(feat: pd.DataFrame, proj: pd.DataFrame | None = None) -> pd.DataFrame:
    proj = load() if proj is None else proj
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
