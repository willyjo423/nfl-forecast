"""The DFS layer as game features (see sports/dfs.py for the sources)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NBA_DFS_M = ["dfs_avail_diff", "dfs_change_diff", "dfs_missing_diff"]
NBA_DFS_T = ["dfs_avail_sum", "dfs_change_sum"]
NHL_G_M = ["goalie_q_diff", "home_backup", "away_backup", "sk_avail_diff", "sk_change_diff"]
NHL_G_T = ["goalie_q_sum", "sk_avail_sum", "sk_change_sum"]


# --------------------------------------------------------------------- NBA
def nba_dfs(g: pd.DataFrame, hist: pd.DataFrame, tonight: dict | None = None) -> pd.DataFrame:
    """hist: date, team, dfs_avail, dfs_missing. tonight: {team: {...}} for
    games not in hist yet (today's live boards)."""
    h = hist.copy() if len(hist) else pd.DataFrame(columns=["date", "team", "dfs_avail", "dfs_missing"])
    if tonight:
        add = []
        for gid, d, home, away in g.loc[~g["done"].astype(bool), ["game_id", "date", "home", "away"]].itertuples(index=False):
            for t in (home, away):
                if t in tonight:
                    add.append({"date": d, "team": t, "dfs_avail": tonight[t]["dfs_avail"],
                                "dfs_missing": tonight[t]["dfs_missing"]})
        if add:
            h = pd.concat([h, pd.DataFrame(add)]).drop_duplicates(["date", "team"], keep="last")
    h = h.sort_values(["team", "date"])
    # this team's usual strength: its own last 10 games with data
    h["dfs_base"] = h.groupby("team")["dfs_avail"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    h["dfs_change"] = h["dfs_avail"] - h["dfs_base"]
    k = h.set_index(["date", "team"])
    out = g.copy()
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([out["date"], out[side]])
        got = k.reindex(idx)
        for c in ("dfs_avail", "dfs_missing", "dfs_change"):
            out[f"{side}_{c}"] = got[c].to_numpy()
    out["dfs_avail_diff"] = out["home_dfs_avail"] - out["away_dfs_avail"]
    out["dfs_avail_sum"] = out["home_dfs_avail"] + out["away_dfs_avail"]
    out["dfs_change_diff"] = out["home_dfs_change"] - out["away_dfs_change"]
    out["dfs_change_sum"] = out["home_dfs_change"] + out["away_dfs_change"]
    out["dfs_missing_diff"] = out["away_dfs_missing"] - out["home_dfs_missing"]
    out["has_dfs"] = out["home_dfs_avail"].notna() & out["away_dfs_avail"].notna()
    return out


# --------------------------------------------------------------------- NHL
def nhl_skaters(g: pd.DataFrame, hist: pd.DataFrame, tonight: dict | None = None) -> pd.DataFrame:
    """Skater strength per side: projected DK points of the dressed top 18,
    and its change vs the team's own last 10 games (scratches, call-ups)."""
    h = hist.copy() if len(hist) else pd.DataFrame(columns=["date", "team", "sk_avail"])
    if tonight:
        add = []
        for d, home, away in g.loc[~g["done"].astype(bool), ["date", "home", "away"]].itertuples(index=False):
            for t in (home, away):
                if t in tonight:
                    add.append({"date": d, "team": t, "sk_avail": tonight[t]["sk_avail"]})
        if add:
            h = pd.concat([h, pd.DataFrame(add)]).drop_duplicates(["date", "team"], keep="last")
    h = h.sort_values(["team", "date"])
    h["sk_base"] = h.groupby("team")["sk_avail"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    h["sk_change"] = h["sk_avail"] - h["sk_base"]
    k = h.set_index(["date", "team"])
    out = g.copy()
    for side in ("home", "away"):
        got = k.reindex(pd.MultiIndex.from_arrays([out["date"], out[side]]))
        out[f"{side}_sk_avail"] = got["sk_avail"].to_numpy()
        out[f"{side}_sk_change"] = got["sk_change"].to_numpy()
    out["sk_avail_diff"] = out["home_sk_avail"] - out["away_sk_avail"]
    out["sk_avail_sum"] = out["home_sk_avail"] + out["away_sk_avail"]
    out["sk_change_diff"] = out["home_sk_change"] - out["away_sk_change"]
    out["sk_change_sum"] = out["home_sk_change"] + out["away_sk_change"]
    return out


PRIOR_SHOTS = 1500.0
HALF_SHOTS = 2500.0   # decay by shots faced, not calendar


def goalie_table(logs: pd.DataFrame) -> pd.DataFrame:
    """Per goalie appearance: his quality BEFORE that game (goals saved above
    an average goalie per 28 shots, shrunk toward average)."""
    L = logs.sort_values(["date", "game_id"]).copy()
    L["season"] = pd.to_datetime(L["date"]).map(lambda d: d.year if d.month >= 8 else d.year - 1)
    lg = L.groupby("season").apply(lambda d: 1 - d["goals"].sum() / max(d["shots"].sum(), 1))
    L["lg_sv"] = L["season"].map(lg)
    L["gsaa"] = (L["shots"] - L["goals"]) - L["shots"] * L["lg_sv"]
    q = np.full(len(L), 0.0)
    state: dict = {}
    for i, (gid, s, ga) in enumerate(zip(L["goalie_id"], L["shots"], L["gsaa"])):
        num, den = state.get(gid, (0.0, 0.0))
        q[i] = 28.0 * num / (den + PRIOR_SHOTS)
        keep = np.exp(-np.log(2) * s / HALF_SHOTS)
        state[gid] = (num * keep + ga, den * keep + s)
    L["q_before"] = q
    L["q_after_state"] = None
    return L, state


def current_quality(state: dict) -> dict:
    return {gid: 28.0 * num / (den + PRIOR_SHOTS) for gid, (num, den) in state.items()}


def nhl_goalies(g: pd.DataFrame, logs: pd.DataFrame, tonight: dict | None = None) -> pd.DataFrame:
    """Starter quality per side. Played games: who actually started. Upcoming:
    the DFS board's start probabilities (confirmations), else the team's
    regular starter."""
    out = g.copy()
    if logs.empty:
        for c in NHL_G_M + NHL_G_T:
            out[c] = np.nan
        return out
    L, state = goalie_table(logs)
    st = L[L["started"] == 1].drop_duplicates(["game_id", "team"])
    key = st.set_index(["game_id", "team"])
    names = L.drop_duplicates("goalie_id").set_index("goalie_id")["goalie"].to_dict()
    by_name = {str(v).lower(): k for k, v in names.items()}
    qnow = current_quality(state)
    # each team's regular starter: most starts in its last 10 games
    st_sorted = st.sort_values("date")
    regular = {t: d.tail(10)["goalie_id"].mode().iloc[0] for t, d in st_sorted.groupby("team")}
    # regular starter as of each game (for the backup flag on history)
    st_sorted = st_sorted.assign(reg=st_sorted.groupby("team")["goalie_id"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).apply(lambda x: pd.Series(x).mode().iloc[0], raw=False)))
    regkey = st_sorted.set_index(["game_id", "team"])["reg"]

    def side(row, team):
        gid = row["game_id"]
        if (gid, team) in key.index:
            r = key.loc[(gid, team)]
            reg = regkey.get((gid, team))
            backup = float(pd.notna(reg) and reg != r["goalie_id"])
            return float(r["q_before"]), backup, str(r["goalie"]), 1.0, "started"
        if tonight and team in tonight and tonight[team]:
            lst = tonight[team]
            tot = sum(pp for _, pp, _ in lst) or 1.0
            q, nm, top = 0.0, lst[0][0], lst[0]
            for n, pp, _ in lst:
                q += pp / tot * qnow.get(by_name.get(n.lower()), 0.0)
            reg = regular.get(team)
            backup = float(by_name.get(nm.lower()) not in (None, reg)) * top[1]
            return q, backup, nm, top[1], top[2]
        reg = regular.get(team)
        return qnow.get(reg, 0.0), 0.0, names.get(reg, ""), np.nan, "regular starter"

    cols = {k: [] for k in ("home_gq", "away_gq", "home_backup", "away_backup",
                            "home_goalie", "away_goalie", "home_gpp", "away_gpp", "home_gst", "away_gst")}
    for _, row in out.iterrows():
        for s in ("home", "away"):
            q, b, nm, pp, gs = side(row, row[s])
            cols[f"{s}_gq"].append(q)
            cols[f"{s}_backup"].append(b)
            cols[f"{s}_goalie"].append(nm)
            cols[f"{s}_gpp"].append(pp)
            cols[f"{s}_gst"].append(gs)
    for k, v in cols.items():
        out[k] = v
    out["goalie_q_diff"] = out["home_gq"] - out["away_gq"]
    out["goalie_q_sum"] = out["home_gq"] + out["away_gq"]
    return out
