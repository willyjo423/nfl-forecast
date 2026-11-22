"""Game forecasts for the NBA and NHL: margin, total and win probability.

Two stages, both walk-forward (nothing is ever fitted on a game it predicts):

1. Team ratings. For every game date, offence and defence ratings solved by
   ridge least squares on all earlier games, exponentially decayed (recent
   form counts more) and carried across seasons at reduced weight:
       points = mu + home edge + offence(team) - defence(opponent)
   That gives each side's expected points before anything else is known.
2. A small linear model on top, refit each season on earlier seasons only:
   the rating margin and total, rest (back-to-backs), and the DFS layer -
   NBA: tonight's available team strength from the DFS player projections;
   NHL: the starting goalie's quality. The DFS layer is a group like the NFL
   model's: it is measured game by game against the same model without it, and
   it is only used if it makes the forecasts more accurate (t <= -2, and in both
   halves of the games).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge

log = logging.getLogger("sports.model")

SETTINGS = {
    "nba": {"halflife": 45.0, "carry": 0.45, "ridge": 8.0, "window": 500},
    "nhl": {"halflife": 60.0, "carry": 0.45, "ridge": 25.0, "window": 500},
}


# ------------------------------------------------------------------ ratings
def _solve(hist: pd.DataFrame, day, teams_ix, nT, cfg, sport):
    age = (day - hist["d"]).dt.days.to_numpy(float)
    w = np.exp(-np.log(2) * age / cfg["halflife"])
    w = np.where(hist["season"].to_numpy() < hist["season_now"].iloc[0], w * cfg["carry"], w)
    n = len(hist)
    # two rows per game: home scoring and away scoring
    rows = 2 * n
    X = np.zeros((rows, 2 + 2 * nT))
    y = np.empty(rows)
    W = np.empty(rows)
    hi = hist["home"].map(teams_ix).to_numpy()
    ai = hist["away"].map(teams_ix).to_numpy()
    neutral = hist["neutral"].astype(bool).to_numpy()
    r = np.arange(n)
    X[r, 0] = 1
    X[r, 1] = np.where(neutral, 0, 1)
    X[r, 2 + hi] = 1
    X[r, 2 + nT + ai] = -1
    y[:n] = hist["home_pts"].to_numpy(float)
    X[n + r, 0] = 1
    X[n + r, 1] = np.where(neutral, 0, -0.0)
    X[n + r, 2 + ai] = 1
    X[n + r, 2 + nT + hi] = -1
    y[n:] = hist["away_pts"].to_numpy(float)
    W[:n] = w
    W[n:] = w
    # the home edge is applied symmetrically: +h/2 to home, -h/2 to away
    X[:n, 1] = np.where(neutral, 0, 0.5)
    X[n:, 1] = np.where(neutral, 0, -0.5)
    lam = np.r_[0, 0, np.full(2 * nT, cfg["ridge"])]
    sw = np.sqrt(W)
    A = X * sw[:, None]
    b = y * sw
    beta = np.linalg.solve(A.T @ A + np.diag(lam), A.T @ b)
    return beta


def team_ratings(games: pd.DataFrame, sport: str) -> pd.DataFrame:
    cfg = SETTINGS[sport]
    g = games.sort_values(["date", "game_id"]).reset_index(drop=True).copy()
    g["d"] = pd.to_datetime(g["date"])
    teams = sorted(set(g["home"]) | set(g["away"]))
    tix = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    done_mask = g["done"].astype(bool) & g["home_pts"].notna() & g["away_pts"].notna()
    rh = np.full(len(g), np.nan)
    ra = np.full(len(g), np.nan)
    known = np.zeros(len(g))
    for day, idx in g.groupby("d").groups.items():
        lo = day - pd.Timedelta(days=cfg["window"])
        hist = g[done_mask & (g["d"] < day) & (g["d"] >= lo)]
        if len(hist) < 60:
            continue
        hist = hist.assign(season_now=g.loc[idx[0], "season"])
        beta = _solve(hist, day, tix, nT, cfg, sport)
        mu, hca = beta[0], beta[1]
        off, de = beta[2:2 + nT], beta[2 + nT:]
        cur = hist[hist["season"] == g.at[idx[0], "season"]]
        played = pd.concat([cur["home"], cur["away"]]).value_counts()
        for i in idx:
            h, a = tix[g.at[i, "home"]], tix[g.at[i, "away"]]
            edge = 0.0 if bool(g.at[i, "neutral"]) else hca / 2
            rh[i] = mu + edge + off[h] - de[a]
            ra[i] = mu - edge + off[a] - de[h]
            # games each side has played this season (early-season flag)
            known[i] = min(played.get(g.at[i, "home"], 0), played.get(g.at[i, "away"], 0))
    g["rate_home"] = rh
    g["rate_away"] = ra
    g["rate_margin"] = rh - ra
    g["rate_total"] = rh + ra
    g["rate_games"] = known
    return g


# --------------------------------------------------------------------- rest
def add_rest(g: pd.DataFrame) -> pd.DataFrame:
    long = pd.concat([g[["game_id", "d", "home"]].rename(columns={"home": "team"}),
                      g[["game_id", "d", "away"]].rename(columns={"away": "team"})])
    long = long.sort_values(["team", "d"])
    long["rest"] = long.groupby("team")["d"].diff().dt.days.clip(upper=5).fillna(5)
    r = long.set_index(["game_id", "team"])["rest"]
    g["home_rest"] = [r.get((gid, t), 5) for gid, t in zip(g["game_id"], g["home"])]
    g["away_rest"] = [r.get((gid, t), 5) for gid, t in zip(g["game_id"], g["away"])]
    g["home_b2b"] = (g["home_rest"] <= 1).astype(float)
    g["away_b2b"] = (g["away_rest"] <= 1).astype(float)
    g["rest_diff"] = g["home_rest"].clip(upper=3) - g["away_rest"].clip(upper=3)
    return g


# --------------------------------------------------------------- stage two
BASE = ["rate_margin", "home_b2b", "away_b2b", "rest_diff"]
BASE_T = ["rate_total", "home_b2b", "away_b2b"]


def _X(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if X[c].isna().any():
            X[c + "_na"] = X[c].isna().astype(float)
            X[c] = X[c].fillna(0.0)
    return X


def walk_forward(df: pd.DataFrame, extra_m: list[str], extra_t: list[str],
                 min_seasons: int = 2) -> pd.DataFrame:
    """Out-of-sample predictions, one season at a time."""
    d = df[df["done"].astype(bool) & df["rate_margin"].notna() & (df["rate_games"] >= 5)].copy()
    d["margin"] = d["home_pts"] - d["away_pts"]
    d["total"] = d["home_pts"] + d["away_pts"]
    seasons = sorted(d["season"].unique())
    out = []
    for i, s in enumerate(seasons):
        if i < min_seasons:
            continue
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < 500 or te.empty:
            continue
        cm, ct = BASE + extra_m, BASE_T + extra_t
        Xtr, Xte = _X(tr, cm), _X(te, cm).reindex(columns=_X(tr, cm).columns, fill_value=0.0)
        mm = Ridge(alpha=1.0).fit(Xtr, tr["margin"])
        Ttr, Tte = _X(tr, ct), _X(te, ct).reindex(columns=_X(tr, ct).columns, fill_value=0.0)
        mt = Ridge(alpha=1.0).fit(Ttr, tr["total"])
        p = te[["game_id", "date", "season", "margin", "total", "line_margin", "line_total"]].copy()
        p["pred_margin"] = mm.predict(Xte)
        p["pred_total"] = mt.predict(Tte)
        ptr = mm.predict(Xtr)
        win = (tr["margin"] > 0).astype(int)
        lr = LogisticRegression(C=1.0).fit(ptr.reshape(-1, 1), win)
        p["home_win_prob"] = lr.predict_proba(p[["pred_margin"]].to_numpy())[:, 1]
        out.append(p)
    return pd.concat(out) if out else pd.DataFrame()


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) < 50:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d)


def evaluate(oos: pd.DataFrame) -> dict:
    if oos.empty:
        return {}
    m = {"n_games": int(len(oos)),
         "margin_mae": float((oos["margin"] - oos["pred_margin"]).abs().mean()),
         "total_mae": float((oos["total"] - oos["pred_total"]).abs().mean()),
         "win_accuracy": float(((oos["home_win_prob"] > 0.5) == (oos["margin"] > 0)).mean()),
         "win_brier": float(((oos["home_win_prob"] - (oos["margin"] > 0)) ** 2).mean()),
         "seasons": [int(s) for s in sorted(oos["season"].unique())]}
    lined = oos.dropna(subset=["line_margin"])
    if len(lined) >= 200:
        m["market_n"] = int(len(lined))
        m["market_margin_mae"] = float((lined["margin"] - lined["line_margin"]).abs().mean())
        m["model_margin_mae_on_lined"] = float((lined["margin"] - lined["pred_margin"]).abs().mean())
    lt = oos.dropna(subset=["line_total"])
    if len(lt) >= 200:
        m["market_total_mae"] = float((lt["total"] - lt["line_total"]).abs().mean())
        m["model_total_mae_on_lined"] = float((lt["total"] - lt["pred_total"]).abs().mean())
    return m


def compare(base: pd.DataFrame, run: pd.DataFrame, mask_col: str | None = None) -> dict:
    """Does the extra group help? Paired on identical games, in both halves."""
    j = base.merge(run, on="game_id", suffixes=("_b", "_r"))
    if mask_col:
        j = j[j[mask_col + "_b"]] if mask_col + "_b" in j else j
    res = {}
    for tgt in ("margin", "total"):
        eb = (j[f"{tgt}_b"] - j[f"pred_{tgt}_b"]).abs().to_numpy()
        er = (j[f"{tgt}_r"] - j[f"pred_{tgt}_r"]).abs().to_numpy()
        d, t, n = paired(er, eb)
        h = len(er) // 2
        halves = [paired(er[:h], eb[:h])[:2], paired(er[h:], eb[h:])[:2]]
        res[tgt] = {"n": n, "delta": round(d, 3), "t": round(t, 2),
                    "halves": [[round(x, 3), round(y, 2)] for x, y in halves]}
    return res


def adopt(cmp: dict, tgt: str) -> bool:
    r = cmp.get(tgt) or {}
    hv = r.get("halves") or [[np.nan], [np.nan]]
    return bool(r.get("t", 0) <= -2 and all(x[0] < 0 for x in hv))


def fit_final(df: pd.DataFrame, extra_m: list[str], extra_t: list[str]):
    d = df[df["done"].astype(bool) & df["rate_margin"].notna() & (df["rate_games"] >= 5)].copy()
    d["margin"] = d["home_pts"] - d["away_pts"]
    d["total"] = d["home_pts"] + d["away_pts"]
    cm, ct = BASE + extra_m, BASE_T + extra_t
    Xm, Xt = _X(d, cm), _X(d, ct)
    mm = Ridge(alpha=1.0).fit(Xm, d["margin"])
    mt = Ridge(alpha=1.0).fit(Xt, d["total"])
    pm = mm.predict(Xm)
    lr = LogisticRegression(C=1.0).fit(pm.reshape(-1, 1), (d["margin"] > 0).astype(int))
    sig_m = float(np.std(d["margin"] - pm))
    sig_t = float(np.std(d["total"] - mt.predict(Xt)))
    return {"m": mm, "t": mt, "lr": lr, "cols_m": list(Xm.columns), "cols_t": list(Xt.columns),
            "cm": cm, "ct": ct, "sig_m": sig_m, "sig_t": sig_t,
            "coef_m": dict(zip(Xm.columns, np.round(mm.coef_, 3))),
            "coef_t": dict(zip(Xt.columns, np.round(mt.coef_, 3)))}


def predict(fit: dict, rows: pd.DataFrame) -> pd.DataFrame:
    Xm = _X(rows, fit["cm"]).reindex(columns=fit["cols_m"], fill_value=0.0)
    Xt = _X(rows, fit["ct"]).reindex(columns=fit["cols_t"], fill_value=0.0)
    out = rows.copy()
    out["pred_margin"] = fit["m"].predict(Xm)
    out["pred_total"] = fit["t"].predict(Xt)
    out["home_win_prob"] = fit["lr"].predict_proba(out[["pred_margin"]].to_numpy())[:, 1]
    z = norm.ppf(0.75)
    out["margin_p25"] = out["pred_margin"] - z * fit["sig_m"]
    out["margin_p75"] = out["pred_margin"] + z * fit["sig_m"]
    out["total_p25"] = out["pred_total"] - z * fit["sig_t"]
    out["total_p75"] = out["pred_total"] + z * fit["sig_t"]
    return out
