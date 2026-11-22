"""Model vs market: does the model add anything the closing line doesn't have?

The model is ~0.36 pts behind the closing line, so on its own it cannot beat
it. The sharper question is whether it carries INDEPENDENT information. If it
does, a blend of line and model beats the line alone, and the blend - not the
raw model - is the number worth forecasting with. If it does not, the blend
weight comes out at ~0 and the line is the forecast.

Everything here is walk-forward: the weight used for a season is fitted only
on earlier seasons, and every claim is checked in the early and late halves
separately (the guard that caught overfit rules in the DFS backtests).

    report = market_blend(oos, X_all)   # from build.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean of a-b and its t statistic."""
    d = a - b
    if len(d) < 30:
        return float("nan"), float("nan")
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def _blend_walk(df: pd.DataFrame, pred: str, line: str, target: str) -> pd.Series:
    """Blend weight on the model chosen from prior seasons only."""
    out = pd.Series(np.nan, index=df.index)
    for s in sorted(df["season_tested"].unique()):
        prior = df[df["season_tested"] < s]
        if len(prior) < 500:
            continue
        maes = [np.abs(w * prior[pred] + (1 - w) * prior[line] - prior[target]).mean()
                for w in WEIGHTS]
        w = float(WEIGHTS[int(np.argmin(maes))])
        cur = df["season_tested"] == s
        out[cur] = w * df.loc[cur, pred] + (1 - w) * df.loc[cur, line]
        df.loc[cur, "_w"] = w
    return out


def _ats(df: pd.DataFrame, pick_from: str, line: str) -> dict:
    edge = df[pick_from] - df[line]
    win = np.where(edge > 0, df["margin"] > df[line], df["margin"] < df[line])
    push = np.isclose(df["margin"], df[line]) | (edge.abs() < 1e-9)
    w = pd.Series(win, index=df.index)[~push]
    return {"n": int(len(w)), "win_pct": round(float(w.mean()), 4) if len(w) else None}


def market_blend(oos: pd.DataFrame, X: pd.DataFrame | None = None) -> dict:
    res: dict = {}
    m = oos.dropna(subset=["market_margin", "margin"]).copy()
    if len(m) < 500:
        return res
    m["_w"] = np.nan
    m["blend_margin"] = _blend_walk(m, "pred_margin", "market_margin", "margin")
    b = m.dropna(subset=["blend_margin"])
    err_line = (b["margin"] - b["market_margin"]).abs().to_numpy()
    err_blend = (b["margin"] - b["blend_margin"]).abs().to_numpy()
    d, t = _paired(err_blend, err_line)
    half = len(b) // 2
    halves = [_paired(err_blend[sl], err_line[sl]) for sl in (slice(0, half), slice(half, None))]
    res["margin"] = {
        "games": int(len(b)), "weights_by_season": {int(k): float(v) for k, v in
                                                   b.groupby("season_tested")["_w"].first().items()},
        "line_mae": round(float(err_line.mean()), 3), "blend_mae": round(float(err_blend.mean()), 3),
        "blend_minus_line": round(d, 3), "t": round(t, 2),
        "halves": [{"delta": round(h[0], 3), "t": round(h[1], 2)} for h in halves]}

    # Where the model was built to help: a new starting quarterback.
    if X is not None and {"home_qb_new", "away_qb_new"} <= set(X.columns):
        qb = X.reindex(b.index)[["home_qb_new", "away_qb_new"]].fillna(0).max(axis=1) > 0
        sub = {}
        for label, mask in (("new starter", qb.to_numpy()), ("settled", ~qb.to_numpy())):
            dd, tt = _paired(err_blend[mask], err_line[mask])
            sub[label] = {"n": int(mask.sum()), "blend_minus_line": round(dd, 3), "t": round(tt, 2)}
        res["margin"]["by_qb"] = sub
        # ATS against the CLOSING line, picks only where model and line
        # disagree by 2.5+ on a new-starter game (the subset with signal).
        g = b[qb.to_numpy()]
        g = g[(g["pred_margin"] - g["market_margin"]).abs() >= 2.5]
        res["margin"]["ats_new_starter_2p5"] = _ats(g, "pred_margin", "market_margin")
        gh = [g.iloc[: len(g) // 2], g.iloc[len(g) // 2:]]
        res["margin"]["ats_new_starter_2p5_halves"] = [_ats(x, "pred_margin", "market_margin") for x in gh]

    tt_ = oos.dropna(subset=["total_line", "total"]).copy()
    if len(tt_) >= 500:
        tt_["_w"] = np.nan
        tt_["blend_total"] = _blend_walk(tt_, "pred_total", "total_line", "total")
        bt = tt_.dropna(subset=["blend_total"])
        el = (bt["total"] - bt["total_line"]).abs().to_numpy()
        eb = (bt["total"] - bt["blend_total"]).abs().to_numpy()
        d2, t2 = _paired(eb, el)
        res["total"] = {"games": int(len(bt)),
                        "weights_by_season": {int(k): float(v) for k, v in
                                              bt.groupby("season_tested")["_w"].first().items()},
                        "line_mae": round(float(el.mean()), 3), "blend_mae": round(float(eb.mean()), 3),
                        "blend_minus_line": round(d2, 3), "t": round(t2, 2)}
    return res


def summarize(r: dict) -> str:
    if not r:
        return "  (not enough lined games)"
    out = []
    mg = r["margin"]
    out.append(f"  margin: line MAE {mg['line_mae']:.3f}, line+model blend {mg['blend_mae']:.3f} "
               f"({mg['blend_minus_line']:+.3f}, t = {mg['t']:+.2f}) over {mg['games']} games")
    out.append("    halves: " + ", ".join(f"{h['delta']:+.3f} (t {h['t']:+.2f})" for h in mg["halves"]))
    out.append(f"    model weight by season: {mg['weights_by_season']}")
    for k, v in (mg.get("by_qb") or {}).items():
        out.append(f"    {k:12s} n={v['n']:5d}  blend-line {v['blend_minus_line']:+.3f} (t {v['t']:+.2f})")
    if "ats_new_starter_2p5" in mg:
        a = mg["ats_new_starter_2p5"]
        out.append(f"    ATS, new-starter games where model and line differ by 2.5+: "
                   f"{a['win_pct']} on {a['n']} (halves {mg['ats_new_starter_2p5_halves']}); "
                   f"break-even 0.524")
    if "total" in r:
        t = r["total"]
        out.append(f"  total: line MAE {t['line_mae']:.3f}, blend {t['blend_mae']:.3f} "
                   f"({t['blend_minus_line']:+.3f}, t = {t['t']:+.2f})")
    return "\n".join(out)
