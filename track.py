"""Grade archived forecasts against what actually happened.

The bootstrap's walk-forward evaluation is honest, but it is still the model
marking its own homework on data that existed when it was written. This grades
only forecasts that were archived *before* the games were played, which is the
one measurement nobody can accidentally cheat on.

    python track.py            # print the report
    python track.py --write    # also write docs/results.html

If the forward record ever diverges sharply from the backtest, the backtest is
wrong and this is how you find out.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config
import dataset

log = logging.getLogger(__name__)


def load_forecasts() -> pd.DataFrame:
    """Every archived forecast, one row per game, newest file wins.

    A week is often forecast several times as lines move and injuries land.
    Grading all of them would count the same game repeatedly and quietly
    weight busy weeks more heavily, so only the last forecast of each game
    survives.
    """
    rows = []
    for path in sorted(config.FORECASTS.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skipping %s: %s", path.name, exc)
            continue
        for g in payload.get("games") or []:
            f = g.get("forecast") or {}
            rows.append({
                "game_id": g.get("game_id"),
                "file": path.name,
                "generated_at": payload.get("generated_at"),
                "season": payload.get("season"), "week": g.get("week"),
                "home_team": g.get("home_team"), "away_team": g.get("away_team"),
                "pred_margin": f.get("margin"), "pred_total": f.get("total"),
                "home_win_prob": f.get("home_win_prob"),
                "sigma": f.get("sigma"),
                "p25": f.get("margin_p25"), "p75": f.get("margin_p75"),
                "market_margin": g.get("market_margin"),
                "games_played": (g.get("context") or {}).get("games_played"),
                "qb_change": max(
                    (g.get("quarterbacks") or {}).get("home_new") or 0,
                    (g.get("quarterbacks") or {}).get("away_new") or 0),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["game_id"])
    return (df.sort_values("generated_at")
              .drop_duplicates("game_id", keep="last")
              .reset_index(drop=True))


def grade(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts to final scores, keeping only games that finished."""
    if forecasts.empty:
        return forecasts
    seasons = sorted({int(s) for s in forecasts["season"].dropna().unique()})
    actual = dataset.build_games(seasons=seasons)
    actual = actual.loc[actual["completed"],
                        ["game_id", "margin", "game_total", "home_score",
                         "away_score"]]
    actual["game_id"] = actual["game_id"].astype(str)
    df = forecasts.merge(actual, on="game_id", how="inner")
    if df.empty:
        return df

    df["margin_err"] = df["pred_margin"] - df["margin"]
    df["total_err"] = df["pred_total"] - df["game_total"]
    df["home_won"] = (df["margin"] > 0).astype(float)
    df.loc[df["margin"] == 0, "home_won"] = np.nan
    df["picked_right"] = ((df["home_win_prob"] > 0.5).astype(float)
                          == df["home_won"]).astype(float)
    df["in_band"] = ((df["margin"] >= df["p25"]) & (df["margin"] <= df["p75"]))
    return df


def summarise(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    out = {
        "n": int(len(df)),
        "seasons": sorted({int(s) for s in df["season"].dropna().unique()}),
        "margin_mae": float(df["margin_err"].abs().mean()),
        "margin_bias": float(df["margin_err"].mean()),
        "total_mae": float(df["total_err"].abs().mean()),
    }
    picked = df.dropna(subset=["picked_right"])
    if len(picked):
        out["winner_pct"] = float(picked["picked_right"].mean())

    # The stated middle half should contain about half the results. Much more
    # and the forecasts are timid; much less and they are overconfident, which
    # is the failure that matters.
    out["in_band_pct"] = float(df["in_band"].mean())
    out["in_band_target"] = 0.50

    lined = df.dropna(subset=["market_margin"])
    if len(lined) >= 10:
        out["market_mae"] = float((lined["market_margin"] - lined["margin"]).abs().mean())
        out["model_mae_on_lined"] = float(lined["margin_err"].abs().mean())
        out["vs_market"] = out["model_mae_on_lined"] - out["market_mae"]

    # Win-probability calibration in wide bands - narrow ones are meaningless
    # until there are thousands of games.
    wp = df.dropna(subset=["home_won"])
    if len(wp) >= 20:
        bins = pd.cut(wp["home_win_prob"], [0, .35, .5, .65, .8, 1.0])
        grouped = wp.groupby(bins, observed=True).agg(
            predicted=("home_win_prob", "mean"),
            observed=("home_won", "mean"), n=("home_won", "size"))
        out["calibration"] = [
            {"band": str(i), "predicted": round(float(r["predicted"]), 3),
             "observed": round(float(r["observed"]), 3), "n": int(r["n"])}
            for i, r in grouped.iterrows()]

    for key, col in (("by_week", "week"), ("by_qb_change", "qb_change")):
        if col in df.columns and df[col].notna().any():
            g = df.groupby(df[col], observed=True)["margin_err"].agg(
                n="size", mae=lambda s: float(s.abs().mean()))
            out[key] = {str(k): {"n": int(v["n"]), "mae": round(float(v["mae"]), 2)}
                        for k, v in g.iterrows()}
    return out


def report(s: dict) -> str:
    if not s:
        return ("No graded forecasts yet. They appear once archived forecasts "
                "have games that have finished.")
    lines = [
        "FORWARD-ONLY RESULTS  (forecasts archived before kickoff)",
        "=" * 62,
        f"Games graded           : {s['n']:,}",
        f"Seasons                : {', '.join(str(x) for x in s['seasons'])}",
        f"Margin MAE             : {s['margin_mae']:.2f} pts",
        f"Margin bias            : {s['margin_bias']:+.2f} pts "
        f"({'toward the home side' if s['margin_bias'] > 0 else 'toward the away side'})",
        f"Total MAE              : {s['total_mae']:.2f} pts",
    ]
    if "winner_pct" in s:
        lines.append(f"Winner picked          : {s['winner_pct'] * 100:.1f}%")
    lines.append(
        f"Landed in stated range : {s['in_band_pct'] * 100:.1f}%  "
        f"(should be about 50%)")
    if s["n"] >= 30:
        if s["in_band_pct"] < 0.40:
            lines.append("  -> the ranges are too narrow; the model is "
                         "overconfident")
        elif s["in_band_pct"] > 0.62:
            lines.append("  -> the ranges are wider than they need to be")

    if "vs_market" in s:
        d = s["vs_market"]
        lines += ["",
                  f"Closing line MAE       : {s['market_mae']:.2f} pts",
                  f"Model MAE, same games  : {s['model_mae_on_lined']:.2f} pts",
                  f"  -> {abs(d):.2f} pts "
                  f"{'worse than' if d > 0 else 'better than'} the market"]

    if s.get("calibration"):
        lines += ["", "Win probability, by band:"]
        for row in s["calibration"]:
            lines.append(f"  {row['band']:<14} said {row['predicted'] * 100:5.1f}%, "
                         f"happened {row['observed'] * 100:5.1f}%  (n={row['n']})")

    if s.get("by_qb_change"):
        lines += ["", "By whether a starting quarterback changed:"]
        for k, v in sorted(s["by_qb_change"].items()):
            label = "a new starter" if k in ("1.0", "1") else "both settled"
            lines.append(f"  {label:<16} MAE {v['mae']:.2f}  (n={v['n']})")
    return "\n".join(lines)


def render_html(s: dict) -> str:
    from dashboard import CSS
    body = report(s).replace("&", "&amp;").replace("<", "&lt;")
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>NFL forecast results</title>"
            f"<style>{CSS}pre{{white-space:pre-wrap;font:13px/1.6 ui-monospace,"
            "SFMono-Regular,Menlo,monospace}}</style></head><body>"
            f'<div class="wrap"><h1>Forward-only results</h1>'
            f'<p class="sub">Updated {when}. Every forecast here was archived '
            f'before its game kicked off.</p>'
            f'<div class="card"><pre>{body}</pre></div>'
            f'<footer><a href="./">Back to this week\'s forecasts</a></footer>'
            "</div></body></html>")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true",
                   help="also write docs/results.html")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    forecasts = load_forecasts()
    if forecasts.empty:
        print("No archived forecasts yet.")
        return 0
    graded = grade(forecasts)
    s = summarise(graded)
    print(report(s))

    if args.write:
        (config.DOCS / "results.html").write_text(render_html(s))
        (config.DATA / "results.json").write_text(json.dumps(s, indent=2, default=str))
        print(f"\nwrote {config.DOCS / 'results.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
