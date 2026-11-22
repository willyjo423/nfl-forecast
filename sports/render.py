"""NBA and NHL pages, in the NFL page's style, with the sport tabs."""
from __future__ import annotations

import datetime as dt
import html
from zoneinfo import ZoneInfo

import numpy as np

from dashboard import CSS

ET = ZoneInfo("America/New_York")
TAB_CSS = ""


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def tabs(on: str) -> str:
    items = [("nfl", "NFL", "index.html"), ("nba", "NBA", "nba.html"), ("nhl", "NHL", "nhl.html")]
    return '<nav class="tabs">' + "".join(
        f'<a href="{href}" class="{"on" if k == on else ""}">{lbl}</a>' for k, lbl, href in items) + "</nav>"


def half(x: float) -> float:
    return round(x * 2) / 2


def american(p: float) -> str:
    p = min(max(p, 0.01), 0.99)
    return f"-{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def _f(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def payload(sport: str, pred, metrics: dict, tonight: dict, now: dt.datetime) -> dict:
    games = []
    for r in (pred.sort_values(["date", "start_utc"]).to_dict("records") if len(pred) else []):
        try:
            t = dt.datetime.fromisoformat(str(r["start_utc"]).replace("Z", "+00:00")).astimezone(ET)
            when = t.strftime("%a %-I:%M %p ET")
        except Exception:                                      # noqa: BLE001
            when = r["date"]
        nd = 0 if sport == "nba" else 1
        m, tot = float(r["pred_margin"]), float(r["pred_total"])
        hp, ap = (tot + m) / 2, (tot - m) / 2
        g = {"game_id": r["game_id"], "date": r["date"], "when": when, "home": r["home"], "away": r["away"],
             "forecast": {"home_points": round(hp, nd), "away_points": round(ap, nd),
                          "margin": round(m, 1), "total": round(tot, 1),
                          "home_win_prob": round(float(r["home_win_prob"]), 4),
                          "margin_p25": round(float(r["margin_p25"]), 1),
                          "margin_p75": round(float(r["margin_p75"]), 1),
                          "total_p25": round(float(r["total_p25"]), 1),
                          "total_p75": round(float(r["total_p75"]), 1)},
             "rest": {"home": _f(r.get("home_rest"), 0), "away": _f(r.get("away_rest"), 0)},
             "market": {"margin": _f(r.get("line_margin")), "total": _f(r.get("line_total")),
                        "moneyline": [r["ml_home"], r["ml_away"]] if r.get("ml_home") is not None
                        and not (isinstance(r.get("ml_home"), float) and np.isnan(r["ml_home"])) else None}}
        g["flags"] = _flags(sport, r, metrics.get("flags") or {})
        if sport == "nba":
            g["dfs"] = {"home_avail": _f(r.get("home_dfs_avail")), "away_avail": _f(r.get("away_dfs_avail")),
                        "home_change": _f(r.get("home_dfs_change")), "away_change": _f(r.get("away_dfs_change")),
                        "home_out": (tonight.get(r["home"]) or {}).get("dfs_out", []),
                        "away_out": (tonight.get(r["away"]) or {}).get("dfs_out", [])}
        else:
            sk = (tonight or {}).get("skaters") or {}
            g["skaters"] = {"home_avail": _f(r.get("home_sk_avail")), "away_avail": _f(r.get("away_sk_avail")),
                            "home_change": _f(r.get("home_sk_change")), "away_change": _f(r.get("away_sk_change")),
                            "home_out": (sk.get(r["home"]) or {}).get("sk_out", []),
                            "away_out": (sk.get(r["away"]) or {}).get("sk_out", [])}
            g["goalies"] = {s: {"name": r.get(f"{s}_goalie"), "start_prob": _f(r.get(f"{s}_gpp"), 2),
                                "status": r.get(f"{s}_gst"), "quality": _f(r.get(f"{s}_gq"), 2)}
                            for s in ("home", "away")}
        games.append(g)
    return {"sport": sport, "generated_at": now.isoformat(), "games": games, "model_metrics": metrics}


def _flags(sport: str, r: dict, table: dict) -> list[dict]:
    """Games whose model-vs-line gap falls in a range where one side has hit
    above break-even this season and last."""
    from sports import vs_line as VL
    out = []
    home, away = r["home"], r["away"]
    lm, lt = _f(r.get("line_margin")), _f(r.get("line_total"))
    if sport == "nba" and lm is not None:
        gap = float(r["pred_margin"]) - lm
        row = VL.flag_for(table, "spread", gap)
        if row and abs(gap) >= 1e-9:
            model_home = gap > 0
            take_home = model_home if row["flag"] == "model" else not model_home
            t = home if take_home else away
            num = -lm if take_home else lm
            out.append({"market": "Spread", "take": f"{t} {num:+g}" if num else f"{t} PK",
                        "why": row["flag"], "hit": row["hit"], "n": row["n"], "gap": row["gap"]})
    if lt is not None:
        gap = float(r["pred_total"]) - lt
        row = VL.flag_for(table, "total", gap)
        if row and abs(gap) >= 1e-9:
            over = gap > 0 if row["flag"] == "model" else gap < 0
            out.append({"market": "Total", "take": f"{'Over' if over else 'Under'} {lt:g}",
                        "why": row["flag"], "hit": row["hit"], "n": row["n"], "gap": row["gap"]})
    mh, ma = r.get("ml_home"), r.get("ml_away")
    if sport == "nhl" and mh is not None and ma is not None and not (isinstance(mh, float) and np.isnan(mh)):
        ih = -mh / (-mh + 100) if mh < 0 else 100 / (mh + 100)
        ia = -ma / (-ma + 100) if ma < 0 else 100 / (ma + 100)
        edge = (float(r["home_win_prob"]) - ih / (ih + ia)) * 100
        row = VL.flag_for(table, "moneyline", edge)
        if row:
            model_home = edge > 0
            take_home = model_home if row["flag"] == "model" else not model_home
            t, o = (home, mh) if take_home else (away, ma)
            out.append({"market": "Moneyline", "take": f"{t} {o:+g}", "why": row["flag"],
                        "hit": row["hit"], "n": row["n"], "gap": row["gap"]})
    return out


def _card(sport: str, g: dict) -> str:
    f = g["forecast"]
    home, away = g["home"], g["away"]
    p = f["home_win_prob"]
    hp_, ap_ = f["home_points"], f["away_points"]
    unit = "pts" if sport == "nba" else "goals"
    lead = (f'{_e(home)} <span class="num">{hp_}&ndash;{ap_}</span>' if hp_ >= ap_
            else f'{_e(away)} <span class="num">{ap_}&ndash;{hp_}</span>')
    fav = home if p >= 0.5 else away
    conf = round(100 * max(p, 1 - p))
    hpct = round(100 * p)
    bar = (f'<div class="prob-bar"><i class="{"trail" if p >= .5 else "lead"}" style="width:{100 - hpct}%"></i>'
           f'<i class="{"lead" if p >= .5 else "trail"}" style="width:{hpct}%"></i></div>'
           f'<div class="prob-ends"><span>{_e(away)} {100 - hpct}%</span><span>{_e(home)} {hpct}%</span></div>')
    mk = g.get("market") or {}
    import lines_view as LV
    if sport == "nba":
        lines = LV.table(home, away, f["margin"], f["total"], mk.get("margin"), mk.get("total"), p)
    else:
        ml = mk.get("moneyline") or ()
        lines = LV.table(home, away, None, f["total"], None, mk.get("total"), p,
                         moneyline=tuple(ml), total_fmt="1")
    extra = ""
    if sport == "nba" and g.get("dfs"):
        d = g["dfs"]

        def side(t, av, ch, outs):
            if av is None:
                return f"{_e(t)}: no DFS board yet"
            c = "" if ch is None else f" ({ch:+.0f} vs usual)"
            o = f" &middot; out: {_e(', '.join(outs[:4]))}" if outs else ""
            return f"{_e(t)} <b>{av:.0f}</b> DK pts available{c}{o}"
        extra = (f'<div class="dfs">DFS team strength &mdash; {side(away, d["away_avail"], d["away_change"], d["away_out"])}'
                 f'<br>{side(home, d["home_avail"], d["home_change"], d["home_out"])}</div>')
    if sport == "nhl" and g.get("goalies"):
        def gl(t, x):
            nm = x.get("name") or "unknown"
            st = x.get("status") or ""
            pp = x.get("start_prob")
            pps = f" {round(pp * 100)}%" if pp is not None else ""
            q = x.get("quality")
            qs = "" if q is None else f" &middot; {q:+.2f} goals/game vs average"
            return f"{_e(t)}: <b>{_e(nm)}</b> ({_e(st)}{pps}){qs}"
        sk = g.get("skaters") or {}

        def skl(t, av, ch, outs):
            if av is None:
                return f"{_e(t)}: no DFS board yet"
            c = "" if ch is None else f" ({ch:+.0f} vs usual)"
            o = f" &middot; out: {_e(', '.join(outs[:4]))}" if outs else ""
            return f"{_e(t)} <b>{av:.0f}</b> DK pts dressed{c}{o}"
        extra = (f'<div class="dfs">DFS skater strength &mdash; '
                 f'{skl(away, sk.get("away_avail"), sk.get("away_change"), sk.get("away_out") or [])}'
                 f'<br>{skl(home, sk.get("home_avail"), sk.get("home_change"), sk.get("home_out") or [])}</div>'
                 f'<div class="dfs">Goalies (DFS confirmations) &mdash; {gl(away, g["goalies"]["away"])}'
                 f'<br>{gl(home, g["goalies"]["home"])}</div>')
    fl = g.get("flags") or []
    if fl:
        items = "".join(
            f'<div style="margin-top:4px"><span class="flagnew" style="background:var(--good);color:#04140a">FLAG</span> '
            f'<b>{_e(x["take"])}</b> <span style="color:var(--dim)">({_e(x["market"])}: '
            f'{"model&#39;s side" if x["why"] == "model" else "market side &mdash; fading the model"}; '
            f'{x["hit"] * 100:.1f}% over {x["n"]} games with a {_e(x["gap"])} gap)</span></div>'
            for x in fl)
        extra = f'<div class="dfs" style="border-top-color:var(--good)">{items}</div>' + extra
    rest = g.get("rest") or {}
    b2b = [t for t, r in ((away, rest.get("away")), (home, rest.get("home"))) if r is not None and r <= 1]
    meta = f'<div class="meta">Back-to-back: {_e(", ".join(b2b))}</div>' if b2b else ""
    return (f'<div class="card"><div class="head"><span class="match">{_e(away)} at {_e(home)}</span>'
            f'<span class="when">{_e(g["when"])}</span></div>'
            f'<div class="verdict">{lead}</div><div class="because">{_e(fav)} to win &mdash; <b>{conf}%</b></div>'
            f'{bar}{lines}{extra}{meta}</div>')


def page(sport: str, payload: dict) -> str:
    name = sport.upper()
    m = payload.get("model_metrics") or {}
    cells = [("Games", str(len(payload["games"])))]
    if m.get("margin_mae"):
        cells.append(("Margin MAE", f'{m["margin_mae"]:.2f}'))
    if m.get("total_mae"):
        cells.append(("Total MAE", f'{m["total_mae"]:.2f}'))
    if m.get("win_accuracy"):
        cells.append(("Winner picked", f'{m["win_accuracy"] * 100:.0f}%'))
    strip = '<div class="strip">' + "".join(f"<div>{_e(k)}<b>{_e(v)}</b></div>" for k, v in cells) + "</div>"
    when = payload["generated_at"][:16].replace("T", " ")
    flagged = [(g, x) for g in payload["games"] for x in (g.get("flags") or [])]
    if flagged:
        strip += ('<div class="note" style="border-left-color:var(--good)"><b>Flagged plays</b> &mdash; games whose '
                  'model-vs-line gap is in a range where one side has hit above 52.4% (this season + last, 50+ games):'
                  + "".join(f'<div style="margin-top:6px">{_e(g["away"])} at {_e(g["home"])} &middot; '
                            f'<b>{_e(x["take"])}</b> ({_e(x["market"])}, {x["hit"] * 100:.1f}% / {x["n"]} games'
                            f'{", market side" if x["why"] == "market" else ""})</div>' for g, x in flagged)
                  + '</div>')
    strip += _record(m.get("vs_line"), "This season") + _record(m.get("vs_line_last_season"), "Last season")
    driver = ("Driven by the DFS models: each team's strength tonight from the DFS player projections "
              "(who is playing, late scratches included), on top of team ratings and rest."
              if sport == "nba" else
              "Driven by the DFS models: each team's skater strength tonight from the DFS projections "
              "(who dresses), the starting goalie from the DFS confirmations and his record, on top of "
              "team ratings and rest. Moneylines are fair odds from the win probability.")
    games = payload["games"]
    by_day: dict = {}
    for g in games:
        by_day.setdefault(g["date"], []).append(g)
    body = "".join(
        f'<h2 style="font-size:15px;color:var(--dim);margin:18px 0 10px">'
        f'{dt.date.fromisoformat(d).strftime("%A %b %-d")}</h2>' + "".join(_card(sport, g) for g in gs)
        for d, gs in sorted(by_day.items())) or '<div class="card">No games today or tomorrow.</div>'
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{name} forecasts</title><style>{CSS}{TAB_CSS}</style></head><body>"
            f'<div class="wrap">{tabs(sport)}<h1>{name} forecasts</h1>'
            f'<p class="sub">Generated {_e(when)} ET &middot; {_e(driver)}</p>{strip}{body}'
            '<footer>Forecasts only. No betting advice.</footer></div></body></html>')


def _record(v: dict | None, title: str) -> str:
    """Model's lean vs the line, by how far apart they were."""
    if not v or not v.get("games_with_line"):
        return ""
    parts = []
    for key, name in (("spread", "Spread (model's side vs the line)"), ("moneyline", "Moneyline (model's side at the book's price)"),
                      ("total", "Total (model's over/under vs the line)")):
        rows = v.get(key) or []
        if not rows:
            continue
        body = "".join(
            f'<tr><td class="k">{_e(r["gap"])}</td><td class="m">{r["won"]}&ndash;{r["lost"]}</td>'
            f'<td class="l">{r["pct"] * 100:.1f}%</td><td class="lean"><span>{r["n"]} games</span></td></tr>'
            for r in rows)
        extra = ""
        if key == "moneyline" and v.get("moneyline_units") is not None:
            extra = f'<div class="meta">1 unit on every lean: {v["moneyline_units"]:+.2f} units</div>'
        parts.append(f'<div style="font-size:13px;color:var(--dim);margin-top:8px">{name}</div>'
                     f'<table class="mvl"><tr><th>Gap</th><th>W&ndash;L</th><th>Win %</th><th></th></tr>{body}</table>{extra}')
    if not parts:
        return ""
    return (f'<details class="card" style="padding:12px 16px"><summary style="cursor:pointer;font-weight:600">'
            f'{_e(title)} vs the line &mdash; {v["games_with_line"]} games with a saved line</summary>'
            f'{"".join(parts)}<div class="meta">Each game forecast before it was played (walk-forward). '
            f'Break-even at standard -110 pricing is 52.4%.</div></details>')
