"""One clear 'model vs line' table for every card (NFL, NBA, NHL).

            Model       Line        Model's lean
  Spread    HOU -5      HOU -12.5   IND +12.5  (7.5 pts)
  Total     44          44.5        Under 44.5 (0.5)
  Win       HOU 63%

Margins are home-perspective (positive = home team favoured), as everywhere
else in the pipeline.
"""
from __future__ import annotations

import html

CSS = """
.mvl{width:100%;border-collapse:collapse;margin:12px 0 4px;font-size:14px;font-variant-numeric:tabular-nums}
.mvl th{font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.05em;font-weight:600;
  text-align:left;padding:0 6px 4px 0}
.mvl td{padding:7px 6px 7px 0;border-top:1px solid var(--line);vertical-align:top}
.mvl td.k{color:var(--faint);font-size:12px;text-transform:uppercase;letter-spacing:.04em;width:62px}
.mvl td.m{font-weight:700}
.mvl td.l{color:var(--dim)}
.mvl td.lean b{color:var(--good)}
.mvl td.lean span{color:var(--faint);font-size:12px}
"""


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _half(x: float) -> float:
    return round(x * 2) / 2


def spread_text(home: str, away: str, margin: float) -> str:
    """'HOU -5' for a home margin of +5; "Pick'em" at 0."""
    s = _half(margin)
    if s == 0:
        return "Pick'em"
    return f"{home} -{abs(s):g}" if s > 0 else f"{away} -{abs(s):g}"


def _at_line(team: str, team_margin_line: float) -> str:
    """The team's spread at the line: favoured by 3 -> 'BUF -3', dog -> '+3'."""
    if team_margin_line == 0:
        return f"{team} PK"
    return f"{team} {-team_margin_line:+g}"


def american(p: float) -> str:
    p = min(max(p, 0.01), 0.99)
    return f"-{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def implied(odds: float) -> float:
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def table(home: str, away: str, model_margin: float | None, model_total: float | None,
          line_margin: float | None, line_total: float | None, win_prob: float | None,
          moneyline: tuple | None = None, total_fmt: str = "g") -> str:
    """moneyline=(home_odds, away_odds) from the book, or () when there is no
    line yet; None for spread sports."""
    rows = []
    if moneyline is not None:
        p = win_prob if win_prob is not None else 0.5
        fav, pf = (home, p) if p >= 0.5 else (away, 1 - p)
        dog = away if fav == home else home
        model = f"{_e(fav)} {american(pf)}<br><span style='font-weight:400;color:var(--dim)'>{_e(dog)} {american(1 - pf)}</span>"
        if moneyline:
            ho, ao = moneyline
            ih, ia = implied(ho), implied(ao)
            mh = ih / (ih + ia)                      # vig removed
            line = f"{_e(home)} {ho:+g}<br>{_e(away)} {ao:+g}"
            edge = p - mh
            if abs(edge) < 0.01:
                lean = "<span>agrees with the line</span>"
            elif edge > 0:
                lean = f"<b>{_e(home)} {ho:+g}</b> <span>(model {round(100 * p)}% vs line {round(100 * mh)}%)</span>"
            else:
                lean = f"<b>{_e(away)} {ao:+g}</b> <span>(model {round(100 * (1 - p))}% vs line {round(100 * (1 - mh))}%)</span>"
        else:
            line, lean = "&mdash;", "<span>no line yet</span>"
        rows.append(("Moneyline", model, line, lean))
    elif model_margin is not None:
        line = spread_text(home, away, line_margin) if line_margin is not None else "&mdash;"
        lean = "<span>no line yet</span>"
        if line_margin is not None:
            d = model_margin - line_margin
            if abs(d) < 0.5:
                lean = "<span>agrees with the line</span>"
            elif d > 0:
                lean = f"<b>{_e(_at_line(home, line_margin))}</b> <span>({abs(d):.1f} pts)</span>"
            else:
                lean = f"<b>{_e(_at_line(away, -line_margin))}</b> <span>({abs(d):.1f} pts)</span>"
        rows.append(("Spread", _e(spread_text(home, away, model_margin)),
                     _e(line) if line_margin is not None else line, lean))
    if model_total is not None:
        mt = _half(model_total) if total_fmt == "g" else round(model_total, 1)
        line = f"{line_total:g}" if line_total is not None else "&mdash;"
        lean = "<span>no line yet</span>"
        if line_total is not None:
            d = model_total - line_total
            if abs(d) < (0.5 if total_fmt == "g" else 0.1):
                lean = "<span>agrees with the line</span>"
            else:
                lean = (f"<b>{'Over' if d > 0 else 'Under'} {line_total:g}</b> "
                        f"<span>({abs(d):.1f})</span>")
        rows.append(("Total", f"{mt:g}", line, lean))
    if win_prob is not None and moneyline is None:
        fav = home if win_prob >= 0.5 else away
        rows.append(("Win", f"{_e(fav)} {round(100 * max(win_prob, 1 - win_prob))}%", "&mdash;", ""))
    body = "".join(f'<tr><td class="k">{k}</td><td class="m">{m}</td><td class="l">{l}</td>'
                   f'<td class="lean">{n}</td></tr>' for k, m, l, n in rows)
    return ('<table class="mvl"><tr><th></th><th>Model</th><th>Line</th><th>Model\'s lean</th></tr>'
            f"{body}</table>")
