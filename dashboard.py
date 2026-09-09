"""The forecast page.

Design decisions carried over from the college build, each of which came from
getting it wrong first:

* **The forecast leads, not a pick.** The model trails the closing line, and a
  page that opens with a recommendation invites a use it cannot support.
* **No bare signed margins.** A minus sign means "the home team lost by" in an
  outcome range and "is favoured by" on a market line, and those readings only
  disagree in the minority of games where the home team is the underdog. So
  ranges are written out in words with a named team, and the bar carries
  explicit direction labels at both ends.
* **The probability bar is coloured by the favoured side**, not the home side.
  Green on the 19% end reads as an endorsement of the wrong team.
* **A difference is only called a difference when it is big enough.** The
  college page once announced that "bigger outcomes belong to Team 037" off
  11.7 against 11.5.
* **Missing means missing.** No reading is invented to fill a gap.

New here: the starting quarterbacks, and a flag when one of them is not the man
who started that team's last game. That was the only feature group in the whole
model that measured as a real improvement, and it is worth nearly half a point
of accuracy on the games where it fires, so it belongs on the card.
"""
from __future__ import annotations

import html
import json

CSS = """
:root{
  --bg:#0f1115; --card:#171a21; --line:#252a34; --ink:#e8eaed; --dim:#9aa2ad;
  --faint:#6b7280; --accent:#4ea1ff; --good:#3fb950; --warn:#d29922;
  --home:#4ea1ff; --away:#f0883e;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --ink:#14171c;
         --dim:#5b6472; --faint:#8b95a3; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:28px 18px 64px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin:0 0 22px}
.strip{display:flex;flex-wrap:wrap;gap:14px;background:var(--card);
  border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:20px}
.strip div{font-size:12px;color:var(--dim)}
.strip b{display:block;font-size:17px;color:var(--ink);font-variant-numeric:tabular-nums}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:8px;padding:14px 16px;margin-bottom:22px;font-size:13px;color:var(--dim)}
.note b{color:var(--ink)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px;margin-bottom:14px}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:12px}
.match{font-weight:600;font-size:16px}
.when{color:var(--faint);font-size:12px;white-space:nowrap}
.verdict{font-size:19px;font-weight:600;margin-bottom:2px}
.verdict .num{font-variant-numeric:tabular-nums;color:var(--accent)}
.because{color:var(--dim);font-size:13px;margin-bottom:10px}
.prob-bar{display:flex;height:7px;border-radius:4px;overflow:hidden;background:var(--line)}
.prob-bar i{display:block}
.prob-bar i.lead{background:var(--good)}
.prob-bar i.trail{background:var(--faint);opacity:.45}
.prob-ends,.band-ends{display:flex;justify-content:space-between;
  font-size:11px;color:var(--faint);margin-top:4px}
.band-row{margin:14px 0 4px}
.band{position:relative;height:22px}
.track{position:absolute;top:10px;left:0;right:0;height:2px;background:var(--line)}
.iqr{position:absolute;top:6px;height:10px;background:var(--accent);opacity:.32;border-radius:3px}
.zero{position:absolute;top:2px;width:1px;height:18px;background:var(--faint)}
.med{position:absolute;top:2px;width:2px;height:18px;background:var(--accent)}
.band-label{font-size:13px;color:var(--dim);margin-top:6px}
.band-label b{color:var(--ink)}
.band-label.total{margin-top:2px}
.qbs{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0 2px;font-size:13px}
.qb{color:var(--dim)}
.qb b{color:var(--ink)}
.flagnew{display:inline-block;background:var(--warn);color:#1a1200;font-size:10px;
  font-weight:700;border-radius:3px;padding:1px 5px;margin-left:5px;
  letter-spacing:.03em;vertical-align:1px}
.meta{color:var(--faint);font-size:12px;margin-top:10px}
.market{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
.cmp-row{display:flex;gap:14px;font-size:13px;padding:2px 0}
.cmp-k{width:66px;color:var(--faint);font-size:11px;text-transform:uppercase;
  letter-spacing:.05em;padding-top:2px}
.cmp-v{color:var(--dim);font-variant-numeric:tabular-nums}
.gap{color:var(--faint);font-size:12px;margin-top:6px}
footer{color:var(--faint);font-size:12px;margin-top:32px;text-align:center}
"""


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _forecast(g: dict) -> str:
    f = g.get("forecast") or {}
    if not f:
        return ""
    home, away = g["home_team"], g["away_team"]
    hp, ap = f.get("home_points"), f.get("away_points")
    prob = f.get("home_win_prob") or 0.5
    home_pct = int(round(prob * 100))

    if hp is None or ap is None:
        line = ""
    elif hp >= ap:
        line = f'{_e(home)} <span class="num">{hp}&ndash;{ap}</span>'
    else:
        line = f'{_e(away)} <span class="num">{ap}&ndash;{hp}</span>'

    fav = home if home_pct >= 50 else away
    conf = home_pct if home_pct >= 50 else 100 - home_pct

    # Colour the favoured side, not the home side.
    hc = "lead" if home_pct >= 50 else "trail"
    ac = "trail" if home_pct >= 50 else "lead"
    return (
        f'<div class="verdict">{line}</div>'
        f'<div class="because">{_e(fav)} to win &mdash; <b>{conf}%</b></div>'
        f'<div class="prob-bar"><i class="{ac}" style="width:{100 - home_pct}%"></i>'
        f'<i class="{hc}" style="width:{home_pct}%"></i></div>'
        f'<div class="prob-ends"><span>{_e(away)} {100 - home_pct}%</span>'
        f'<span>{_e(home)} {home_pct}%</span></div>')


def _band(g: dict) -> str:
    """The middle half of the model's own predictive distribution."""
    f = g.get("forecast") or {}
    p25, p75, med = f.get("margin_p25"), f.get("margin_p75"), f.get("margin")
    if p25 is None or p75 is None or med is None:
        return ""

    lo = min(p25, med, 0.0) - 6
    hi = max(p75, med, 0.0) + 6
    span = max(hi - lo, 1.0)

    def pos(v):
        return max(0.0, min(100.0, (v - lo) / span * 100.0))

    a, b = pos(p25), pos(p75)
    home, away = g["home_team"], g["away_team"]

    # Never a bare signed number - see the module docstring.
    if p25 >= 0 and p75 >= 0:
        label = (f'Half of the time, <b>{_e(home)}</b> wins by '
                 f'<b>{p25:.0f} to {p75:.0f}</b>')
    elif p25 < 0 and p75 < 0:
        label = (f'Half of the time, <b>{_e(away)}</b> wins by '
                 f'<b>{abs(p75):.0f} to {abs(p25):.0f}</b>')
    else:
        label = (f'Half of the time this lands between <b>{_e(away)} by '
                 f'{abs(p25):.0f}</b> and <b>{_e(home)} by {p75:.0f}</b>')

    # The same statement for the combined score, sitting under the margin one.
    # No direction to get wrong here, so it needs none of the naming the line
    # above does - a total is just a number, and 41 to 55 reads the same way
    # round for both teams.
    t25, t75 = f.get("total_p25"), f.get("total_p75")
    total_label = ""
    if t25 is not None and t75 is not None:
        total_label = (f'<div class="band-label total">and the two scores add '
                       f'up to between <b>{t25:.0f} and {t75:.0f}</b></div>')

    return (
        '<div class="band-row"><div class="band">'
        '<div class="track"></div>'
        f'<div class="iqr" style="left:{a:.1f}%;width:{max(b - a, 1.0):.1f}%"></div>'
        f'<div class="zero" style="left:{pos(0.0):.1f}%"></div>'
        f'<div class="med" style="left:{pos(med):.1f}%"></div></div>'
        f'<div class="band-ends"><span>&#8592; {_e(away)} wins</span>'
        f'<span>{_e(home)} wins &#8594;</span></div>'
        f'<div class="band-label">{label}</div>{total_label}</div>')


def _quarterbacks(g: dict) -> str:
    q = g.get("quarterbacks") or {}
    if not q.get("home") and not q.get("away"):
        return ""
    out = []
    for side, team in (("away", g["away_team"]), ("home", g["home_team"])):
        name = q.get(side)
        if not name:
            continue
        # Two different claims, and the first live page conflated them: a
        # quarterback returning in week 1 after resting the previous January
        # is not a new starter. "New starter" is an in-season change; "first
        # start for this team" is the one that survives an offseason.
        if q.get(f"{side}_new") == 1:
            flag = '<span class="flagnew">NEW STARTER</span>'
        elif q.get(f"{side}_first") == 1:
            flag = '<span class="flagnew">FIRST START HERE</span>'
        else:
            flag = ""
        starts = q.get(f"{side}_starts")
        tail = (f' <span style="opacity:.7">({starts:.0f} career starts)</span>'
                if starts is not None and starts < 16 else "")
        out.append(f'<span class="qb">{_e(team)}: <b>{_e(name)}</b>{tail}{flag}</span>')
    return f'<div class="qbs">{"".join(out)}</div>' if out else ""


def _context(g: dict) -> str:
    c = g.get("context") or {}
    bits = []
    for side, team in (("away", g["away_team"]), ("home", g["home_team"])):
        rest = c.get(f"{side}_rest")
        if rest is not None and rest <= 4:
            bits.append(f"{_e(team)} on a short week ({rest:.0f} days)")
        elif rest is not None and rest >= 11:
            bits.append(f"{_e(team)} off a bye")
    if c.get("div_game") == 1:
        bits.append("division game")
    if g.get("neutral_site"):
        bits.append("neutral site")
    wx = g.get("weather_text")
    if wx:
        bits.append(wx)
    played = c.get("games_played")
    if played is not None and played <= 2:
        bits.append(f"only {played:.0f} games of form so far &mdash; "
                    f"the range above is wider because of it")
    return f'<div class="meta">{" &middot; ".join(bits)}</div>' if bits else ""


def _market(g: dict) -> str:
    f = g.get("forecast") or {}
    rows = []
    home = g["home_team"]
    if g.get("market_margin") is not None:
        mm = g["market_margin"]
        side = home if mm > 0 else g["away_team"]
        rows.append(("Market", f'{_e(side)} by {abs(mm):.1f}',
                     f'total {g["market_total"]:.1f}'
                     if g.get("market_total") is not None else ""))
    if f.get("margin") is not None:
        m = f["margin"]
        side = home if m > 0 else g["away_team"]
        rows.append(("Model", f'{_e(side)} by {abs(m):.1f}',
                     f'total {f["total"]:.1f}' if f.get("total") else ""))
    if not rows:
        return '<div class="market"><div class="gap">No market line yet.</div></div>'

    body = "".join(
        f'<div class="cmp-row"><span class="cmp-k">{_e(k)}</span>'
        f'<span class="cmp-v">{a}</span><span class="cmp-v">{b}</span></div>'
        for k, a, b in rows)

    gap = ""
    if g.get("market_margin") is None:
        # The model row still stands on its own; say the comparison is absent
        # rather than leaving a row that looks like it has a counterpart.
        gap = "No market line yet, so there is nothing to compare against."
    elif g.get("vs_market") is not None:
        d = abs(g["vs_market"])
        if d < 0.5:
            gap = "The model and the market agree."
        else:
            lean = home if g["vs_market"] > 0 else g["away_team"]
            gap = (f'The model is {d:.1f} points more favourable to '
                   f'{_e(lean)} than the market. Measured over 3,757 past '
                   f'games, a gap this size predicted nothing about who '
                   f'covers &mdash; it is shown as orientation, not a play.')
    return f'<div class="market">{body}<div class="gap">{gap}</div></div>'


def _card(g: dict) -> str:
    when = g.get("kickoff_et") or ""
    return (
        '<div class="card">'
        f'<div class="head"><span class="match">{_e(g["away_team"])} at '
        f'{_e(g["home_team"])}</span><span class="when">{_e(when)}</span></div>'
        f'{_forecast(g)}{_band(g)}{_quarterbacks(g)}{_context(g)}{_market(g)}'
        '</div>')


def _strip(payload: dict) -> str:
    m = payload.get("model_metrics") or {}
    cells = [("Games", f'{len(payload.get("games", []))}')]
    if m.get("margin_mae"):
        cells.append(("Margin MAE", f'{m["margin_mae"]:.2f}'))
    if m.get("market_margin_mae"):
        cells.append(("Closing line", f'{m["market_margin_mae"]:.2f}'))
    if m.get("win_accuracy"):
        cells.append(("Winner picked", f'{m["win_accuracy"] * 100:.0f}%'))
    if m.get("ats_win_pct"):
        cells.append(("Against the spread", f'{m["ats_win_pct"] * 100:.1f}%'))
    body = "".join(f"<div>{_e(k)}<b>{_e(v)}</b></div>" for k, v in cells)
    return f'<div class="strip">{body}</div>'


def _honesty(payload: dict) -> str:
    m = payload.get("model_metrics") or {}
    gap = m.get("mae_vs_market")
    ats = m.get("ats_win_pct")
    lines = ["<b>What this page is, and is not.</b> It forecasts each game: a "
             "projected score, a win probability, and an honest range around "
             "the margin."]
    if gap is not None:
        lines.append(
            f"Out of sample it misses the final margin by "
            f"{m['margin_mae']:.2f} points on average, against "
            f"{m['market_margin_mae']:.2f} for the closing line &mdash; so the "
            f"market's number is {abs(gap):.2f} points better than this one.")
    if ats is not None:
        lines.append(
            f"Where the two disagree, backing the model went "
            f"{ats * 100:.1f}% against the spread over {m.get('ats_n', 0):,} "
            f"games, with break-even at 52.4%. That is a null result, and it "
            f"is why nothing here is presented as a bet.")
    return f'<div class="note">{" ".join(lines)}</div>'


def render(payload: dict, standalone: bool = True) -> str:
    games = payload.get("games") or []
    when = (payload.get("generated_at") or "")[:16].replace("T", " ")
    season, week = payload.get("season"), payload.get("week")

    body = (
        '<div class="wrap">'
        f'<h1>NFL forecasts &mdash; {_e(season)} week {_e(week)}</h1>'
        f'<p class="sub">Generated {_e(when)} ET. '
        f'Model figures are out-of-sample, from the last training run.</p>'
        f'{_strip(payload)}{_honesty(payload)}'
        + ("".join(_card(g) for g in games) if games
           else '<div class="card">No games scheduled.</div>')
        + '<footer>Forecasts only. No betting advice.</footer></div>')

    if not standalone:
        return body
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>NFL forecasts &mdash; {_e(season)} week {_e(week)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def render_from_file(path: str) -> str:
    with open(path) as fh:
        return render(json.load(fh))


if __name__ == "__main__":
    import sys
    print(render_from_file(sys.argv[1]))
