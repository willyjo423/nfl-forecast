"""Offline checks of the forecast job and the page it writes.

No network. The point of most of these is the display logic, because that is
where the college build's real mistakes lived - not in the model, which was
tested, but in a minus sign that meant two different things three lines apart.

    python test_daily.py
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import date

import numpy as np
import pandas as pd

import dashboard
import daily
import fixtures
import nflverse

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def section(t):
    print(f"\n{t}\n{'-' * 66}")


def game(home="KC", away="BUF", hp=27, ap=20, prob=0.68, sigma=12.0,
         margin=7.0, market=3.5, **kw):
    g = {
        "game_id": "x", "kickoff_utc": "2026-09-13T17:00:00",
        "kickoff_et": "Sun 1:00 PM", "week": 2,
        "home_team": home, "away_team": away, "neutral_site": False,
        "forecast": {"home_points": hp, "away_points": ap, "margin": margin,
                     "total": hp + ap, "home_win_prob": prob, "sigma": sigma,
                     "margin_p25": round(margin - 0.6745 * sigma, 1),
                     "margin_p75": round(margin + 0.6745 * sigma, 1)},
        "quarterbacks": {"home": "P. Mahomes", "away": "J. Allen",
                         "home_new": 0, "away_new": 0, "home_first": 0, "away_first": 0,
                         "home_starts": 120, "away_starts": 110},
        "context": {"home_rest": 7, "away_rest": 7, "div_game": 0,
                    "is_dome": 0, "games_played": 6},
        "weather": {"temp_f": 61.0, "wind_mph": 8.0},
        "weather_text": "61°F",
        "market_margin": market, "market_total": 47.5,
        "vs_market": None if market is None else round(margin - market, 1),
    }
    g.update(kw)
    return g


def payload(games, **kw):
    p = {"generated_at": "2026-09-11T08:00:00", "season": 2026, "week": 2,
         "games": games,
         "model_metrics": {"margin_mae": 10.37, "market_margin_mae": 10.01,
                           "mae_vs_market": 0.36, "win_accuracy": 0.638,
                           "ats_win_pct": 0.5039, "ats_n": 3757}}
    p.update(kw)
    return p


# ------------------------------------------------------------------ season
def test_season() -> None:
    section("SEASON AND SLATE")
    check("September is the new season", daily.current_season(date(2026, 9, 13)) == 2026)
    check("December stays in it", daily.current_season(date(2026, 12, 20)) == 2026)
    check("January belongs to the year before",
          daily.current_season(date(2027, 1, 18)) == 2026,
          "the playoffs are last season's, not this one's")
    check("February too", daily.current_season(date(2027, 2, 8)) == 2026)
    check("March starts the next", daily.current_season(date(2027, 3, 1)) == 2027)

    import dataset
    raw = fixtures.make_games(seasons=(2025, 2026), weeks=6,
                              unplayed_last_week=True)
    original = nflverse.load_games_raw
    nflverse.load_games_raw = lambda: raw
    try:
        g = dataset.build_games()
    finally:
        nflverse.load_games_raw = original

    nxt = daily.pick_slate(g, 2026, None, None)
    check("default slate is the first unplayed week", not nxt.empty
          and bool((~nxt["completed"]).any()),
          "a page opened on a Friday should show this week, not last")
    check("default slate is one week", nxt["week"].nunique() == 1)
    check("explicit week is honoured",
          set(daily.pick_slate(g, 2026, 3, None)["week"]) == {3})
    check("a season with no games returns empty",
          daily.pick_slate(g, 1998, None, None).empty)


# --------------------------------------------------------------- direction
def test_direction() -> None:
    """The bug that took three rounds to see on the college build."""
    section("DIRECTION - NO BARE SIGNED MARGINS")

    home_fav = dashboard._band(game(margin=9.0, sigma=6.0))
    check("a home favourite is named, not signed",
          "KC</b> wins by" in home_fav, home_fav[:160])
    check("no bare negative number in the label",
          not re.search(r">-\d", home_fav))

    away_fav = dashboard._band(game(margin=-9.0, sigma=6.0))
    check("an away favourite names the away team",
          "BUF</b> wins by" in away_fav, away_fav[:160])
    check("and shows a positive number for them",
          "-" not in re.sub(r"<[^>]+>", "", away_fav).split("Half")[-1],
          "an away win should read as a win, not a negative home margin")

    straddle = dashboard._band(game(margin=1.0, sigma=14.0))
    check("a range crossing zero says both directions",
          "BUF by" in straddle and "KC by" in straddle, straddle[:200])

    for g in (game(margin=9.0), game(margin=-9.0), game(margin=0.5)):
        out = dashboard._band(g)
        check(f"direction labels present (margin {g['forecast']['margin']})",
              "wins" in out and "&#8592;" in out and "&#8594;" in out)


def test_probability_bar() -> None:
    section("PROBABILITY BAR")
    home = dashboard._forecast(game(prob=0.81))
    check("home favourite: home half is the highlighted one",
          home.index('class="lead"') > home.index('class="trail"'),
          "the lead class must land on the second (home) segment")
    check("home favourite named", "KC to win" in home)

    away = dashboard._forecast(game(prob=0.19, hp=17, ap=27))
    check("away favourite named", "BUF to win" in away, away[:120])
    check("away favourite: away half is highlighted",
          away.index('class="lead"') < away.index('class="trail"'),
          "colouring the home side green at 19% endorses the wrong team")
    check("confidence is shown from the favourite's side",
          "<b>81%</b>" in away, "19% would be the wrong way round")


def test_quarterbacks() -> None:
    section("QUARTERBACKS ON THE CARD")
    plain = dashboard._quarterbacks(game())
    check("both starters listed", "P. Mahomes" in plain and "J. Allen" in plain)
    check("settled starters carry no flag", "NEW STARTER" not in plain)

    q = game()
    q["quarterbacks"] = {**q["quarterbacks"], "away_new": 1,
                         "away": "M. Trubisky", "away_starts": 3}
    flagged = dashboard._quarterbacks(q)
    check("a new starter is flagged", "NEW STARTER" in flagged)
    check("an inexperienced starter shows his starts",
          "3 career starts" in flagged, flagged)
    check("only the changed side is flagged", flagged.count("NEW STARTER") == 1)

    # The bug the first live page showed: Mahomes came back in week 1 flagged
    # NEW STARTER, because a backup had finished the previous January. Across a
    # season boundary the in-season flag is unknown, not true.
    returning = game()
    returning["quarterbacks"] = {**returning["quarterbacks"],
                                 "home_new": None, "home_first": 0}
    out = dashboard._quarterbacks(returning)
    check("a returning starter in week 1 is not called new",
          "NEW STARTER" not in out, out)

    debut = game()
    debut["quarterbacks"] = {**debut["quarterbacks"], "home_new": None,
                             "home_first": 1, "home": "C. Ward",
                             "home_starts": 0}
    out = dashboard._quarterbacks(debut)
    check("a genuine first start for the team is flagged",
          "FIRST START HERE" in out, out)
    check("and is not called an in-season change",
          "NEW STARTER" not in out.replace("FIRST START HERE", ""))

    blank = dashboard._quarterbacks(
        {**game(), "quarterbacks": {"home": None, "away": None}})
    check("unknown starters produce nothing, not a blank row", blank == "")


def test_market() -> None:
    section("MARKET ROW")
    out = dashboard._market(game(margin=7.0, market=3.5))
    check("market side is named, not signed", "KC by 3.5" in out, out[:200])
    check("model side is named", "KC by 7.0" in out)
    check("the gap is described as orientation",
          "not a play" in out and "predicted nothing" in out)

    away_line = dashboard._market(game(margin=-2.0, market=-6.0))
    check("an away-favoured line names the away team",
          "BUF by 6.0" in away_line, away_line[:200])

    agree = dashboard._market(game(margin=3.6, market=3.5))
    check("near-agreement says so", "agree" in agree)

    none = dashboard._market(game(market=None))
    check("a missing line says so plainly", "No market line yet" in none)


def test_context() -> None:
    section("CONTEXT LINE")
    short = game()
    short["context"] = {**short["context"], "away_rest": 4}
    check("a short week is called out", "short week" in dashboard._context(short))

    bye = game()
    bye["context"] = {**bye["context"], "home_rest": 13}
    check("a bye is called out", "off a bye" in dashboard._context(bye))

    thin = game()
    thin["context"] = {**thin["context"], "games_played": 1}
    check("thin evidence is explained, not hidden",
          "wider" in dashboard._context(thin),
          "the reader should be told why the range is broad")

    quiet = dashboard._context({**game(), "weather_text": "",
                                "context": {"home_rest": 7, "away_rest": 7,
                                            "div_game": 0, "is_dome": 0,
                                            "games_played": 8}})
    check("an unremarkable game adds no clutter", quiet == "")


def test_page() -> None:
    section("WHOLE PAGE")
    p = payload([game(), game(home="SF", away="LA", margin=-4.0, prob=0.41)])
    out = dashboard.render(p)

    check("is a complete document",
          out.startswith("<!doctype html") and out.rstrip().endswith("</html>"))
    check("carries a viewport for phones", 'name="viewport"' in out)
    check("styles are inline, nothing external to fetch",
          "<style>" in out and "http" not in out.split("<style>")[1].split("</style>")[0])
    check("both games rendered", out.count('class="card"') == 2)
    check("the headline says what it is", "NFL forecasts" in out)

    check("the page states it trails the market",
          "points better than this one" in out,
          "the honest framing must survive into the rendered page")
    check("the null ATS result is stated",
          "50.4%" in out and "52.4%" in out)
    check("no betting advice claimed", "No betting advice" in out)

    empty = dashboard.render(payload([]))
    check("an empty slate still renders", "No games scheduled" in empty)

    # Nothing user-supplied should be able to inject markup.
    nasty = dashboard.render(payload([game(home="<script>alert(1)</script>")]))
    check("team names are escaped", "<script>alert" not in nasty)


def test_record() -> None:
    section("JSON RECORD")
    row = pd.Series({
        "game_id": "2026_02_BUF_KC", "kickoff": pd.Timestamp("2026-09-13 17:00"),
        "week": 2, "home_team": "KC", "away_team": "BUF", "neutral_site": False,
        "pred_margin": 6.5, "pred_total": 47.0, "margin_sigma": 12.0,
        "home_win_prob": 0.70, "pred_home_points": 26.75, "pred_away_points": 20.25,
        "home_qb": "P. Mahomes", "away_qb": "J. Allen",
        "home_qb_new": 0.0, "away_qb_new": 1.0,
        "home_qb_starts": 120.0, "away_qb_starts": 110.0,
        "home_rest": 7.0, "away_rest": 4.0, "div_game": 0.0, "is_dome": 0.0,
        "home_played": 1.0, "away_played": 2.0,
        "temp_f": np.nan, "wind_mph": np.nan,
        "market_margin": 3.0, "total_line": 48.5,
    })
    r = daily._record(row)
    check("serialises to JSON", isinstance(json.dumps(r), str))
    check("missing weather becomes null, not a number",
          r["weather"]["temp_f"] is None,
          "a filled-in default would be a confident lie")
    check("the band brackets the forecast",
          r["forecast"]["margin_p25"] < r["forecast"]["margin"]
          < r["forecast"]["margin_p75"])
    check("the band width follows sigma",
          abs((r["forecast"]["margin_p75"] - r["forecast"]["margin_p25"])
              - 2 * 0.6745 * 12.0) < 0.2)
    check("evidence is the thinner of the two sides",
          r["context"]["games_played"] == 1,
          "a well-known team does not make its unknown opponent known")
    check("the gap to the market is carried", r["vs_market"] == 3.5)
    check("a new starter survives into the payload",
          r["quarterbacks"]["away_new"] == 1)


def main() -> int:
    print("NFL forecast job and page - offline checks")
    for fn in (test_season, test_direction, test_probability_bar,
               test_quarterbacks, test_market, test_context, test_page,
               test_record):
        try:
            fn()
        except Exception:  # noqa: BLE001
            global FAIL
            FAIL += 1
            FAILURES.append(f"{fn.__name__} raised")
            print(f"  CRASH in {fn.__name__}")
            traceback.print_exc()

    print(f"\n{'=' * 66}\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\nfailures:")
        for f in FAILURES:
            print(f"  - {f}")
    print("=" * 66)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
