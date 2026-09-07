"""One live run that settles the real nflverse schema.

This exists because the college build lost time to assumed field names. Rather
than guessing again, this probe fetches the live data once and prints exactly
what arrived: every column, which canonical fields resolved, how many rows
actually carry a value in each, and which columns the mapping ignores.

It also tests the one assumption the rest of the pipeline is built on - that
`spread_line` is a home-perspective margin - against real results, and says so
either way instead of trusting a comment.

    python selftest.py                 # games only, fast
    python selftest.py --pbp 2024      # also probe one play-by-play season

Nothing here is fatal. A field that turns out to be missing is a finding, not
an error, so the probe always finishes and always prints the summary.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import config
import nflverse
import schema

log = logging.getLogger("selftest")

# The fields I could not confirm from the published docs. The probe answers
# each one explicitly so the answer is not buried in a long list.
UNCONFIRMED = [
    "home_rest", "away_rest", "div_game",
    "home_moneyline", "away_moneyline",
    "home_qb", "away_qb", "home_qb_id", "away_qb_id",
    "game_type", "temp", "wind", "surface", "roof",
]

RULE = "=" * 74
THIN = "-" * 74


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def sub(title: str) -> None:
    print(f"\n{title}\n{THIN}")


def columns_block(df: pd.DataFrame) -> None:
    sub(f"EVERY COLUMN AS PUBLISHED  ({len(df.columns)} total)")
    cols = sorted(df.columns)
    width = max((len(c) for c in cols), default=10) + 2
    per_row = max(1, 74 // width)
    for i in range(0, len(cols), per_row):
        print("  " + "".join(c.ljust(width) for c in cols[i:i + per_row]))


def mapping_block(df: pd.DataFrame, mapping: dict, required: list,
                  label: str) -> dict:
    rep = schema.report(df, mapping, required)
    cov = schema.coverage(df, mapping)

    sub(f"{label}: CANONICAL FIELDS")
    print(f"  rows: {rep['n_rows']:,}")
    print(f"  resolved: {len(rep['resolved'])}/{len(mapping)}")

    if rep["missing_required"]:
        print(f"\n  !! REQUIRED AND MISSING: {rep['missing_required']}")
        print("     the pipeline cannot run without these - tell me and I "
              "will remap them")
    else:
        print("  all required fields present")

    if rep["missing"]:
        print(f"\n  missing (optional): {', '.join(rep['missing'])}")

    sub(f"{label}: COVERAGE  (share of rows carrying a real value)")
    print("  a column can exist and still be empty; this is the number that")
    print("  decides whether a feature is worth building on.\n")
    for field in sorted(cov, key=lambda f: (-cov[f], f)):
        pct = cov[field] * 100
        bar = "#" * int(round(pct / 5))
        flag = ""
        if cov[field] == 0.0:
            flag = "  <- present but entirely empty"
        elif cov[field] < 0.5:
            flag = "  <- thin"
        print(f"  {field:<18} {pct:6.1f}%  {bar:<20}{flag}")

    if rep["unmapped"]:
        sub(f"{label}: COLUMNS THE MAPPING IGNORES")
        print("  not a problem - this is where the next useful feature "
              "usually turns up.\n")
        print("  " + ", ".join(rep["unmapped"]))
    return rep


def unconfirmed_block(rep: dict, cov: dict) -> None:
    sub("THE FIELDS I COULD NOT CONFIRM FROM THE DOCS")
    for field in UNCONFIRMED:
        if field in rep["resolved"]:
            pct = cov.get(field, 0.0) * 100
            verdict = "YES" if pct > 0 else "NAMED BUT EMPTY"
            print(f"  {field:<18} {verdict:<16} {pct:5.1f}% populated")
        else:
            print(f"  {field:<18} {'NO':<16} not in the file")


def seasons_block(g: pd.DataFrame) -> None:
    sub("SEASONS AND COMPLETENESS")
    by = g.groupby("season").agg(
        games=("game_id", "size"),
        played=("completed", "sum"),
        spread=("spread_line", lambda s: s.notna().mean()),
        total=("total_line", lambda s: s.notna().mean()),
    )
    recent = by.tail(20)
    print("  season  games  played  spread%  total%")
    for season, row in recent.iterrows():
        print(f"  {int(season):<7} {int(row['games']):<6} "
              f"{int(row['played']):<7} {row['spread'] * 100:6.1f}% "
              f"{row['total'] * 100:6.1f}%")
    print(f"\n  full range: {int(g['season'].min())}-{int(g['season'].max())}")
    window = g[(g["season"] >= config.TRAIN_START_YEAR)
               & (g["season"] <= config.TRAIN_END_YEAR)]
    usable = window[window["completed"] & window["market_margin"].notna()]
    print(f"  training window {config.TRAIN_START_YEAR}-{config.TRAIN_END_YEAR}: "
          f"{len(window):,} games, {len(usable):,} complete with a market line")


def orientation_block(g: pd.DataFrame) -> None:
    """The one assumption everything downstream rests on."""
    head("SPREAD ORIENTATION - THE ASSUMPTION UNDER TEST")
    print("The code assumes spread_line is a HOME-perspective margin: positive")
    print("means the home side is favoured by that many points. The sportsbook")
    print("convention is the opposite sign. Getting this backwards would not")
    print("crash anything - it would just quietly invert every prediction.\n")

    d = g[g["completed"] & g["spread_line"].notna()].copy()
    if d.empty:
        print("  no completed games with a spread - cannot test")
        return

    d["margin"] = d["home_score"] - d["away_score"]
    corr = float(np.corrcoef(d["spread_line"], d["margin"])[0, 1])
    as_is = float((d["spread_line"] - d["margin"]).abs().mean())
    flipped = float((-d["spread_line"] - d["margin"]).abs().mean())

    # Favourites should win more often than not, whichever way we read it.
    fav_home = d[d["spread_line"] > 3]
    fav_home_wins = float((fav_home["margin"] > 0).mean()) if len(fav_home) else float("nan")

    print(f"  games tested                        {len(d):,}")
    print(f"  corr(spread_line, home margin)      {corr:+.3f}")
    print(f"  mean |spread - margin| as written   {as_is:.2f} pts")
    print(f"  mean |spread - margin| if flipped   {flipped:.2f} pts")
    print(f"  home team wins when spread_line>3   {fav_home_wins * 100:.1f}% "
          f"of {len(fav_home):,}")

    print()
    if corr > 0.3 and as_is < flipped:
        print("  VERDICT: CONFIRMED. Positive spread_line goes with the home")
        print("  team winning. The code is correct as written; no flip needed.")
    elif corr < -0.3 and flipped < as_is:
        print("  VERDICT: INVERTED. spread_line runs the other way. Tell me and")
        print("  I will change one line in nflverse.py:")
        print("      g['market_margin'] = -g['spread_line']")
    else:
        print("  VERDICT: UNCLEAR - the relationship is too weak to call.")
        print("  Send me this block and I will look at it directly.")


def market_block(g: pd.DataFrame) -> None:
    """How good the closing line is, which is the bar the model must clear."""
    d = g[g["completed"] & g["market_margin"].notna()]
    if d.empty:
        return
    sub("THE BAR: HOW ACCURATE THE CLOSING LINE IS")
    err = (d["market_margin"] - (d["home_score"] - d["away_score"]))
    print(f"  market margin MAE   {err.abs().mean():.2f} pts   "
          f"(over {len(d):,} games)")
    print(f"  market margin bias  {err.mean():+.2f} pts")
    t = d[d["total_line"].notna()]
    if len(t):
        terr = t["total_line"] - (t["home_score"] + t["away_score"])
        print(f"  market total MAE    {terr.abs().mean():.2f} pts   "
              f"(over {len(t):,} games)")
    print("\n  this is the number any model has to beat, and mostly will not.")
    print("  the college build settled at roughly market MAE + 1.5 pts.")


def hfa_block(g: pd.DataFrame) -> None:
    sub("HOME FIELD, MEASURED")
    d = g[g["completed"]]
    by_era = d.groupby(d["season"] // 5 * 5)["margin"].mean()
    print(f"  overall home margin  {d['margin'].mean():+.2f} pts")
    for era, val in by_era.items():
        print(f"    {int(era)}-{int(era) + 4:<6} {val:+.2f}")
    print(f"\n  config.HFA_PRIOR is currently {config.HFA_PRIOR}")


def pbp_block(season: int) -> None:
    head(f"PLAY-BY-PLAY PROBE - {season}")
    try:
        pbp = nflverse.load_pbp(season)
    except nflverse.DataUnavailable as exc:
        print(f"  could not fetch: {exc}")
        print("  if this is the parquet failing, the csv.gz fallback should")
        print("  have caught it - send me the log line above.")
        return

    print(f"  rows: {len(pbp):,}   columns: {len(pbp.columns)}")
    mapping_block(pbp, schema.PBP_FIELDS, schema.REQUIRED_PBP_FIELDS,
                  "PLAY-BY-PLAY")

    eff = nflverse.team_game_efficiency(pbp)
    sub("EFFICIENCY AGGREGATION")
    if eff.empty:
        print("  produced nothing - the play filter or the column names are off")
        return
    print(f"  {len(eff):,} team-games from {eff['game_id'].nunique():,} games")
    print(f"  expected roughly 2 rows per game: "
          f"{len(eff) / max(eff['game_id'].nunique(), 1):.2f}\n")
    num = eff.select_dtypes("number").drop(columns=["season", "week"],
                                           errors="ignore")
    desc = num.describe().T[["mean", "std", "min", "max"]]
    print(desc.round(3).to_string())
    print("\n  sanity: epa_per_play should sit near 0.0 with sd around 0.2,")
    print("  success_rate near 0.45, plays per team-game near 60.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbp", type=int, default=None, metavar="SEASON",
                    help="also probe one season of play-by-play")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s")

    head("NFLVERSE DATA PROBE")
    print("Fetching the live games file. No key, no account, no rate limit.")

    try:
        raw = nflverse.load_games_raw()
    except nflverse.DataUnavailable as exc:
        print(f"\nFAILED: {exc}")
        print("\nEvery source for the games file refused. If this is running on")
        print("GitHub Actions, that is unexpected - send me the log.")
        return 1

    columns_block(raw)
    rep = mapping_block(raw, schema.GAME_FIELDS, schema.REQUIRED_GAME_FIELDS,
                        "GAMES")
    unconfirmed_block(rep, schema.coverage(raw, schema.GAME_FIELDS))

    head("AFTER NORMALISING")
    g = nflverse.load_games()
    print(f"  {len(g):,} rows survived typing and cleaning")
    seasons_block(g)
    hfa_block(g)
    market_block(g)
    orientation_block(g)

    if args.pbp:
        pbp_block(args.pbp)

    head("WHAT TO SEND BACK")
    print("The whole of this output. The three things that decide what gets")
    print("built next are: the coverage table, the unconfirmed-fields block,")
    print("and the spread orientation verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
