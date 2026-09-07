"""Offline verification of the NFL data layer.

Runs with no network. Every check either passes or prints exactly what it
expected and what it got, so a failure is a diagnosis rather than a hint.

    python test_data.py

The checks that matter most are the ones about *meaning* rather than
mechanics: that the spread convention is what the code assumes, that garbage
plays are excluded rather than silently averaged in, and that a missing column
degrades to NA instead of raising.
"""
from __future__ import annotations

import gzip
import io
import sys
import traceback

import numpy as np
import pandas as pd

import fixtures
import nflverse
import schema

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def near(name: str, got: float, want: float, tol: float) -> None:
    check(name, abs(got - want) <= tol, f"got {got:.4f}, wanted {want:.4f} +/- {tol}")


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * 66}")


# ------------------------------------------------------------------ schema
def test_schema() -> None:
    section("SCHEMA MAPPING")
    g = fixtures.make_games()

    rep = schema.report(g, schema.GAME_FIELDS, schema.REQUIRED_GAME_FIELDS)
    check("no required game field missing", not rep["missing_required"],
          str(rep["missing_required"]))
    check("mapping resolves most fields", len(rep["resolved"]) >= 25,
          f"resolved {len(rep['resolved'])}")

    # A rename must not break anything - this is the failure mode that cost
    # time on the college build.
    renamed = fixtures.make_renamed_games()
    rep2 = schema.report(renamed, schema.GAME_FIELDS,
                         schema.REQUIRED_GAME_FIELDS)
    check("renamed columns still resolve", not rep2["missing_required"],
          str(rep2["missing_required"]))
    check("alias picks up game_date", "gameday" in rep2["resolved"])
    check("alias picks up temperature", "temp" in rep2["resolved"])
    check("alias picks up home_qb", "home_qb" in rep2["resolved"])

    # A missing column must become NA, not an exception.
    trimmed = g.drop(columns=["div_game", "home_moneyline", "referee"])
    norm = schema.normalise(trimmed, schema.GAME_FIELDS)
    check("missing column becomes a column", "div_game" in norm.columns)
    check("missing column is all NA", norm["div_game"].isna().all())
    check("missing column does not lose rows", len(norm) == len(trimmed))

    cov = schema.coverage(g, schema.GAME_FIELDS)
    check("coverage reports game_id fully present", cov["game_id"] == 1.0,
          f"{cov['game_id']}")
    check("coverage sees temp is partial", 0.5 < cov["temp"] < 1.0,
          f"{cov['temp']:.3f}")

    # Unmapped detection is how the next feature gets found.
    extra = g.copy()
    extra["some_new_nflverse_column"] = 1
    rep3 = schema.report(extra, schema.GAME_FIELDS,
                         schema.REQUIRED_GAME_FIELDS)
    check("unmapped columns are reported",
          "some_new_nflverse_column" in rep3["unmapped"])

    check("empty frame does not raise",
          list(schema.normalise(pd.DataFrame(), schema.GAME_FIELDS).columns)
          == list(schema.GAME_FIELDS))


# ------------------------------------------------------------------ parsing
def test_parsing() -> None:
    section("FILE PARSING")
    g = fixtures.make_games(seasons=(2024,), weeks=2)

    csv_bytes = g.to_csv(index=False).encode()
    parsed = nflverse._read_table(csv_bytes, "games.csv")
    check("plain csv parses", len(parsed) == len(g), f"{len(parsed)} vs {len(g)}")

    buf = io.BytesIO()
    with gzip.open(buf, "wb") as fh:
        fh.write(csv_bytes)
    parsed_gz = nflverse._read_table(buf.getvalue(), "pbp.csv.gz")
    check("gzipped csv parses", len(parsed_gz) == len(g))

    check("cache paths are stable",
          nflverse._cache_path("http://x/games.csv", ".bin")
          == nflverse._cache_path("http://x/games.csv", ".bin"))
    check("cache paths differ by url",
          nflverse._cache_path("http://x/games.csv", ".bin")
          != nflverse._cache_path("http://y/games.csv", ".bin"))


# ------------------------------------------------------------------ loading
def load_offline(**kwargs) -> pd.DataFrame:
    """load_games with the network replaced by fixtures."""
    raw = fixtures.make_games()
    original = nflverse.load_games_raw
    nflverse.load_games_raw = lambda: raw
    try:
        return nflverse.load_games(**kwargs)
    finally:
        nflverse.load_games_raw = original


def test_loading() -> None:
    section("GAME LOADING")
    g = load_offline()

    # 3 seasons x 17 weeks x 16 games.
    check("rows survive", len(g) == 816, f"{len(g)}")
    check("season is integer", g["season"].dtype.kind in "iu", str(g["season"].dtype))
    check("week is integer", g["week"].dtype.kind in "iu", str(g["week"].dtype))
    check("teams are upper case", (g["home_team"] == g["home_team"].str.upper()).all())
    check("kickoff parsed", g["kickoff"].notna().mean() > 0.99,
          f"{g['kickoff'].notna().mean():.3f}")
    check("sorted by time", g["season"].is_monotonic_increasing)

    played = g[g["completed"]]
    check("some games are complete", len(played) == 800, f"{len(played)}")
    check("some games are scheduled", (~g["completed"]).sum() > 0,
          f"{(~g['completed']).sum()}")
    check("margin agrees with scores",
          (played["margin"] == played["home_score"] - played["away_score"]).all())
    check("scheduled games have no margin", g.loc[~g["completed"], "margin"].isna().all())
    check("game_total agrees with scores",
          (played["game_total"] == played["home_score"] + played["away_score"]).all())

    filtered = load_offline(seasons=[2023])
    check("season filter works", set(filtered["season"]) == {2023},
          str(sorted(set(filtered['season']))))

    reg = load_offline(regular_only=True)
    check("regular-season filter keeps rows", len(reg) == len(g), f"{len(reg)}")


# ------------------------------------------------- the orientation assumption
def test_spread_orientation() -> None:
    section("SPREAD ORIENTATION  (the assumption everything rests on)")
    g = load_offline()
    d = g[g["completed"] & g["market_margin"].notna()]

    corr = float(np.corrcoef(d["market_margin"], d["margin"])[0, 1])
    check("market_margin correlates positively with home margin", corr > 0.3,
          f"corr={corr:+.3f}")

    as_is = float((d["market_margin"] - d["margin"]).abs().mean())
    flipped = float((-d["market_margin"] - d["margin"]).abs().mean())
    check("as written beats flipped", as_is < flipped,
          f"as-is {as_is:.2f} vs flipped {flipped:.2f}")

    big = d[d["market_margin"] > 6]
    rate = float((big["margin"] > 0).mean())
    check("big home favourites win most of the time", rate > 0.6,
          f"{rate:.3f} of {len(big)}")

    bias = float((d["market_margin"] - d["margin"]).mean())
    check("market is roughly unbiased", abs(bias) < 1.5, f"bias {bias:+.2f}")

    # And prove the test itself has teeth: an inverted file must fail it.
    inverted = fixtures.make_games()
    inverted["spread_line"] = -inverted["spread_line"]
    original = nflverse.load_games_raw
    nflverse.load_games_raw = lambda: inverted
    try:
        bad = nflverse.load_games()
    finally:
        nflverse.load_games_raw = original
    bad = bad[bad["completed"]]
    bad_corr = float(np.corrcoef(bad["market_margin"], bad["margin"])[0, 1])
    check("an inverted file would be caught", bad_corr < -0.3,
          f"corr={bad_corr:+.3f} - if this is not negative the check is useless")


# ------------------------------------------------------------- efficiency
def test_efficiency() -> None:
    section("EFFICIENCY AGGREGATION")
    games = fixtures.make_games(seasons=(2024,), weeks=6)
    pbp = fixtures.make_pbp(games)
    eff = nflverse.team_game_efficiency(pbp)

    played = games[games["home_score"].notna()]
    check("two rows per game", len(eff) == 2 * len(played),
          f"{len(eff)} rows for {len(played)} games")
    check("team and opponent columns present",
          {"team", "opponent"} <= set(eff.columns))
    check("no team faces itself", (eff["team"] != eff["opponent"]).all())

    # Special teams and kneels must be gone, or plays-per-game inflates. The
    # fixture generates ~62 scrimmage plays per team; the count that survives
    # is lower because garbage time is also excluded, which is the point -
    # so the assertion is that it fell somewhat, not that it fell to nothing.
    mean_plays = float(eff["plays"].mean())
    check("garbage time actually removed some plays", 45 < mean_plays < 62,
          f"{mean_plays:.1f} - expected below the ~62 generated")
    check("but not most of them", mean_plays > 0.8 * 62,
          f"{mean_plays:.1f} of ~62 survived")
    check("no special teams leaked",
          eff["plays"].max() < 90, f"max {eff['plays'].max()}")

    check("epa_per_play finite", np.isfinite(eff["epa_per_play"]).all())
    near("success rate plausible", float(eff["success_rate"].mean()), 0.5, 0.12)
    check("pass rate plausible", 0.4 < eff["pass_rate"].mean() < 0.75,
          f"{eff['pass_rate'].mean():.3f}")

    # The planted signal: winners should show better efficiency.
    merged = eff.merge(
        played[["game_id", "home_team", "home_score", "away_score"]],
        on="game_id")
    merged["own_margin"] = np.where(
        merged["team"] == merged["home_team"],
        merged["home_score"] - merged["away_score"],
        merged["away_score"] - merged["home_score"])
    corr = float(np.corrcoef(merged["epa_per_play"], merged["own_margin"])[0, 1])
    check("efficiency recovers the planted signal", corr > 0.2,
          f"corr={corr:+.3f}")

    # Passing and rushing EPA replaced the qb_epa column the first live probe
    # exposed as a duplicate of epa. They must be genuinely different numbers,
    # or the replacement achieved nothing.
    check("pass and rush EPA both produced",
          {"pass_epa", "rush_epa"} <= set(eff.columns))
    gap = float((eff["pass_epa"] - eff["rush_epa"]).abs().mean())
    check("pass and rush EPA are not the same column", gap > 0.01,
          f"mean absolute difference {gap:.4f}")
    check("garbage share is recorded",
          eff["garbage_share"].between(0, 1).all())

    check("empty input returns empty frame",
          nflverse.team_game_efficiency(pd.DataFrame()).empty)

    # A season missing an optional column must still aggregate.
    thin = pbp.drop(columns=["penalty", "down"])
    eff2 = nflverse.team_game_efficiency(thin)
    check("missing optional pbp column still aggregates", len(eff2) == len(eff),
          f"{len(eff2)} vs {len(eff)}")

    no_sacks = pbp.drop(columns=["sack"])
    eff3 = nflverse.team_game_efficiency(no_sacks)
    check("an absent metric becomes NaN, not zero",
          eff3["sack_rate"].isna().all(),
          "a zero here would claim nobody was sacked")


def main() -> int:
    print("NFL data layer - offline checks")
    for fn in (test_schema, test_parsing, test_loading,
               test_spread_orientation, test_efficiency):
        try:
            fn()
        except Exception:  # noqa: BLE001 - a crash is a failure, and reportable
            global FAIL
            FAIL += 1
            FAILURES.append(f"{fn.__name__} raised")
            print(f"  CRASH in {fn.__name__}")
            traceback.print_exc()

    print(f"\n{'=' * 66}")
    print(f"{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\nfailures:")
        for f in FAILURES:
            print(f"  - {f}")
    print("=" * 66)
    print("\nThese prove the code is self-consistent against a synthetic file.")
    print("They cannot prove the real nflverse columns are spelled the way")
    print("the fixtures spell them - only `python selftest.py` can, and it")
    print("prints the answer either way.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
