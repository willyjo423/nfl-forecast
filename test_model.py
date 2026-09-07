"""Offline verification of the modelling layers.

No network. Everything runs against `fixtures.py`, whose synthetic world has
planted ground truth, so these checks can assert that a layer *recovers a
relationship known to exist* rather than merely that it ran.

The most important checks here are the leakage ones. A model that has seen the
future produces beautiful numbers and is worthless, and the failure is silent -
so the ratings engine is asked for the same answer twice, once with future
weeks present in the table and once with them deleted, and the two must be
bit-identical.

    python test_model.py
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import config
import fixtures
import nflverse

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


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * 66}")


def load_offline(raw: pd.DataFrame) -> pd.DataFrame:
    import dataset
    original = nflverse.load_games_raw
    nflverse.load_games_raw = lambda: raw
    try:
        return dataset.build_games()
    finally:
        nflverse.load_games_raw = original


# ------------------------------------------------------------------ dataset
def test_dataset() -> None:
    section("SITUATIONAL COLUMNS")
    g = load_offline(fixtures.make_games())

    for col in ("neutral_site", "is_dome", "is_turf", "home_short_week",
                "rest_diff", "home_game_no", "temp_f", "wind_mph"):
        check(f"{col} derived", col in g.columns)

    check("dome flag is binary", set(g["is_dome"].dropna().unique()) <= {0.0, 1.0})
    check("some games are indoors", g["is_dome"].sum() > 0)
    check("short weeks detected", g["home_short_week"].sum() > 0)
    check("indoor games get a filled temperature",
          g.loc[g["is_dome"] == 1, "temp_f"].notna().all())
    outdoor_unplayed = g[(g["is_dome"] == 0) & (~g["completed"])]
    if len(outdoor_unplayed):
        check("unplayed outdoor games keep a missing temperature",
              outdoor_unplayed["temp_f"].isna().all(),
              "a filled default here would be a confident lie")

    per_team = g.groupby(["season", "home_team"])["home_game_no"].max()
    check("game numbers advance", per_team.max() > 5, f"{per_team.max()}")


# ------------------------------------------------------------------ ratings
def test_ratings() -> None:
    section("RATINGS")
    from ratings import PreseasonPriors, RatingsEngine

    g = load_offline(fixtures.make_games())
    seasons = sorted(g["season"].unique())
    engine = RatingsEngine(g, PreseasonPriors())
    engine.build_priors(seasons)

    late = engine.ratings_before(seasons[-1], 12)
    check("late-season table is populated", len(late) >= 30, f"{len(late)}")
    check("ratings are centred", abs(late["rating"].mean()) < 1.0,
          f"{late['rating'].mean():.3f}")
    check("ratings have real spread", late["rating"].std() > 1.5,
          f"sd={late['rating'].std():.2f}")
    check("home field is positive and sane",
          0.0 < late.attrs["hfa"] < 6.0, f"{late.attrs['hfa']:.2f}")

    # Does the rating recover the strength that generated the fixtures?
    truth = fixtures.team_strengths(np.random.default_rng(1729), seasons[-1])
    joined = late.set_index("team")["rating"]
    common = [t for t in joined.index if t in truth]
    corr = float(np.corrcoef(joined.loc[common],
                             [truth[t] for t in common])[0, 1])
    check("ratings track the planted strengths", corr > 0.5, f"corr={corr:+.3f}")

    # Week 1 must not be a flat zero for everyone.
    wk1 = [engine.lookup(seasons[-1], 1, t)["rating"] for t in truth]
    check("week 1 uses the preseason prior", float(np.std(wk1)) > 0.5,
          f"sd={np.std(wk1):.3f} - if this is 0 every opener is a pick'em")

    # --- the leakage test ---
    target_season, target_week = seasons[-1], 8
    full = engine.ratings_before(target_season, target_week)

    truncated = g[~((g["season"] == target_season) & (g["week"] >= target_week))]
    e2 = RatingsEngine(truncated, PreseasonPriors())
    e2.build_priors(seasons)
    partial = e2.ratings_before(target_season, target_week)

    same = (len(full) == len(partial)
            and np.allclose(full["rating"].to_numpy(),
                            partial["rating"].to_numpy(), atol=1e-9))
    check("deleting future weeks changes nothing", same,
          "the ratings are seeing games they should not")


# --------------------------------------------------------------- efficiency
def test_efficiency() -> None:
    section("EFFICIENCY ENGINE")
    from efficiency import EfficiencyEngine

    games = fixtures.make_games(seasons=(2023, 2024), weeks=10)
    eff_raw = nflverse.team_game_efficiency(fixtures.make_pbp(games))
    engine = EfficiencyEngine(eff_raw)

    check("engine reports itself available", engine.available)
    table = engine.stats_before(2024, 8)
    check("solve produces a table", not table.empty)
    check("every team is rated", len(table) >= 30, f"{len(table)}")
    check("offence is centred",
          abs(table["off_epa_per_play"].mean()) < 0.05,
          f"{table['off_epa_per_play'].mean():+.4f}")

    # Week 1 of a later season must still work, off the prior season alone.
    wk1 = engine.stats_before(2024, 1)
    check("week 1 falls back to last season", not wk1.empty,
          "with no carryover, every September game starts blind")

    # Leakage: the window must exclude the target week.
    ages = [engine.lookup(2024, 5, t)["eff_games"] for t in ["KC", "SF", "BUF"]]
    check("evidence accumulates", max(ages) > 0, f"{ages}")

    thin = engine.stats_before(2023, 1)
    check("the very first week has nothing to say", thin.empty,
          "there is no prior season in this fixture, so this must be empty")


# ------------------------------------------------------------- quarterbacks
def test_quarterbacks() -> None:
    section("QUARTERBACKS")
    from qb import QBEngine, build_starter_table

    raw, truth = fixtures.make_games_with_qbs()
    g = load_offline(raw)
    eff = nflverse.team_game_efficiency(fixtures.make_pbp(g))

    starters = build_starter_table(g, eff)
    check("two rows per game", len(starters) == 2 * len(g))
    check("starters identified", starters["qb_id"].notna().mean() > 0.99,
          f"{starters['qb_id'].notna().mean():.3f}")

    engine = QBEngine(starters)
    seasons = sorted(g["season"].unique())

    # Continuity: a backup's first start must be flagged.
    flags = []
    for row in g.itertuples(index=False):
        if row.season != seasons[-1]:
            continue
        got = engine.lookup(row.season, row.week, row.home_team, row.home_qb_id)
        flags.append((row.home_qb_id.endswith("QB2"), got["qb_new_starter"]))
    changes = [f for is_backup, f in flags if is_backup and f == 1.0]
    check("backup starts are flagged as new", len(changes) > 5,
          f"only {len(changes)} flagged across a season")

    starters_flagged = [f for is_backup, f in flags
                        if not is_backup and f == 1.0]
    check("settled starters are mostly not flagged",
          len(starters_flagged) < 0.35 * len(flags),
          f"{len(starters_flagged)} of {len(flags)} - a flag on every game "
          f"is not a signal")

    # Value: with enough starts, the good quarterback should rate above the bad.
    late = seasons[-1]
    gaps = []
    for team in fixtures.TEAMS[:12]:
        v1 = engine.lookup(late, 16, team, f"{team}_QB1")["qb_value"]
        v2 = engine.lookup(late, 16, team, f"{team}_QB2")["qb_value"]
        if np.isfinite(v1) and np.isfinite(v2):
            gaps.append(v1 - v2)
    check("starters rate above their backups",
          len(gaps) > 5 and float(np.mean(gaps)) > 0,
          f"mean gap {np.mean(gaps) if gaps else float('nan'):+.4f} "
          f"over {len(gaps)} teams")

    # An unknown quarterback must produce honest ignorance, not a zero.
    unknown = engine.lookup(late, 10, "KC", None)
    check("an unnamed starter yields NaN, not 0",
          np.isnan(unknown["qb_value"]) and unknown["qb_known"] == 0)

    # Leakage: value at week W must not move when later weeks are deleted.
    trimmed = starters[~((starters["season"] == late) & (starters["week"] >= 9))]
    e2 = QBEngine(trimmed)
    a = engine.lookup(late, 9, "KC", "KC_QB1")["qb_value"]
    b = e2.lookup(late, 9, "KC", "KC_QB1")["qb_value"]
    check("deleting future weeks changes nothing",
          (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12,
          f"{a} vs {b}")


# ------------------------------------------------------------------ features
def build_everything(seasons=(2022, 2023, 2024)):
    from efficiency import EfficiencyEngine
    from features import build_features
    from qb import QBEngine, build_starter_table
    from ratings import PreseasonPriors, RatingsEngine

    raw, _ = fixtures.make_games_with_qbs(seasons=seasons)
    g = load_offline(raw)
    eff_raw = nflverse.team_game_efficiency(fixtures.make_pbp(g))

    seasons = sorted(g["season"].unique())
    engine = RatingsEngine(g, PreseasonPriors())
    engine.build_priors(seasons)
    efficiency = EfficiencyEngine(eff_raw)
    quarterbacks = QBEngine(build_starter_table(g, eff_raw))
    return build_features(g, engine, efficiency, quarterbacks)


def test_features() -> None:
    section("FEATURE MATRIX")
    from features import FEATURE_COLUMNS, MARKET_COLUMNS, training_matrix

    feat = build_everything()
    check("every declared feature is present",
          all(c in feat.columns for c in FEATURE_COLUMNS),
          str([c for c in FEATURE_COLUMNS if c not in feat.columns]))
    check("all features are numeric",
          all(pd.api.types.is_numeric_dtype(feat[c]) for c in FEATURE_COLUMNS))

    # The rule that keeps model-vs-market meaningful.
    for col in MARKET_COLUMNS:
        check(f"{col} is not a feature", col not in FEATURE_COLUMNS,
              "training on the line would make the comparison circular")

    X, y = training_matrix(feat)
    check("training matrix has rows", len(X) > 500, f"{len(X)}")
    check("targets are present", set(["margin", "total", "home_win"]) <= set(y.columns))
    check("no target leaked into X",
          not ({"margin", "total", "home_win"} & set(X.columns)))

    filled = X.notna().mean()
    check("core features are populated",
          filled["proj_margin"] > 0.99 and filled["rating_diff"] > 0.99,
          f"proj_margin {filled['proj_margin']:.3f}")
    check("efficiency features are mostly populated",
          filled["edge_epa_per_play_home"] > 0.7,
          f"{filled['edge_epa_per_play_home']:.3f}")

    check("proj_margin correlates with the outcome",
          float(feat[["proj_margin", "margin"]].corr().iloc[0, 1]) > 0.3,
          f"{feat[['proj_margin', 'margin']].corr().iloc[0, 1]:+.3f}")


# ------------------------------------------------------------------- model
def test_model() -> None:
    section("MODEL")
    import model as model_mod
    from features import training_matrix

    # Six seasons, not three: the scale calibration and the win-probability
    # curve are both fitted on held-out seasons, and with three there is
    # nothing to hold out. Testing the model on a history too short to trigger
    # its own calibration would leave that code entirely unexercised.
    feat = build_everything(seasons=tuple(range(2019, 2025)))
    X, y = training_matrix(feat)
    m = model_mod.NFLModel().fit(X, y)
    preds = m.predict(X)

    check("scale calibration was fitted",
          m.margin_calibration != (1.0, 0.0),
          "with six seasons there is enough history; (1.0, 0.0) means the "
          "held-out pass never ran")
    check("scale calibration stayed sane",
          0.55 <= m.margin_calibration[0] <= 1.45,
          str(m.margin_calibration))

    check("predictions have the right shape", len(preds) == len(X))
    check("totals are plausible", 30 < preds["pred_total"].mean() < 60,
          f"{preds['pred_total'].mean():.1f}")

    # Win probability must be monotonic and must not flatten out.
    grid = X.head(1).copy()
    probs = []
    for margin in (-28, -14, -3, 0, 3, 14, 28):
        row = grid.copy()
        row["proj_margin"] = margin
        probs.append(float(m.predict(row)["home_win_prob"].iloc[0]))
    check("win probability rises with margin",
          all(b >= a - 1e-9 for a, b in zip(probs, probs[1:])),
          str([round(p, 3) for p in probs]))
    check("a big favourite is a big favourite", probs[-1] > 0.80,
          f"28-point favourite priced at {probs[-1]:.3f}")
    check("a pick'em is near a coin flip", 0.35 < probs[3] < 0.75,
          f"{probs[3]:.3f}")

    # Uncertainty must never grow with evidence.
    if m.sigma_by_played:
        sigmas = [c["sigma"] for c in m.sigma_by_played]
        check("uncertainty is non-increasing in evidence",
              all(a >= b - 1e-9 for a, b in zip(sigmas, sigmas[1:])),
              str([round(s, 2) for s in sigmas]))

    # A stale model must say what to do about it, not raise a bare KeyError.
    try:
        m.check_compatible(X.drop(columns=["proj_total"]))
        check("stale feature set is caught", False, "no error raised")
    except model_mod.FeatureMismatchError as exc:
        check("stale feature set is caught with an actionable message",
              "retrain" in str(exc).lower() or "bootstrap" in str(exc).lower())

    section("WALK-FORWARD")
    oos = model_mod.walk_forward(feat, min_train_seasons=4)
    check("walk-forward produced rows", len(oos) > 100, f"{len(oos)}")
    metrics = model_mod.evaluate(oos)
    check("margin MAE is finite", np.isfinite(metrics.get("margin_mae", np.nan)))
    check("market comparison computed", "market_margin_mae" in metrics)

    # The crushed-margin test, done properly.
    #
    # The obvious version - "predicted spread should be close to actual spread"
    # - is wrong, and I had it wrong first time round. A good forecast is
    # SUPPOSED to be less variable than reality: its standard deviation should
    # be about its own correlation times the real one, because the part it
    # cannot predict is noise and pretending otherwise just adds error.
    #
    # What genuinely diagnoses the college build's 3.28x compression bug is the
    # regression of outcome on forecast, out of sample. If the model says 7 and
    # such games really finish at 7 on average, the slope is 1. A slope well
    # above 1 means every prediction is scaled too small and needs multiplying
    # up - which is exactly what that bug looked like.
    ok = np.isfinite(oos["pred_margin"]) & np.isfinite(oos["margin"])
    slope = float(np.polyfit(oos.loc[ok, "pred_margin"],
                             oos.loc[ok, "margin"], 1)[0])
    check("forecasts are on the right scale", 0.6 < slope < 1.5,
          f"outcome-on-forecast slope {slope:.2f}; 1.0 is calibrated, "
          f"well above 1 means the predictions are systematically too small")

    ratio = float(oos["pred_margin"].std() / oos["margin"].std())
    corr = float(oos.loc[ok, ["pred_margin", "margin"]].corr().iloc[0, 1])
    check("spread of forecasts matches their own skill",
          abs(ratio - corr) < 0.15,
          f"forecast sd is {ratio:.2f} of reality's while correlation is "
          f"{corr:.2f}; these should roughly agree")

    print(f"\n  (fixture MAE {metrics['margin_mae']:.2f} vs market "
          f"{metrics.get('market_margin_mae', float('nan')):.2f}, "
          f"calibration slope {slope:.2f} - these are synthetic numbers and "
          f"mean nothing about real football)")


def test_ablation() -> None:
    """Does the measurement harness actually measure anything?

    The whole discipline of this project rests on feature groups earning their
    place through walk-forward MAE. That is only worth anything if the harness
    can tell a real effect from a useless one - so it is shown both. The same
    fixture is built twice, once where the starting quarterback is worth six
    points and once where every quarterback is identical, and the quarterback
    features must help in the first case and not in the second.

    Without the second half, "the harness said it helps" would be indis-
    tinguishable from "adding columns always helps a bit".
    """
    section("ABLATION HARNESS")
    import model as model_mod
    from features import FEATURE_GROUPS

    seasons = tuple(range(2019, 2025))

    def qb_delta(effect: float) -> float:
        from efficiency import EfficiencyEngine
        from features import build_features
        from qb import QBEngine, build_starter_table
        from ratings import PreseasonPriors, RatingsEngine

        raw, _ = fixtures.make_games_with_qbs(seasons=seasons, qb_effect=effect)
        g = load_offline(raw)
        eff_raw = nflverse.team_game_efficiency(fixtures.make_pbp(g))
        engine = RatingsEngine(g, PreseasonPriors())
        engine.build_priors(sorted(g["season"].unique()))
        feat = build_features(g, engine, EfficiencyEngine(eff_raw),
                              QBEngine(build_starter_table(g, eff_raw)))
        abl = model_mod.ablation(feat, FEATURE_GROUPS)
        return abl.get("quarterback", {}).get("delta", float("nan"))

    real = qb_delta(8.0)
    check("a planted quarterback effect is detected", real < -0.05,
          f"delta {real:+.3f} - the harness cannot see an 8-point effect")

    none = qb_delta(0.0)
    check("an absent effect is not invented", none > -0.05,
          f"delta {none:+.3f} with identical quarterbacks - the harness is "
          f"rewarding extra columns rather than measuring skill")

    print(f"\n  (quarterbacks worth 8 pts: {real:+.3f} MAE; "
          f"worth nothing: {none:+.3f})")


def main() -> int:
    print("NFL modelling layers - offline checks")
    for fn in (test_dataset, test_ratings, test_efficiency, test_quarterbacks,
               test_features, test_model, test_ablation):
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
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
