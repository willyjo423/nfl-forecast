"""Bootstrap: fetch everything, measure everything, train the model.

Run this once to create the model, and again whenever the feature code changes.
It deliberately does the measuring *before* the training, and prints what each
feature group is worth, because a number that nobody looked at is not evidence.

    python build.py                 # full run
    python build.py --no-ablation   # skip the group-by-group measurement
    python build.py --seasons 2015 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

import config
import dataset
import model as model_mod
import storage
from efficiency import EfficiencyEngine
from features import FEATURE_GROUPS, build_features, training_matrix
from qb import QBEngine, build_starter_table
from ratings import PreseasonPriors, RatingsEngine

log = logging.getLogger("build")

MODEL_PATH = config.MODELS / "nfl_model.joblib"
FEATURES_PATH = config.DATA / "features"
METRICS_PATH = config.DATA / "metrics.json"


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def assemble(seasons: list[int]) -> pd.DataFrame:
    """Everything from raw files to a feature matrix."""
    rule("DATA")
    games = dataset.build_games(seasons=seasons)

    eff_raw = dataset.load_efficiency_for(seasons)
    efficiency = EfficiencyEngine(eff_raw) if not eff_raw.empty else None

    rule("QUARTERBACKS")
    starters = build_starter_table(games, eff_raw)
    quarterbacks = QBEngine(starters)
    cov = quarterbacks.coverage()
    print(f"  team-games            {cov['rows']:,}")
    print(f"  starter identified    {cov.get('identified', 0) * 100:.1f}%")
    print(f"  with passing EPA      {cov.get('with_pass_epa', 0) * 100:.1f}%")
    print(f"  distinct quarterbacks {cov.get('distinct_qbs', 0):,}")
    if cov.get("identified", 0) < 0.9:
        print("  NOTE: thin starter coverage - the quarterback features will "
              "be mostly missing and should measure as a null.")

    rule("RATINGS")
    priors = PreseasonPriors()
    engine = RatingsEngine(games, priors)
    engine.build_priors(seasons)
    print(f"  priors chained across {len(seasons)} seasons "
          f"(carryover {config.YEAR_CARRYOVER})")

    rule("FEATURES")
    t0 = time.time()
    feat = build_features(games, engine, efficiency, quarterbacks)
    import dfs_link
    feat = dfs_link.attach(feat)
    print(f"  NFL-DFS team projections on {feat['home_dfs_core'].notna().mean() * 100:.0f}% of games")
    print(f"  {len(feat):,} games x {len(feat.columns)} columns "
          f"in {time.time() - t0:.0f}s")

    X, y = training_matrix(feat)
    print(f"  usable for training: {len(X):,}")
    missing = X.isna().mean().sort_values(ascending=False)
    worst = missing[missing > 0.2]
    if len(worst):
        print("\n  features missing on more than 20% of rows:")
        for name, share in worst.items():
            print(f"    {name:<26} {share * 100:5.1f}%")
        print("  (the booster handles these natively; listed so a genuine")
        print("   plumbing failure is not mistaken for a sparse feature)")
    return feat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    lo, hi = args.seasons or (config.TRAIN_START_YEAR, config.TRAIN_END_YEAR)
    seasons = list(range(lo, hi + 1))

    feat = assemble(seasons)
    storage.save(feat, FEATURES_PATH)

    if not args.no_ablation:
        rule("WHAT EACH FEATURE GROUP IS WORTH")
        print("Walk-forward MAE. Each group is measured ON ITS OWN against the")
        print("ratings baseline, paired game by game, with the t statistic")
        print("beside it. |t| under 2 means the difference cannot be told from")
        print("zero and the group is not earning its place.\n")
        abl = model_mod.ablation(feat, FEATURE_GROUPS)
        if abl:
            print(f"  {'group':<16} {'cols':>5} {'MAE':>7} {'delta':>8} "
                  f"{'t':>7}   verdict")
            for name, r in abl.items():
                t = r.get("t")
                ts = "     -" if t is None or pd.isna(t) else f"{t:+7.2f}"
                print(f"  {name:<16} {r['columns']:>5} {r['margin_mae']:>7.3f} "
                      f"{r['delta']:>+8.3f} {ts}   {r.get('verdict', '')}")

        rule("WHERE THE QUARTERBACK LAYER ACTS")
        print("A group can be worth little on average and a great deal on the")
        print("games it is actually about. A backup starting is that shape.\n")
        subs = model_mod.subgroup_report(
            feat,
            cols=FEATURE_GROUPS["strength"] + FEATURE_GROUPS["quarterback"],
            base_cols=FEATURE_GROUPS["strength"],
            subsets={
                "one side on a new starter":
                    (feat["home_qb_new"] == 1) | (feat["away_qb_new"] == 1),
                "both sides settled":
                    (feat["home_qb_new"] == 0) & (feat["away_qb_new"] == 0),
                "a starter under 5 career starts":
                    (feat["home_qb_starts"] < 5) | (feat["away_qb_starts"] < 5),
                "weeks 1-4": feat["week"] <= 4,
                "weeks 5 and later": feat["week"] > 4,
            })
        for label, r in subs.items():
            if "note" in r:
                print(f"  {label:<34} {r['note']}")
                continue
            print(f"  {label:<34} n={r['n']:>5}  {r['delta']:>+7.3f} pts  "
                  f"(t = {r['t']:+.2f})")

    rule("DOES THE NFL-DFS PLAYER MODEL ADD TO THE FORECAST?")
    from features import FEATURE_COLUMNS
    from dfs_link import DFS_COLUMNS
    if feat["home_dfs_core"].notna().mean() > 0.3:
        has = feat["home_dfs_core"].notna() & feat["away_dfs_core"].notna()
        newqb = (feat["home_qb_new"] == 1) | (feat["away_qb_new"] == 1)
        big = (feat["home_dfs_change"].abs() >= 8) | (feat["away_dfs_change"].abs() >= 8)
        dfs_report = model_mod.subgroup_report(
            feat, cols=FEATURE_COLUMNS,
            base_cols=[c for c in FEATURE_COLUMNS if c not in DFS_COLUMNS],
            subsets={"all games with DFS data": has,
                     "a new starting QB": has & newqb,
                     "a team's core down/up 8+ DK pts": has & big,
                     "settled games": has & ~newqb & ~big})
        print("Model with vs without the DFS team projections, paired (for the record):")
        for label, r in dfs_report.items():
            if "delta" not in r:
                print(f"  {label:<34} n={r.get('n', 0):>5}  too few games")
                continue
            print(f"  {label:<34} n={r['n']:>5}  {r['delta']:>+7.3f} pts  (t = {r['t']:+.2f})")
    else:
        dfs_report = {}
        print("  team projections not available yet - skipped")

    rule("WALK-FORWARD EVALUATION")
    oos = model_mod.walk_forward(feat)
    metrics = model_mod.evaluate(oos)
    print(model_mod.summarize(metrics))

    rule("MODEL vs MARKET: DOES A LINE+MODEL BLEND BEAT THE LINE?")
    import market
    Xm, _ = training_matrix(feat)
    metrics["market_blend"] = market.market_blend(oos, Xm)
    metrics["dfs_link"] = dfs_report
    print(market.summarize(metrics["market_blend"]))
    keep = [c for c in ("season_tested", "week", "margin", "total", "pred_margin", "pred_total",
                        "market_margin", "total_line") if c in oos.columns]
    oos[keep].join(Xm[[c for c in ("home_qb_new", "away_qb_new") if c in Xm.columns]]) \
        .to_csv(config.DATA / "oos.csv")
    model_mod.save_metrics(metrics, METRICS_PATH)

    rule("FITTING THE FINAL MODEL")
    X, y = training_matrix(feat)
    final = model_mod.NFLModel().fit(X, y)
    storage.save_model(final, MODEL_PATH)
    print(f"  trained on {len(X):,} games, "
          f"{min(final.trained_seasons)}-{max(final.trained_seasons)}")
    print(f"  saved to {MODEL_PATH.name}")

    rule("HONEST SUMMARY")
    if metrics.get("mae_vs_market") is not None:
        d = metrics["mae_vs_market"]
        if d > 0:
            print(f"The model is {d:.2f} points worse than the closing line.")
            print("That is the expected result and not a failure: the NFL")
            print("closing number is the most efficient price in sport. Use")
            print("this to forecast games, not to bet against that number.")
        else:
            print(f"The model is {abs(d):.2f} points better than the closing")
            print("line out of sample. Treat that with suspicion until it")
            print("survives a season of forward-only results in track.py -")
            print("beating this number is rare and usually a bug.")
    print(json.dumps({k: v for k, v in metrics.items()
                      if isinstance(v, (int, float))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
