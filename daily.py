"""The zero-input daily run.

    python daily.py                 # the next slate
    python daily.py --week 3
    python daily.py --date 2026-09-13

It works out the season and the week on its own, pulls the schedule, rebuilds
ratings and quarterback state from everything played so far, fetches forecast
weather for the outdoor venues, and writes a JSON payload plus a standalone
HTML page. Nothing to submit.

Each run also archives its forecasts, so `track.py` can grade them later
against what actually happened. A model that is only ever evaluated on its own
backtest is a model nobody has checked.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
import dataset
import storage
import venues
from efficiency import EfficiencyEngine
from features import build_features
from model import FeatureMismatchError
from qb import QBEngine, build_starter_table
from ratings import PreseasonPriors, RatingsEngine
from weather import WeatherService, describe

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MODEL_PATH = config.MODELS / "nfl_model.joblib"


def current_season(today: date) -> int:
    """An NFL season is named for the year it starts in.

    So January and February belong to the previous season - the playoffs of the
    year before, not the opening weeks of the year they fall in.
    """
    return today.year if today.month >= 3 else today.year - 1


def pick_slate(games: pd.DataFrame, season: int, week: int | None,
               target: date | None) -> pd.DataFrame:
    """The games to forecast: an explicit week, an explicit date, or next up."""
    season_games = games[games["season"] == season]
    if season_games.empty:
        return season_games

    if week is not None:
        return season_games[season_games["week"] == week].copy()

    if target is not None:
        day = pd.to_datetime(season_games["gameday"], errors="coerce").dt.date
        return season_games[day == target].copy()

    # Default: the earliest week that still has an unplayed game. Mid-week that
    # is the rest of this week, not next week, which is what someone opening
    # the page on a Friday actually wants.
    pending = season_games[~season_games["completed"]]
    if pending.empty:
        last = int(season_games["week"].max())
        return season_games[season_games["week"] == last].copy()
    return season_games[season_games["week"] == int(pending["week"].min())].copy()


def run(week: int | None = None, target: date | None = None,
        with_weather: bool = True, weather_budget: float = 120.0) -> dict:
    today = datetime.now(ET).date()
    season = current_season(target or today)
    log.info("season %s", season)

    model = storage.load_model(MODEL_PATH)
    if model is None:
        raise SystemExit(
            f"No trained model at {MODEL_PATH}. Run the Bootstrap workflow "
            f"first - it trains the model and commits it to the repo.")

    # Ratings need this season and the one before it, for the preseason prior.
    games = dataset.build_games(seasons=[season - 1, season])
    if games.empty:
        raise SystemExit("No games returned.")

    slate = pick_slate(games, season, week, target)
    if slate.empty:
        return {"generated_at": datetime.now(ET).isoformat(), "season": season,
                "week": week, "games": [], "note": "No games scheduled."}
    wk = int(slate["week"].iloc[0])
    log.info("week %s: %d games", wk, len(slate))

    # Efficiency and quarterback history: this season plus last, so week 1 is
    # not blind. Play-by-play for the current season appears a day or two after
    # each slate, so an early-season run may legitimately find only last year's.
    eff = dataset.load_efficiency_for([season - 1, season])
    efficiency = EfficiencyEngine(eff) if not eff.empty else None
    quarterbacks = QBEngine(build_starter_table(games, eff))

    engine = RatingsEngine(games, PreseasonPriors())
    engine.build_priors([season - 1, season])

    if with_weather:
        wx = WeatherService(budget_seconds=weather_budget)
        wx.prefetch(slate)
        log.info("weather: %s", wx.coverage())
        for col in ("temp_f", "wind_mph"):
            slate[col] = slate[col].astype(float)
        for i, g in slate.iterrows():
            if g["completed"]:
                continue
            got = wx.at_kickoff(g["home_team"], g.get("stadium"), g.get("kickoff"),
                                bool(g.get("neutral_site", False)))
            slate.at[i, "temp_f"] = got["temp_f"]
            slate.at[i, "wind_mph"] = got["wind_mph"]

    feat = build_features(slate, engine, efficiency, quarterbacks)
    preds = model.predict(feat)
    out = feat.join(preds)

    records = [_record(r) for _, r in out.iterrows()]
    records.sort(key=lambda x: (x["kickoff_utc"] or "", x["home_team"]))

    metrics = {}
    if (config.DATA / "metrics.json").exists():
        metrics = json.loads((config.DATA / "metrics.json").read_text())

    return {
        "generated_at": datetime.now(ET).isoformat(),
        "season": season,
        "week": wk,
        "games": records,
        "model_metrics": metrics,
    }


def _j(v, digits=1):
    """NaN is not valid JSON, so an unavailable reading goes out as null."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), digits)


def _record(r) -> dict:
    margin = float(r["pred_margin"])
    total = float(r["pred_total"])
    sigma = float(r["margin_sigma"])

    # The middle half of the model's own predictive distribution. The college
    # version drew this from comparable historical games; here it comes from
    # the residual spread the model measured for this much evidence, which is
    # both more direct and the thing the sigma curve was built to express.
    p25, p75 = margin - 0.6745 * sigma, margin + 0.6745 * sigma

    wx = {"temp_f": _j(r.get("temp_f")), "wind_mph": _j(r.get("wind_mph"))}
    kickoff = r.get("kickoff")

    return {
        "game_id": None if pd.isna(r["game_id"]) else str(r["game_id"]),
        "kickoff_utc": None if pd.isna(kickoff) else pd.Timestamp(kickoff).isoformat(),
        "kickoff_et": (None if pd.isna(kickoff)
                       else pd.Timestamp(kickoff).strftime("%a %-I:%M %p")),
        "week": int(r["week"]),
        "home_team": r["home_team"], "away_team": r["away_team"],
        "neutral_site": bool(r["neutral_site"]),

        "forecast": {
            "home_points": round(float(r["pred_home_points"])),
            "away_points": round(float(r["pred_away_points"])),
            "margin": round(margin, 1),
            "total": round(total, 1),
            "home_win_prob": round(float(r["home_win_prob"]), 4),
            "sigma": round(sigma, 1),
            "margin_p25": round(p25, 1),
            "margin_p75": round(p75, 1),
        },

        "quarterbacks": {
            "home": None if pd.isna(r.get("home_qb")) else str(r.get("home_qb")),
            "away": None if pd.isna(r.get("away_qb")) else str(r.get("away_qb")),
            "home_new": _j(r.get("home_qb_new"), 0),
            "away_new": _j(r.get("away_qb_new"), 0),
            "home_starts": _j(r.get("home_qb_starts"), 0),
            "away_starts": _j(r.get("away_qb_starts"), 0),
        },

        "context": {
            "home_rest": _j(r.get("home_rest"), 0),
            "away_rest": _j(r.get("away_rest"), 0),
            "div_game": _j(r.get("div_game"), 0),
            "is_dome": _j(r.get("is_dome"), 0),
            "games_played": int(min(r["home_played"], r["away_played"])),
        },

        "weather": wx,
        "weather_text": describe(wx, _j(r.get("is_dome"), 0)),

        # Reference only. The bootstrap measured 50.4% against the spread over
        # 3,757 out-of-sample games, so the gap between these two numbers is
        # reported and nothing is dressed up as a play.
        "market_margin": _j(r.get("market_margin")),
        "market_total": _j(r.get("total_line")),
        "vs_market": (None if pd.isna(r.get("market_margin"))
                      else round(margin - float(r["market_margin"]), 1)),
    }


def archive(payload: dict) -> str | None:
    """Keep every forecast so it can be graded later."""
    if not payload.get("games"):
        return None
    stamp = datetime.now(ET).strftime("%Y%m%d-%H%M")
    path = config.FORECASTS / f"{payload['season']}-wk{payload['week']:02d}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Forecast the next NFL slate")
    p.add_argument("--week", type=int, help="explicit week number")
    p.add_argument("--date", help="YYYY-MM-DD; forecast that day's games")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--weather-budget", type=float, default=120.0)
    p.add_argument("--out", default=str(config.DOCS / "predictions.json"))
    p.add_argument("--html", default=str(config.DOCS / "index.html"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    target = (datetime.strptime(args.date, "%Y-%m-%d").date()
              if args.date else None)
    try:
        payload = run(week=args.week, target=target,
                      with_weather=not args.no_weather,
                      weather_budget=args.weather_budget)
    except FeatureMismatchError as exc:
        print(f"\n::error::{exc}\n")
        return 3

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    from dashboard import render
    with open(args.html, "w") as fh:
        fh.write(render(payload))

    kept = archive(payload)
    print(f"{len(payload['games'])} games forecast for week {payload.get('week')}")
    print(f"JSON -> {args.out}")
    print(f"HTML -> {args.html}")
    if kept:
        print(f"archived -> {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
