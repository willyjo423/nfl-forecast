"""Central configuration for the NFL build.

Constants that differ from the college version are marked, because most of the
differences are not cosmetic - they reflect a genuinely different sport. The
ones the first live probe settled are marked MEASURED, with the number it
reported, so nobody has to wonder later whether a value was reasoned or copied.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
FORECASTS = DATA / "forecasts"

for _p in (DATA, CACHE, MODELS, DOCS, FORECASTS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Sources ---------------------------------------------------------------
# nflverse needs no key and no account. Each resource lists fallbacks; the
# GitHub mirror and the author's own host go down independently.
GAMES_URLS = [
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "https://github.com/nflverse/nfldata/raw/master/data/games.csv",
    "http://www.habitatring.com/games.csv",
]

PBP_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"


def pbp_urls(season: int) -> list[str]:
    """Parquet first, gzipped CSV as the fallback when pyarrow is missing."""
    return [
        f"{PBP_RELEASE}/play_by_play_{season}.parquet",
        f"{PBP_RELEASE}/play_by_play_{season}.csv.gz",
    ]


# Live forecasts still need a weather service. MEASURED: temp and wind are
# populated for 69% of all rows - the gap is indoor games (where roof already
# says so) plus games not yet played, which is exactly the set a forecast is
# about. So the games file covers history and Open-Meteo covers the slate.
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# --- Training window -------------------------------------------------------
# MEASURED: the games file runs 1999-2026 with complete spreads and totals from
# 1999 onward, so the floor is a judgement about football rather than a data
# limit. 2006 keeps roughly 5,400 games - about 25% more than a 2010 floor -
# while staying inside the modern passing era. Widen or narrow it with the
# environment variable and let walk-forward MAE settle the argument.
TRAIN_START_YEAR = int(os.environ.get("TRAIN_START_YEAR", 2006))
TRAIN_END_YEAR = int(os.environ.get("TRAIN_END_YEAR", 2026))
REGULAR_SEASON_WEEKS = 18

# --- Ratings engine --------------------------------------------------------
# The right ridge strength is roughly (game noise variance) / (true team-quality
# variance). NFL margins scatter about 13 points around their expectation while
# real team quality spans only about 6, which puts the ratio near 5 - nothing
# like the 40 the college build uses, where the talent gap between the best and
# worst teams is enormous and a single result says much more.
#
# The first version of this file carried 18 on the reasoning that 32 teams
# playing 17 games is better determined than 133 playing 12. That confused how
# well-*connected* the schedule is with how *informative* a game is, and it is
# the second that sets the shrinkage. At 18 the projected margins came out with
# a standard deviation of 3.1 points against a real spread of 13.7 - every game
# looked like a coin flip.
#
# Swept on real results afterwards, final MAE across lambda 2 to 12 varied by
# only 0.06 points, with a shallow best near 8. The reason is that the trees
# ride on a *rescaled* ratings baseline, and the rescaling absorbs most of a
# shrinkage change before the model ever sees it. So this is worth getting
# roughly right and not worth tuning: the leverage is elsewhere.
RIDGE_LAMBDA_BASE = 5.0
# NFL margins are tighter, so the blowout cap comes down with them.
MARGIN_CAP = 21.0
# MEASURED: home margin averaged +1.87 over 2015-2019 and +1.90 over 2020-2024,
# against +3.24 in 1999 - home field has roughly halved this century. Those
# figures include playoffs, where the home side is the better seed by
# construction, so the regular-season prior sits a little under them.
HFA_PRIOR = 1.8
# Rosters are more stable than college, but the draft, the cap and a
# strength-of-schedule formula all pull hard toward the mean, and NFL
# year-over-year point-differential correlation is only about 0.5. The college
# value of 0.60 would have been optimistic here, and 0.72 plainly wrong.
#
# Swept on real results: 0.30 to 0.85 spans 0.045 points of MAE, marginally
# favouring the high end. Left at 0.55 because the difference is noise and the
# lower value is the one the sport's own regression to the mean supports.
YEAR_CARRYOVER = 0.55
# Half-life in weeks. Deliberately longer than the college value of 6: with 17
# games instead of 12 there is less data per team, and discarding September
# aggressively leaves November running on four games.
#
# The offline sweep prefers no decay at all, but that result is worthless - the
# synthetic teams have a fixed strength all season, so forgetting can only lose
# information. On real results the whole range from 4 weeks to no decay at all
# spans 0.04 points of MAE, which is nothing. Same story as the ridge strength,
# and the same conclusion: get it roughly right and stop.
RECENCY_HALFLIFE_WEEKS = 8.0

# --- Efficiency ------------------------------------------------------------
# Rates are noisier per game than scoring margin, so they shrink harder.
EFF_LAMBDA = 10.0
# MEASURED: after excluding garbage time, competitive plays per team-game
# averaged 52 but ran as low as 14. A 14-play team-game is a blowout whose
# rates are mostly noise, so the solve weights rows by play count and anything
# under this floor is dropped outright.
MIN_EFF_PLAYS = 20

# --- Quarterbacks ----------------------------------------------------------
# MEASURED: home_qb_id and away_qb_id are present on 96.4% of rows, and on
# essentially all of them inside the training window. This is the one thing the
# NFL data gives us that college never did.
# Starts needed before a quarterback's own record outweighs the league mean.
QB_SHRINKAGE_STARTS = 10.0
# Half-life in games for a quarterback's own history.
QB_HALFLIFE_GAMES = 24.0

# --- Model -----------------------------------------------------------------
RANDOM_SEED = 1729
EDGE_TIERS = [(1.5, "slim"), (2.5, "moderate"), (4.0, "large")]

# --- Runtime ---------------------------------------------------------------
REQUEST_TIMEOUT = 90
MAX_RETRIES = 3
# The games file changes as results land; play-by-play for a finished season
# never changes, so it is cached forever.
GAMES_TTL_SECONDS = int(os.environ.get("GAMES_TTL_SECONDS", 6 * 3600))
IMMUTABLE_BEFORE_SEASON = int(os.environ.get("IMMUTABLE_BEFORE", 2026))
