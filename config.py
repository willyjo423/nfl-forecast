"""Central configuration for the NFL build.

Constants that differ from the college version are marked, because most of the
differences are not cosmetic - they reflect a genuinely different sport.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

for _p in (DATA, CACHE, MODELS, DOCS):
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


# Live forecasts still need a weather service: `temp` and `wind` in the games
# file are populated for games already played, not for ones about to be.
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# --- Training window -------------------------------------------------------
# The NFL plays ~272 games a season against college football's ~800, so the
# window has to be wider to reach a comparable sample. 2010 is a deliberate
# floor: earlier football is different enough that the relationships drift.
TRAIN_START_YEAR = int(os.environ.get("TRAIN_START_YEAR", 2010))
TRAIN_END_YEAR = int(os.environ.get("TRAIN_END_YEAR", 2025))
REGULAR_SEASON_WEEKS = 18

# --- Ratings engine --------------------------------------------------------
# Far better determined than the college version: 32 teams playing 17 games
# each, rather than 133 teams playing 12, so the solve needs less shrinkage.
RIDGE_LAMBDA_BASE = 18.0
# NFL margins are tighter, so the blowout cap comes down with them.
MARGIN_CAP = 21.0
# Home field is worth noticeably less than in college, and has been shrinking.
HFA_PRIOR = 1.7
# Rosters are far more stable year to year, so last season carries more.
YEAR_CARRYOVER = 0.72
# An 18-week season with a bye; half-life in weeks.
RECENCY_HALFLIFE_WEEKS = 5.0

# --- Model -----------------------------------------------------------------
RANDOM_SEED = 1729

# --- Runtime ---------------------------------------------------------------
REQUEST_TIMEOUT = 90
MAX_RETRIES = 3
# The games file changes as results land; play-by-play for a finished season
# never changes, so it is cached forever.
GAMES_TTL_SECONDS = int(os.environ.get("GAMES_TTL_SECONDS", 6 * 3600))
IMMUTABLE_BEFORE_SEASON = int(os.environ.get("IMMUTABLE_BEFORE", 2026))
