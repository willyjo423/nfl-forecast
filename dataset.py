"""Assemble the canonical game table and everything computable before kickoff.

The college version needed four endpoints, a venue table, a haversine function
and a weather service to reach this point. Here almost all of it arrives in one
file, already computed and, per the first live probe, populated on 100% of
rows: rest days, divisional matchups, roof, surface. What is left to do is
mostly translation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
import nflverse

log = logging.getLogger(__name__)

# Roof values in the file. A retracting roof that was closed is, for football
# purposes, a dome; one that was open is an outdoor game.
INDOOR_ROOFS = {"dome", "closed"}
TURF_SURFACES = {"fieldturf", "field turf", "astroturf", "astroplay",
                 "sportturf", "sport turf", "matrixturf", "a_turf",
                 "fieldturf ", "turf"}


def build_games(seasons: list[int] | None = None) -> pd.DataFrame:
    """The full game table with situational columns derived."""
    g = nflverse.load_games(seasons=seasons)

    # --- venue -------------------------------------------------------------
    # The NFL plays a handful of neutral-site games a year: London, Munich,
    # Mexico City, and the odd relocated game. `location` says so directly.
    loc = g["location"].astype(str).str.strip().str.lower()
    g["neutral_site"] = loc.eq("neutral")

    roof = g["roof"].astype(str).str.strip().str.lower()
    g["is_dome"] = roof.isin(INDOOR_ROOFS).astype(float)
    g.loc[roof.isin(["", "nan", "<na>", "none"]), "is_dome"] = np.nan

    surface = g["surface"].astype(str).str.strip().str.lower()
    g["is_turf"] = surface.isin(TURF_SURFACES).astype(float)
    g.loc[surface.isin(["", "nan", "<na>", "none"]), "is_turf"] = np.nan

    # --- rest --------------------------------------------------------------
    for side in ("home", "away"):
        g[f"{side}_rest"] = pd.to_numeric(g[f"{side}_rest"], errors="coerce")
        # A Thursday game on four days' rest is a genuinely different sport
        # from a Sunday game on seven, and the effect is not linear in days.
        g[f"{side}_short_week"] = (g[f"{side}_rest"] <= 4).astype(float)
        g[f"{side}_long_rest"] = (g[f"{side}_rest"] >= 10).astype(float)
    g["rest_diff"] = g["home_rest"] - g["away_rest"]

    g["div_game"] = pd.to_numeric(g["div_game"], errors="coerce")

    # --- weather as recorded ------------------------------------------------
    # MEASURED: temp and wind are present on 69% of rows. The gap is indoor
    # games, where `roof` already carries the information, plus games not yet
    # played. Indoors is filled with a constant because it genuinely is one;
    # an unplayed outdoor game stays NaN until a forecast fills it, because a
    # plausible default there would be a confident lie.
    g["temp_f"] = pd.to_numeric(g["temp"], errors="coerce")
    g["wind_mph"] = pd.to_numeric(g["wind"], errors="coerce")
    indoors = g["is_dome"] == 1.0
    g.loc[indoors & g["temp_f"].isna(), "temp_f"] = 68.0
    g.loc[indoors & g["wind_mph"].isna(), "wind_mph"] = 0.0

    # --- schedule position --------------------------------------------------
    g = _add_game_numbers(g)

    log.info("games: %d rows, seasons %d-%d, %d completed",
             len(g), g["season"].min(), g["season"].max(),
             int(g["completed"].sum()))
    return g


def _add_game_numbers(g: pd.DataFrame) -> pd.DataFrame:
    """How many games into the season each side is.

    Not the same as the week number once byes exist, and it is the honest
    denominator for "how much do we know about this team".
    """
    long = pd.concat([
        g[["game_id", "season", "week", "home_team"]]
            .rename(columns={"home_team": "team"}).assign(side="home"),
        g[["game_id", "season", "week", "away_team"]]
            .rename(columns={"away_team": "team"}).assign(side="away"),
    ], ignore_index=True)
    long = long.sort_values(["team", "season", "week"])
    long["game_no"] = long.groupby(["team", "season"]).cumcount()

    wide = long.pivot_table(index="game_id", columns="side", values="game_no",
                            aggfunc="first")
    wide.columns = [f"{c}_game_no" for c in wide.columns]
    return g.merge(wide, left_on="game_id", right_index=True, how="left")


def load_efficiency_for(seasons: list[int]) -> pd.DataFrame:
    """Team-game efficiency across the requested seasons.

    Each season is a separate download of roughly 20 MB as parquet, cached
    permanently once a season is finished, so a bootstrap pays this once.
    """
    eff = nflverse.load_efficiency(seasons)
    if eff.empty:
        log.warning("no efficiency data - the efficiency and quarterback "
                    "feature groups will be empty")
    return eff


def training_seasons() -> list[int]:
    return list(range(config.TRAIN_START_YEAR, config.TRAIN_END_YEAR + 1))
