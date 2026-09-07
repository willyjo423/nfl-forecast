"""Feature construction.

Every feature here is computable *before* kickoff. Two rules keep that honest:

* Team strength comes from `RatingsEngine.ratings_before(season, week)`, which
  by construction sees no game from `week` or later. The same holds for
  `EfficiencyEngine` and `QBEngine`.
* The betting line is deliberately **not** a feature. If the model trained on
  the closing spread it would mostly learn to reproduce it, and "model versus
  market" would stop meaning anything. Lines are used for evaluation and for
  computing the gap at prediction time, and nowhere else.

Feature groups are declared separately so each one can be switched off and
measured. A group that does not improve walk-forward MAE is reported as a null
rather than kept because it sounded like it should help - which is how the
college build ended up honest about its comparables.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from efficiency import METRICS as EFF_METRICS
from efficiency import EfficiencyEngine
from qb import QBEngine
from ratings import RatingsEngine

log = logging.getLogger(__name__)

# --- feature groups --------------------------------------------------------

STRENGTH_COLUMNS = [
    "rating_diff", "rating_sum", "home_rating", "away_rating",
    "home_off", "home_def", "away_off", "away_def",
    "proj_margin", "proj_total",
    "off_matchup_home", "off_matchup_away",
    "home_played", "away_played", "min_played",
    "home_known", "away_known",
]

SITUATION_COLUMNS = [
    "week", "game_no", "neutral_site", "div_game",
    "home_rest", "away_rest", "rest_diff",
    "home_short_week", "away_short_week",
    "home_long_rest", "away_long_rest",
    "is_dome", "is_turf", "temp_f", "wind_mph",
]

# Efficiency enters as matchup *edges* - one side's offence against the other
# side's defence - rather than as four raw numbers per metric. The solve is
# additive, so the expected rate for this matchup is the sum of the two, not
# the difference. `plays` is tempo, handled separately below.
EFF_EDGE_COLUMNS = [f"edge_{m}_{side}"
                    for m in EFF_METRICS if m not in ("plays", "pass_rate")
                    for side in ("home", "away")]

EFF_RAW_COLUMNS = ["home_off_epa", "home_def_epa", "away_off_epa", "away_def_epa"]

EFF_PACE_COLUMNS = ["pace_sum", "pace_diff", "pass_rate_sum", "eff_games"]

EFFICIENCY_COLUMNS = EFF_EDGE_COLUMNS + EFF_RAW_COLUMNS + EFF_PACE_COLUMNS

QB_COLUMNS = [
    "home_qb_value", "away_qb_value", "qb_value_diff",
    "home_qb_starts", "away_qb_starts",
    "home_qb_new", "away_qb_new",
    "home_qb_team_starts", "away_qb_team_starts",
    "qb_known",
]

FEATURE_GROUPS = {
    "strength": STRENGTH_COLUMNS,
    "situation": SITUATION_COLUMNS,
    "efficiency": EFFICIENCY_COLUMNS,
    "quarterback": QB_COLUMNS,
}

FEATURE_COLUMNS = (STRENGTH_COLUMNS + SITUATION_COLUMNS
                   + EFFICIENCY_COLUMNS + QB_COLUMNS)

TARGETS = ["margin", "total", "home_win"]

# Carried through the pipeline for evaluation and display, never fitted on.
MARKET_COLUMNS = ["market_margin", "total_line", "home_moneyline",
                  "away_moneyline"]

_BLANK_EFF = {f"{side}_{m}": np.nan
              for m in EFF_METRICS for side in ("off", "def")}
_BLANK_EFF["eff_games"] = 0.0


def _num(x, default=np.nan) -> float:
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _efficiency_features(eff_h: dict, eff_a: dict) -> dict:
    out: dict[str, float] = {}
    for m in EFF_METRICS:
        if m in ("plays", "pass_rate"):
            continue
        out[f"edge_{m}_home"] = eff_h[f"off_{m}"] + eff_a[f"def_{m}"]
        out[f"edge_{m}_away"] = eff_a[f"off_{m}"] + eff_h[f"def_{m}"]

    out["home_off_epa"] = eff_h["off_epa_per_play"]
    out["home_def_epa"] = eff_h["def_epa_per_play"]
    out["away_off_epa"] = eff_a["off_epa_per_play"]
    out["away_def_epa"] = eff_a["def_epa_per_play"]

    # Tempo. A game total is largely a question of how many snaps get run, and
    # in this league how often they are thrown.
    home_plays = eff_h["off_plays"] + eff_a["def_plays"]
    away_plays = eff_a["off_plays"] + eff_h["def_plays"]
    out["pace_sum"] = home_plays + away_plays
    out["pace_diff"] = home_plays - away_plays
    out["pass_rate_sum"] = (eff_h["off_pass_rate"] + eff_a["def_pass_rate"]
                            + eff_a["off_pass_rate"] + eff_h["def_pass_rate"])
    out["eff_games"] = min(eff_h.get("eff_games", 0.0),
                           eff_a.get("eff_games", 0.0))
    return out


def build_features(games: pd.DataFrame, engine: RatingsEngine,
                   efficiency: EfficiencyEngine | None = None,
                   quarterbacks: QBEngine | None = None) -> pd.DataFrame:
    """Turn a game table into a model-ready feature matrix."""
    rows = []
    for g in games.itertuples(index=False):
        season, week = int(g.season), int(g.week)
        h = engine.lookup(season, week, g.home_team)
        a = engine.lookup(season, week, g.away_team)

        if efficiency is not None:
            eff_h = efficiency.lookup(season, week, g.home_team)
            eff_a = efficiency.lookup(season, week, g.away_team)
        else:
            eff_h = eff_a = dict(_BLANK_EFF)
        eff = _efficiency_features(eff_h, eff_a)

        if quarterbacks is not None:
            qh = quarterbacks.lookup(season, week, g.home_team,
                                     getattr(g, "home_qb_id", None))
            qa = quarterbacks.lookup(season, week, g.away_team,
                                     getattr(g, "away_qb_id", None))
        else:
            from qb import BLANK as QB_BLANK
            qh = qa = dict(QB_BLANK)

        neutral = bool(g.neutral_site)
        hfa = 0.0 if neutral else h["hfa"]

        proj_home_pts = h["off"] + a["def"] + hfa / 2
        proj_away_pts = a["off"] + h["def"] - hfa / 2

        margin = _num(getattr(g, "margin", np.nan))
        total = _num(getattr(g, "game_total", np.nan))

        row = {
            "game_id": g.game_id,
            "season": season,
            "week": week,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "kickoff": getattr(g, "kickoff", None),
            "home_qb": getattr(g, "home_qb", None),
            "away_qb": getattr(g, "away_qb", None),

            # strength
            "home_rating": h["rating"], "away_rating": a["rating"],
            "rating_diff": h["rating"] - a["rating"],
            "rating_sum": h["rating"] + a["rating"],
            "home_off": h["off"], "home_def": h["def"],
            "away_off": a["off"], "away_def": a["def"],
            "proj_margin": (h["rating"] - a["rating"]) + hfa,
            "proj_total": proj_home_pts + proj_away_pts,
            "off_matchup_home": h["off"] - a["def"],
            "off_matchup_away": a["off"] - h["def"],
            "home_played": h["played"], "away_played": a["played"],
            "min_played": min(h["played"], a["played"]),
            "home_known": h["known"], "away_known": a["known"],

            # situation
            "neutral_site": int(neutral),
            "div_game": _num(getattr(g, "div_game", np.nan)),
            "home_rest": _num(getattr(g, "home_rest", np.nan)),
            "away_rest": _num(getattr(g, "away_rest", np.nan)),
            "rest_diff": _num(getattr(g, "rest_diff", np.nan)),
            "home_short_week": _num(getattr(g, "home_short_week", np.nan)),
            "away_short_week": _num(getattr(g, "away_short_week", np.nan)),
            "home_long_rest": _num(getattr(g, "home_long_rest", np.nan)),
            "away_long_rest": _num(getattr(g, "away_long_rest", np.nan)),
            "is_dome": _num(getattr(g, "is_dome", np.nan)),
            "is_turf": _num(getattr(g, "is_turf", np.nan)),
            "temp_f": _num(getattr(g, "temp_f", np.nan)),
            "wind_mph": _num(getattr(g, "wind_mph", np.nan)),
            "game_no": min(_num(getattr(g, "home_game_no", 0.0), 0.0),
                           _num(getattr(g, "away_game_no", 0.0), 0.0)),

            # quarterbacks
            "home_qb_value": qh["qb_value"], "away_qb_value": qa["qb_value"],
            "home_qb_starts": qh["qb_starts"], "away_qb_starts": qa["qb_starts"],
            "home_qb_new": qh["qb_new_starter"],
            "away_qb_new": qa["qb_new_starter"],
            "home_qb_team_starts": qh["qb_team_starts"],
            "away_qb_team_starts": qa["qb_team_starts"],
            "qb_known": float(qh["qb_known"] and qa["qb_known"]),

            # targets and market, kept out of the feature list
            "margin": margin,
            "total": total,
            "market_margin": _num(getattr(g, "market_margin", np.nan)),
            "total_line": _num(getattr(g, "total_line", np.nan)),
            "home_moneyline": _num(getattr(g, "home_moneyline", np.nan)),
            "away_moneyline": _num(getattr(g, "away_moneyline", np.nan)),
        }
        row.update(eff)
        row["qb_value_diff"] = (np.nan
                                if (np.isnan(row["home_qb_value"])
                                    or np.isnan(row["away_qb_value"]))
                                else row["home_qb_value"] - row["away_qb_value"])
        row["home_win"] = (np.nan if np.isnan(margin)
                           else (1.0 if margin > 0 else (0.0 if margin < 0 else np.nan)))
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def training_matrix(feat: pd.DataFrame,
                    columns: list[str] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into X and y, dropping games we cannot learn from."""
    cols = columns or FEATURE_COLUMNS
    usable = feat["margin"].notna() & feat["total"].notna()
    # A game where neither side has played yet this season is carried by the
    # preseason prior alone; it is still informative, so it stays - but a game
    # with no rating at all on either side is not.
    usable &= (feat["home_known"] == 1) | (feat["away_known"] == 1)
    df = feat.loc[usable].copy()
    keep = TARGETS + ["season", "week"] + [c for c in MARKET_COLUMNS
                                           if c in df.columns]
    return df[cols], df[keep]
