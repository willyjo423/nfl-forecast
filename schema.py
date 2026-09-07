"""Canonical column names for nflverse data.

The college build got burned by assuming field spellings that turned out to be
different, so nothing here is assumed. Every canonical field lists the
candidate source names, missing fields become all-NA columns rather than
raising, and `report()` says exactly what arrived - which is what `selftest.py`
prints, so one live run settles the real schema instead of a guess.
"""
from __future__ import annotations

import pandas as pd

# What the pipeline cannot run without.
REQUIRED_GAME_FIELDS = [
    "game_id", "season", "week", "gameday",
    "home_team", "away_team", "home_score", "away_score",
]

# What sharpens it, and degrades gracefully when absent.
GAME_FIELDS: dict[str, list[str]] = {
    # identity
    "game_id": ["game_id"],
    "season": ["season"],
    "week": ["week"],
    "game_type": ["game_type", "season_type"],
    "gameday": ["gameday", "game_date", "gamedate"],
    "gametime": ["gametime", "game_time"],
    "weekday": ["weekday"],

    # teams and result
    "home_team": ["home_team"],
    "away_team": ["away_team"],
    "home_score": ["home_score"],
    "away_score": ["away_score"],
    "result": ["result"],
    "total": ["total"],
    "location": ["location"],

    # the market
    "spread_line": ["spread_line"],
    "total_line": ["total_line"],
    "home_moneyline": ["home_moneyline"],
    "away_moneyline": ["away_moneyline"],

    # situation - all free here, where the college build had to compute them
    "home_rest": ["home_rest"],
    "away_rest": ["away_rest"],
    "div_game": ["div_game", "divisional_game"],
    "roof": ["roof"],
    "surface": ["surface"],
    "temp": ["temp", "temperature"],
    "wind": ["wind", "wind_speed"],
    "stadium": ["stadium", "stadium_id"],

    # the thing college never had: who actually started at quarterback
    "home_qb": ["home_qb_name", "home_qb"],
    "away_qb": ["away_qb_name", "away_qb"],
    "home_qb_id": ["home_qb_id"],
    "away_qb_id": ["away_qb_id"],
    "home_coach": ["home_coach"],
    "away_coach": ["away_coach"],
    "referee": ["referee"],
}

# Play-by-play columns needed to build team efficiency. Pulling only these
# keeps a 50,000-row season to a few megabytes instead of a few hundred.
PBP_FIELDS: dict[str, list[str]] = {
    "game_id": ["game_id"],
    "season": ["season"],
    "week": ["week"],
    "posteam": ["posteam"],
    "defteam": ["defteam"],
    "play_type": ["play_type"],
    "epa": ["epa"],
    "success": ["success"],
    "yards_gained": ["yards_gained"],
    "pass_attempt": ["pass_attempt", "pass"],
    "rush_attempt": ["rush_attempt", "rush"],
    "down": ["down"],
    "wp": ["wp"],
    "qb_epa": ["qb_epa"],
    "sack": ["sack"],
    "interception": ["interception"],
    "fumble_lost": ["fumble_lost"],
    "penalty": ["penalty"],
}

REQUIRED_PBP_FIELDS = ["game_id", "posteam", "defteam", "epa", "play_type"]


def _first_present(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def normalise(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    """Project a raw frame onto canonical names, tolerating absences."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(mapping))
    out = {}
    for canonical, candidates in mapping.items():
        series = _first_present(df, candidates)
        out[canonical] = series if series is not None else pd.Series(
            [pd.NA] * len(df), index=df.index)
    return pd.DataFrame(out)


def resolved(df: pd.DataFrame, mapping: dict[str, list[str]]) -> list[str]:
    if df is None or df.empty:
        return []
    return [c for c, cands in mapping.items()
            if _first_present(df, cands) is not None]


def missing(df: pd.DataFrame, mapping: dict[str, list[str]]) -> list[str]:
    if df is None or df.empty:
        return list(mapping)
    return [c for c, cands in mapping.items()
            if _first_present(df, cands) is None]


def report(df: pd.DataFrame, mapping: dict[str, list[str]],
           required: list[str]) -> dict:
    """Everything the selftest needs to describe what actually arrived."""
    got, gone = resolved(df, mapping), missing(df, mapping)
    claimed = {c for cands in mapping.values() for c in cands}
    return {
        "n_rows": 0 if df is None else len(df),
        "resolved": got,
        "missing": gone,
        "missing_required": [c for c in gone if c in required],
        # Columns the source provides that this mapping ignores. Worth seeing:
        # it is where the next useful feature usually turns up.
        "unmapped": ([] if df is None or df.empty
                     else sorted(set(df.columns) - claimed)),
    }


def coverage(df: pd.DataFrame, mapping: dict[str, list[str]]) -> dict[str, float]:
    """Share of rows with a real value, per canonical field.

    A column can exist and still be empty - the games file carries `temp` and
    `wind` for played games but not for scheduled ones, and knowing that before
    building on it saves an afternoon.
    """
    if df is None or df.empty:
        return {}
    norm = normalise(df, mapping)
    return {c: float(norm[c].notna().mean()) for c in norm.columns}
