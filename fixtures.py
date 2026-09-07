"""A synthetic nflverse, so the data layer can be tested without the network.

The point is not realism for its own sake. The fixtures carry *planted ground
truth*: each team has a hidden strength, margins are generated from it, and the
spread is generated to agree with the home-perspective convention. A test can
therefore assert that the loader recovers a relationship it knows is there,
rather than only asserting that nothing crashed.

Column names and spellings deliberately match the published nflverse files. If
the real ones differ, `selftest.py` will say so on the first live run and these
fixtures get corrected to match - they are a stand-in, not an authority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

ROOFS = ["outdoors", "dome", "closed", "open"]
SURFACES = ["grass", "fieldturf", "astroturf", "sportturf"]

TRUE_HFA = 2.0
NOISE_SD = 10.0
MARKET_NOISE_SD = 2.5


def team_strengths(rng: np.random.Generator, season: int) -> dict[str, float]:
    """Hidden true strength, drifting from season to season."""
    base = np.random.default_rng(season * 7919).normal(0, 6.0, len(TEAMS))
    drift = rng.normal(0, 1.5, len(TEAMS))
    return dict(zip(TEAMS, base + drift))


def make_games(seasons=(2022, 2023, 2024), weeks: int = 17,
               seed: int = 1729, unplayed_last_week: bool = True) -> pd.DataFrame:
    """An nflverse-shaped games.csv with a recoverable signal inside it."""
    rng = np.random.default_rng(seed)
    rows = []

    for season in seasons:
        strength = team_strengths(rng, season)
        for week in range(1, weeks + 1):
            order = list(TEAMS)
            rng.shuffle(order)
            for i in range(0, len(order) - 1, 2):
                home, away = order[i], order[i + 1]
                edge = strength[home] - strength[away] + TRUE_HFA

                # The market sees the truth, blurred a little.
                spread = float(np.round(
                    (edge + rng.normal(0, MARKET_NOISE_SD)) * 2) / 2)
                total_line = float(np.round(
                    (43.0 + rng.normal(0, 4.0)) * 2) / 2)

                margin = edge + rng.normal(0, NOISE_SD)
                combined = max(20.0, total_line + rng.normal(0, 9.0))
                home_pts = int(round((combined + margin) / 2))
                away_pts = int(round((combined - margin) / 2))
                home_pts, away_pts = max(0, home_pts), max(0, away_pts)

                scheduled = (unplayed_last_week and season == max(seasons)
                             and week == weeks)

                rows.append({
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season,
                    "game_type": "REG",
                    "week": week,
                    "gameday": f"{season}-09-{(week % 28) + 1:02d}",
                    "weekday": "Sunday",
                    "gametime": "13:00",
                    "away_team": away,
                    "away_score": np.nan if scheduled else away_pts,
                    "home_team": home,
                    "home_score": np.nan if scheduled else home_pts,
                    "location": "Home",
                    "result": np.nan if scheduled else home_pts - away_pts,
                    "total": np.nan if scheduled else home_pts + away_pts,
                    # Home-perspective margin: positive = home favoured.
                    "spread_line": spread,
                    "total_line": total_line,
                    "away_moneyline": int(rng.integers(-400, 400)),
                    "home_moneyline": int(rng.integers(-400, 400)),
                    # 4 is a Thursday game on a short week, 13/14 is a bye.
                    "away_rest": int(rng.choice([4, 6, 7, 7, 7, 10, 13])),
                    "home_rest": int(rng.choice([4, 6, 7, 7, 7, 10, 13])),
                    "div_game": int(rng.integers(0, 2)),
                    "roof": str(rng.choice(ROOFS)),
                    "surface": str(rng.choice(SURFACES)),
                    # Only populated for games already played, as in the real file.
                    "temp": np.nan if scheduled else int(rng.integers(20, 90)),
                    "wind": np.nan if scheduled else int(rng.integers(0, 22)),
                    "away_qb_name": f"{away} QB1",
                    "home_qb_name": f"{home} QB1",
                    "away_qb_id": f"00-{hash(away) % 10000:04d}",
                    "home_qb_id": f"00-{hash(home) % 10000:04d}",
                    "away_coach": f"{away} Coach",
                    "home_coach": f"{home} Coach",
                    "referee": "R. Eferee",
                    "stadium": f"{home} Field",
                })

    return pd.DataFrame(rows)


def make_pbp(games: pd.DataFrame, plays_per_team: int = 62,
             seed: int = 4104) -> pd.DataFrame:
    """Play-by-play whose efficiency agrees with each game's actual margin.

    Teams that won by a lot generate better plays. That is backwards causally,
    but it is what lets a test assert that the aggregation recovers something
    real rather than shuffling noise.
    """
    rng = np.random.default_rng(seed)
    played = games[games["home_score"].notna()]
    rows = []

    for _, g in played.iterrows():
        margin = float(g["home_score"] - g["away_score"])
        for team, opp, sign in ((g["home_team"], g["away_team"], 1.0),
                                (g["away_team"], g["home_team"], -1.0)):
            quality = sign * margin / 40.0
            n = int(plays_per_team + rng.integers(-8, 9))
            for _ in range(n):
                is_pass = rng.random() < 0.58
                epa = rng.normal(quality * 0.25, 1.2)
                rows.append({
                    "game_id": g["game_id"],
                    "season": g["season"],
                    "week": g["week"],
                    "posteam": team,
                    "defteam": opp,
                    "play_type": "pass" if is_pass else "run",
                    "epa": epa,
                    "success": float(epa > 0),
                    "yards_gained": float(np.clip(epa * 4 + 4, -10, 70)),
                    "pass_attempt": float(is_pass),
                    "rush_attempt": float(not is_pass),
                    "down": int(rng.integers(1, 5)),
                    "wp": float(np.clip(0.5 + sign * margin / 60
                                        + rng.normal(0, 0.15), 0.01, 0.99)),
                    "qb_epa": epa if is_pass else 0.0,
                    "sack": float(is_pass and rng.random() < 0.07),
                    "interception": float(is_pass and rng.random() < 0.025),
                    "fumble_lost": float(rng.random() < 0.01),
                    "penalty": float(rng.random() < 0.08),
                })

        # Plays the filter is supposed to throw away.
        for kind in ("kickoff", "punt", "field_goal", "qb_kneel", "no_play"):
            rows.append({
                "game_id": g["game_id"], "season": g["season"],
                "week": g["week"], "posteam": g["home_team"],
                "defteam": g["away_team"], "play_type": kind,
                "epa": rng.normal(0, 3.0), "success": 0.0,
                "yards_gained": 0.0, "pass_attempt": 0.0, "rush_attempt": 0.0,
                "down": np.nan, "wp": 0.5, "qb_epa": 0.0, "sack": 0.0,
                "interception": 0.0, "fumble_lost": 0.0, "penalty": 0.0,
            })

    return pd.DataFrame(rows)


def make_games_with_qbs(seasons=(2022, 2023, 2024), weeks: int = 17,
                        seed: int = 1729, qb_effect: float = 6.0
                        ) -> tuple[pd.DataFrame, dict]:
    """Games where the starting quarterback genuinely matters.

    Each team has a starter and a clearly worse backup, the backup starts a
    stretch of games, and the margin reflects it. That gives the quarterback
    layer something real to find - and, just as importantly, gives the ablation
    a case where the honest answer is "this helps", so a null result elsewhere
    can be trusted rather than blamed on the harness.
    """
    rng = np.random.default_rng(seed)
    g = make_games(seasons=seasons, weeks=weeks, seed=seed)

    truth = {}
    for team in TEAMS:
        truth[f"{team}_QB1"] = float(rng.normal(qb_effect / 2, 2.0))
        truth[f"{team}_QB2"] = float(rng.normal(-qb_effect / 2, 2.0))

    # Each team loses its starter for a run of games in one season.
    injured = {}
    for season in seasons:
        for team in TEAMS:
            start = int(rng.integers(2, max(3, weeks - 4)))
            injured[(season, team)] = set(range(start, start + 4))

    for side in ("home", "away"):
        ids, names = [], []
        for row in g.itertuples(index=False):
            team = getattr(row, f"{side}_team")
            out = row.week in injured.get((row.season, team), set())
            qb = f"{team}_QB2" if out else f"{team}_QB1"
            ids.append(qb)
            names.append(qb.replace("_", " "))
        g[f"{side}_qb_id"] = ids
        g[f"{side}_qb_name"] = names

    delta = (g["home_qb_id"].map(truth).to_numpy()
             - g["away_qb_id"].map(truth).to_numpy())
    played = g["home_score"].notna()
    g.loc[played, "home_score"] = (g.loc[played, "home_score"]
                                   + delta[played.to_numpy()]).round()
    g["home_score"] = g["home_score"].clip(lower=0)
    g.loc[played, "result"] = g.loc[played, "home_score"] - g.loc[played, "away_score"]
    g.loc[played, "total"] = g.loc[played, "home_score"] + g.loc[played, "away_score"]
    return g, truth


def make_renamed_games(**kwargs) -> pd.DataFrame:
    """The same games under plausible alternative spellings.

    Used to prove the schema mapping actually tolerates a rename instead of
    only working on the one spelling I happened to write first.
    """
    g = make_games(**kwargs)
    return g.rename(columns={
        "gameday": "game_date",
        "game_type": "season_type",
        "temp": "temperature",
        "wind": "wind_speed",
        "home_qb_name": "home_qb",
        "away_qb_name": "away_qb",
        "div_game": "divisional_game",
    })
