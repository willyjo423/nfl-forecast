"""Opponent-adjusted play-level efficiency, computed week by week.

Why this exists
---------------
Scoring margin alone cannot tell a good team from a lucky one. A team that wins
by 10 after being outgained, on the back of two fumble recoveries, looks
identical to one that won by 10 while dominating every phase - and the second
team is far more likely to win next week. Play-level efficiency separates them
and stabilises much faster.

That matters more in the NFL than it did in college. A 17-game season means
scoring margin is still a small sample in November, and the schedule is short
enough that one bad afternoon moves a rating further than it should.

What is different from the college version
------------------------------------------
Three things.

**The numbers are measured here, not imported.** College pulled a vendor's
pre-aggregated advanced stats; this aggregates raw play-by-play in
`nflverse.team_game_efficiency`, so what "success rate" means is a line of code
rather than someone else's documentation.

**The window crosses seasons.** College could lean on recruiting and returning
production for a September prior. There is no equivalent here, so instead of
starting each season blind, the solve looks back through however many prior
team-games it needs, decaying them by age. In week 1 that is entirely last
season; by week 6 last season barely registers. The decay does the handover, so
no week needs special-casing.

**Rows are weighted by play count.** The first live probe showed competitive
plays per team-game averaging 52 but running as low as 14. A 14-play team-game
is a blowout that got filtered down to almost nothing, and its rates are mostly
noise - so it counts for less, and below `MIN_EFF_PLAYS` it does not count at
all.

Leakage
-------
Everything routes through `stats_before(year, week)`, which is built only from
team-games completed strictly before that point.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

# Aggregated by nflverse.team_game_efficiency. `plays` is carried separately as
# tempo rather than solved as a quality metric.
METRICS = [
    "epa_per_play",
    "success_rate",
    "explosive_rate",
    "pass_epa",
    "rush_epa",
    "yards_per_play",
    "sack_rate",
    "turnover_rate",
    "pass_rate",
    "plays",
]

# Roughly a season and a half of weeks. Long enough that week 1 has a real
# prior, short enough that a team from three years ago does not vote.
EFF_HALFLIFE_WEEKS = 26.0

# How far back to look at all. Beyond this the weights are negligible and the
# solve just gets slower.
EFF_LOOKBACK_WEEKS = 90.0

WEEKS_PER_SEASON = 22.0  # 18 regular + playoffs + the gap to the next opener


def _age_in_weeks(data: pd.DataFrame, year: int, week: int) -> np.ndarray:
    """How long ago each team-game was, in weeks, across season boundaries."""
    season = pd.to_numeric(data["season"], errors="coerce").to_numpy(dtype=float)
    wk = pd.to_numeric(data["week"], errors="coerce").to_numpy(dtype=float)
    return (year - season) * WEEKS_PER_SEASON + (week - wk)


class EfficiencyEngine:
    """Opponent-adjusted efficiency at any point in a season.

    Mirrors RatingsEngine: `stats_before(year, week)` uses only team-games
    played before that point, and results are cached per (year, week).
    """

    def __init__(self, team_games: pd.DataFrame,
                 ridge_lambda: float | None = None):
        self.data = team_games if team_games is not None else pd.DataFrame()
        self.ridge_lambda = ridge_lambda or config.EFF_LAMBDA
        self._cache: dict[tuple[int, int], pd.DataFrame] = {}
        self.available = not self.data.empty

        if self.available:
            self.data = self.data.copy()
            for col in ("season", "week", "plays"):
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
            self.data = self.data.dropna(subset=["season", "week"])
            thin = self.data["plays"] < config.MIN_EFF_PLAYS
            if thin.any():
                log.info("dropping %d team-games with fewer than %d competitive "
                         "plays", int(thin.sum()), config.MIN_EFF_PLAYS)
                self.data = self.data.loc[~thin]
            self.metrics = [m for m in METRICS if m in self.data.columns]
            missing = [m for m in METRICS if m not in self.data.columns]
            if missing:
                log.warning("efficiency metrics absent from the data: %s", missing)
        else:
            self.metrics = []
            log.warning("no play-by-play efficiency available - the model will "
                        "lean on scoring margin alone")

    # -- solving ------------------------------------------------------------
    def _solve(self, played: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
        """One ridge factorisation; every metric solved as a separate RHS.

        For each team-game row: observed = offence(team) + defence(opponent),
        which separates a team's own quality from the standard of opposition it
        has faced. Identical in shape to the points solve in ratings.py, which
        is why the features it feeds are matchup *sums* rather than differences.
        """
        teams = sorted(set(played["team"]) | set(played["opponent"]))
        if len(teams) < 8 or not self.metrics:
            return pd.DataFrame()

        idx = {t: i for i, t in enumerate(teams)}
        n_t, n_r = len(teams), len(played)

        X = np.zeros((n_r, 2 * n_t))
        rows = np.arange(n_r)
        X[rows, played["team"].map(idx).to_numpy()] = 1.0
        X[rows, n_t + played["opponent"].map(idx).to_numpy()] = 1.0

        kept, y_cols = [], []
        for metric in self.metrics:
            col = pd.to_numeric(played[metric], errors="coerce")
            if col.notna().sum() < max(40, 0.2 * n_r):
                continue
            y_cols.append(col.fillna(col.mean()).to_numpy(dtype=float))
            kept.append(metric)
        if not kept:
            return pd.DataFrame()

        Y = np.column_stack(y_cols)
        means = Y.mean(axis=0)

        sw = np.sqrt(weights)
        Xw = X * sw[:, None]
        Yw = (Y - means) * sw[:, None]

        P = np.eye(2 * n_t) * self.ridge_lambda
        # Anchor each half so offence and defence are separately identified.
        anchor = np.zeros((2, 2 * n_t))
        anchor[0, :n_t] = 1.0
        anchor[1, n_t:] = 1.0
        Xa = np.vstack([Xw, anchor * 5.0])
        Ya = np.vstack([Yw, np.zeros((2, len(kept)))])

        beta = np.linalg.solve(Xa.T @ Xa + P, Xa.T @ Ya)

        # Effective sample size, not a raw count: a team whose only evidence is
        # last season should not look as well-measured as one with six games.
        eff_n = (played.assign(_w=weights).groupby("team")["_w"].sum())

        result = {"team": teams,
                  "eff_games": [float(eff_n.get(t, 0.0)) for t in teams]}
        for j, metric in enumerate(kept):
            result[f"off_{metric}"] = beta[:n_t, j]
            result[f"def_{metric}"] = beta[n_t:2 * n_t, j]

        out = pd.DataFrame(result)
        out.attrs["metrics"] = kept
        out.attrs["means"] = dict(zip(kept, means))
        return out

    # -- public API ---------------------------------------------------------
    def stats_before(self, year: int, week: int) -> pd.DataFrame:
        key = (year, week)
        if key in self._cache:
            return self._cache[key]
        if not self.available:
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        age = _age_in_weeks(self.data, year, week)
        recent = (age > 0) & (age <= EFF_LOOKBACK_WEEKS)
        played = self.data.loc[recent]
        if len(played) < 40:
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        decay = np.power(0.5, age[recent] / EFF_HALFLIFE_WEEKS)
        # Weight by how much football each row actually observed, normalised so
        # a typical team-game counts as 1 and the ridge strength keeps meaning
        # the same thing regardless of how many rows are in the window.
        plays = pd.to_numeric(played["plays"], errors="coerce").to_numpy(dtype=float)
        plays = np.nan_to_num(plays, nan=float(np.nanmedian(plays)))
        weights = decay * (plays / max(np.median(plays), 1.0))

        result = self._solve(played, weights)
        self._cache[key] = result
        return result

    def lookup(self, year: int, week: int, team: str) -> dict:
        """Efficiency for one team, or NaNs when there is nothing to report."""
        blank = {f"{side}_{m}": np.nan
                 for m in METRICS for side in ("off", "def")}
        blank["eff_games"] = 0.0

        table = self.stats_before(year, week)
        if table.empty:
            return blank

        row = table.loc[table["team"] == team]
        if row.empty:
            # The solve is centred, so a team with no rows in the window sits
            # at the league mean by construction: zeros, not NaN.
            return {k: 0.0 for k in blank}

        r = row.iloc[0]
        out = dict(blank)
        for m in table.attrs.get("metrics", []):
            out[f"off_{m}"] = float(r[f"off_{m}"])
            out[f"def_{m}"] = float(r[f"def_{m}"])
        out["eff_games"] = float(r["eff_games"])
        return out
