"""Leak-free power ratings.

The single most important guarantee in this project: a rating used to predict a
game in week W of season Y is fit **only** on games played before week W of
season Y, plus a preseason prior built from the previous season's finish.

Method
------
Ridge-regularised least squares on capped scoring margin:

    margin(home, away) = r_home - r_away + hfa

minimising ||Xb - y||^2 + lambda * ||b - prior||^2. The prior term is what
makes week 1 sane: with no games played every rating collapses to its preseason
prior, and as games accumulate the data takes over smoothly. No special-casing
of early weeks is needed.

A second solve splits the same games into offence and defence ratings, which is
what gives the totals model something to work with.

What changed from the college version
-------------------------------------
The prior is much simpler and much more important. College had recruiting
talent, returning production and prior-year SP+ to build a preseason estimate;
the NFL has none of that, so the prior is last season's final rating shrunk by
`YEAR_CARRYOVER`. It also matters more: a 32-team league playing 17 games has
fewer results to work with early on, and the schedule is far less connected in
September than a college slate, so the prior carries the first month.

Franchises, not abbreviations. Relocation renames a team, and without the alias
map in `nflverse.py` the Raiders would arrive in 2020 as an expansion side with
no history.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


def _cap(series: pd.Series, cap: float) -> pd.Series:
    """Soft-cap margins. Blowouts carry information but not linearly."""
    s = series.astype(float)
    sign = np.sign(s)
    mag = s.abs()
    over = mag > cap
    mag = mag.where(~over, cap + np.sqrt(np.clip(mag - cap, 0, None)) * 2.0)
    return sign * mag


def _recency_weights(played: pd.DataFrame, as_of_week: int | None) -> np.ndarray:
    """Exponential decay by how long ago each game was played.

    Half-life is in weeks, so with the default of 5 a game from ten weeks ago
    counts about a quarter as much as last week's. Without this, a September
    blowout anchors a team's rating into December, long after injuries and
    trades have changed the team that played it.
    """
    n = len(played)
    if n == 0:
        return np.ones(0)
    half_life = config.RECENCY_HALFLIFE_WEEKS
    if not half_life or half_life <= 0 or as_of_week is None:
        return np.ones(n)
    weeks_ago = np.clip(
        as_of_week - pd.to_numeric(played["week"], errors="coerce").to_numpy(dtype=float),
        0.0, None)
    weeks_ago = np.nan_to_num(weeks_ago, nan=0.0)
    return np.power(0.5, weeks_ago / float(half_life))


class PreseasonPriors:
    """Where each team starts the season, before a snap is played.

    In this league that is last season's finish, shrunk toward zero. The
    shrinkage is not a fudge: NFL team quality regresses hard year to year, and
    a prior that carried a 12-4 team's full rating into September would be
    reliably too confident about the teams most likely to fall back.
    """

    def __init__(self, carryover: float | None = None):
        self.by_year: dict[int, pd.Series] = {}
        self.carryover = (config.YEAR_CARRYOVER if carryover is None
                          else float(carryover))

    def build(self, year: int, prior_final: pd.Series | None) -> pd.Series:
        if prior_final is None or not len(prior_final):
            self.by_year[year] = pd.Series(dtype=float)
            return self.by_year[year]
        prior = prior_final.astype(float) * self.carryover
        # Centre so the prior means "points better than an average team".
        prior = prior - prior.mean()
        self.by_year[year] = prior
        return prior

    def get(self, year: int, teams: list[str]) -> np.ndarray:
        prior = self.by_year.get(year, pd.Series(dtype=float))
        return prior.reindex(teams).fillna(0.0).to_numpy(dtype=float)


class RatingsEngine:
    """Fits margin and offence/defence ratings at any point in a season."""

    def __init__(self, games: pd.DataFrame, priors: PreseasonPriors | None = None,
                 ridge_lambda: float | None = None):
        self.games = games
        self.priors = priors or PreseasonPriors()
        self.ridge_lambda = ridge_lambda or config.RIDGE_LAMBDA_BASE
        self._cache: dict[tuple[int, int], pd.DataFrame] = {}
        self._final_cache: dict[int, pd.Series] = {}

    # -- solving ------------------------------------------------------------
    def _solve(self, played: pd.DataFrame, year: int,
               as_of_week: int | None = None) -> pd.DataFrame:
        teams = sorted(set(played["home_team"]) | set(played["away_team"]))
        if not teams:
            return pd.DataFrame(columns=["team", "rating", "off", "def", "played"])

        idx = {t: i for i, t in enumerate(teams)}
        n_t, n_g = len(teams), len(played)

        prior = self.priors.get(year, teams)
        w = _recency_weights(played, as_of_week)
        sw = np.sqrt(w)

        neutral = played.get("neutral_site")
        neutral = (np.zeros(n_g, dtype=bool) if neutral is None
                   else neutral.fillna(False).to_numpy(dtype=bool))

        # --- margin solve ---
        X = np.zeros((n_g, n_t + 1))
        rows = np.arange(n_g)
        X[rows, played["home_team"].map(idx).to_numpy()] = 1.0
        X[rows, played["away_team"].map(idx).to_numpy()] = -1.0
        X[:, n_t] = np.where(neutral, 0.0, 1.0)
        y = _cap(played["home_score"] - played["away_score"],
                 config.MARGIN_CAP).to_numpy()

        # Weighted least squares: scaling both sides by sqrt(w) makes an
        # ordinary solve minimise the weighted residual.
        X = X * sw[:, None]
        y = y * sw

        lam = self.ridge_lambda
        P = np.eye(n_t + 1) * lam
        P[n_t, n_t] = lam * 0.25  # let HFA move more freely
        b0 = np.append(prior, config.HFA_PRIOR)

        # Anchor the mean rating at zero so the system is identifiable.
        anchor = np.zeros((1, n_t + 1))
        anchor[0, :n_t] = 1.0
        Xa = np.vstack([X, anchor * 10.0])
        ya = np.append(y, 0.0)

        A = Xa.T @ Xa + P
        rhs = Xa.T @ ya + P @ b0
        beta = np.linalg.solve(A, rhs)
        rating = beta[:n_t]
        hfa = float(beta[n_t])

        # --- offence / defence solve ---
        # Each game contributes two rows: points scored by each side.
        X2 = np.zeros((2 * n_g, 2 * n_t + 1))
        h = played["home_team"].map(idx).to_numpy()
        a = played["away_team"].map(idx).to_numpy()

        r1 = np.arange(n_g)
        X2[r1, h] = 1.0                    # home offence
        X2[r1, n_t + a] = 1.0              # away defence
        X2[r1, 2 * n_t] = np.where(neutral, 0.0, 0.5)

        r2 = np.arange(n_g, 2 * n_g)
        X2[r2, a] = 1.0                    # away offence
        X2[r2, n_t + h] = 1.0              # home defence
        X2[r2, 2 * n_t] = np.where(neutral, 0.0, -0.5)

        y2 = np.concatenate([
            played["home_score"].to_numpy(dtype=float),
            played["away_score"].to_numpy(dtype=float),
        ])

        league_ppg = float(np.mean(y2)) if len(y2) else 22.0
        b02 = np.concatenate([
            np.full(n_t, league_ppg / 2.0) + prior / 4.0,
            np.full(n_t, league_ppg / 2.0) - prior / 4.0,
            [config.HFA_PRIOR],
        ])
        sw2 = np.concatenate([sw, sw])
        X2 = X2 * sw2[:, None]
        y2w = y2 * sw2
        P2 = np.eye(2 * n_t + 1) * (lam * 1.5)
        A2 = X2.T @ X2 + P2
        rhs2 = X2.T @ y2w + P2 @ b02
        beta2 = np.linalg.solve(A2, rhs2)

        played_counts = (
            played["home_team"].value_counts()
            .add(played["away_team"].value_counts(), fill_value=0)
        )

        out = pd.DataFrame({
            "team": teams,
            "rating": rating,
            "off": beta2[:n_t],
            "def": beta2[n_t:2 * n_t],
            "played": [float(played_counts.get(t, 0)) for t in teams],
        })
        out.attrs["hfa"] = hfa
        out.attrs["league_ppg"] = league_ppg
        return out

    # -- public API ---------------------------------------------------------
    def ratings_before(self, year: int, week: int) -> pd.DataFrame:
        """Ratings using only games completed before `week` of `year`."""
        key = (year, week)
        if key in self._cache:
            return self._cache[key]

        g = self.games
        mask = (
            (g["season"] == year)
            & (g["week"] < week)
            & g["home_score"].notna()
            & g["away_score"].notna()
        )
        result = self._solve(g.loc[mask], year, as_of_week=week)
        self._cache[key] = result
        return result

    def final_ratings(self, year: int) -> pd.Series:
        """End-of-season ratings, used only as the *next* year's prior."""
        if year in self._final_cache:
            return self._final_cache[year]
        g = self.games
        mask = ((g["season"] == year) & g["home_score"].notna()
                & g["away_score"].notna())
        # End-of-season summary: weight every game equally.
        res = self._solve(g.loc[mask], year, as_of_week=None)
        series = (res.set_index("team")["rating"] if not res.empty
                  else pd.Series(dtype=float))
        self._final_cache[year] = series
        return series

    def build_priors(self, seasons: list[int]) -> None:
        """Chain each season's prior off the previous season's finish.

        Strictly backwards-looking: season Y's prior is built from Y-1 and
        nothing later, so replaying the chain on a truncated history gives
        identical answers for the seasons that remain.
        """
        for year in sorted(seasons):
            self.priors.build(year, self.final_ratings(year - 1))

    def _from_prior(self, year: int, team: str, hfa: float,
                    league_ppg: float) -> dict:
        """Best estimate for a team with no games this season yet.

        Week 1 has no results at all, so the preseason prior is the *only*
        information available. Returning a flat zero here would make every
        opening-weekend game a pick'em regardless of who was playing.
        """
        prior = self.priors.by_year.get(year, pd.Series(dtype=float))
        rating = float(prior.get(team, 0.0)) if len(prior) else 0.0
        half = league_ppg / 2.0
        return {"rating": rating, "off": half + rating / 4.0,
                "def": half - rating / 4.0, "played": 0.0,
                "hfa": hfa, "known": 0}

    def lookup(self, year: int, week: int, team: str) -> dict:
        table = self.ratings_before(year, week)
        if table.empty:
            return self._from_prior(year, team, config.HFA_PRIOR, 22.0)
        row = table.loc[table["team"] == team]
        hfa = table.attrs.get("hfa", config.HFA_PRIOR)
        if row.empty:
            return self._from_prior(
                year, team, hfa, table.attrs.get("league_ppg", 22.0))
        r = row.iloc[0]
        return {"rating": float(r["rating"]), "off": float(r["off"]),
                "def": float(r["def"]), "played": float(r["played"]),
                "hfa": float(hfa), "known": 1}
