"""Quarterback quality and continuity.

This is the layer college never had. CFBD offered "returning passing
production" as a preseason aggregate and nothing week to week, so a team losing
its starter in October was invisible to the model until the results caught up
three games later. The nflverse games file names the starting quarterback for
96% of rows, which makes two different questions answerable before kickoff:

1. **How good is the man starting?** A weighted history of how the offence has
   passed the ball in the games he has started, shrunk toward the league mean
   by how few of them there are.
2. **Is he the usual starter?** A backup's first start is worth real points and
   is invisible to any rating built on scoring margin, because the margin that
   would reveal it has not been played yet.

The second question is the cheaper and probably the more valuable one, and it
costs nothing but bookkeeping.

Leakage
-------
State is accumulated by walking the schedule forward in time and snapshotting
before each week is played, so a lookup for week W of season Y sees only games
strictly before it - including the league average it shrinks toward, which is
maintained in the same pass rather than computed over the full history.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

BLANK = {
    "qb_value": np.nan,       # points-ish: pass EPA above the running league mean
    "qb_starts": 0.0,         # weighted evidence behind that number
    "qb_known": 0,            # did we identify the starter at all
    "qb_new_starter": np.nan, # 1 if he did not start this team's last game
    "qb_team_starts": 0.0,    # consecutive-ish starts for this team
}


def build_starter_table(games: pd.DataFrame,
                        efficiency: pd.DataFrame | None) -> pd.DataFrame:
    """One row per team per game: who started, and how the offence passed.

    Efficiency is optional. Without it the continuity half still works, which
    matters because continuity is the part that does not need play-by-play.
    """
    frames = []
    for side, other in (("home", "away"), ("away", "home")):
        cols = {
            "game_id": games["game_id"],
            "season": games["season"],
            "week": games["week"],
            "team": games[f"{side}_team"],
            "opponent": games[f"{other}_team"],
            "qb_id": games.get(f"{side}_qb_id"),
            "qb_name": games.get(f"{side}_qb"),
            "completed": games["completed"],
        }
        frames.append(pd.DataFrame(cols))
    long = pd.concat(frames, ignore_index=True)

    long["qb_id"] = long["qb_id"].astype(str).str.strip()
    # A blank identifier is not a quarterback. Falling back to the name would
    # merge every "J. Smith" in twenty years into one player.
    bad = long["qb_id"].isin(["", "nan", "None", "<NA>", "NA"])
    long.loc[bad, "qb_id"] = pd.NA

    if efficiency is not None and not efficiency.empty and "pass_epa" in efficiency:
        eff = efficiency[["game_id", "team", "pass_epa", "pass_epa_n"]].copy()
        long = long.merge(eff, on=["game_id", "team"], how="left")
    else:
        long["pass_epa"] = np.nan
        long["pass_epa_n"] = np.nan

    return long.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


class QBEngine:
    """Quarterback value and continuity at any point in the schedule."""

    def __init__(self, starters: pd.DataFrame,
                 shrinkage: float | None = None,
                 halflife: float | None = None):
        self.starters = starters
        self.shrinkage = (config.QB_SHRINKAGE_STARTS if shrinkage is None
                          else float(shrinkage))
        self.halflife = (config.QB_HALFLIFE_GAMES if halflife is None
                         else float(halflife))
        self._snapshots: dict[tuple[int, int], dict] = {}
        self._boundaries: list[tuple[int, int]] = []
        self._build()

    # -- accumulation --------------------------------------------------------
    def _build(self) -> None:
        """Walk the schedule forward once, snapshotting before each week.

        Doing this incrementally rather than filtering the whole table on every
        lookup is the difference between a few milliseconds and several minutes
        across twenty seasons.
        """
        s = self.starters
        if s.empty:
            return

        # qb_id -> [weighted sum of value, total weight, raw starts]
        qb: dict[str, list[float]] = {}
        # team -> [last qb_id, starts by that qb for this team]
        team_last: dict[str, list] = {}
        league = [0.0, 0.0]  # running weighted mean of pass_epa

        decay = 0.5 ** (1.0 / max(self.halflife, 1.0))

        for (year, week), block in s.groupby(["season", "week"], sort=True):
            for row in block.itertuples(index=False):
                if not row.completed:
                    continue

                qid = row.qb_id
                team = row.team
                value = row.pass_epa

                if isinstance(qid, str) and qid:
                    prev = team_last.get(team)
                    same = prev is not None and prev[0] == qid
                    team_last[team] = [qid, (prev[1] + 1) if same else 1]

                if pd.notna(value):
                    # Decay everyone a touch, so old form fades even for a
                    # quarterback who is not playing.
                    league[0] = league[0] * decay + float(value)
                    league[1] = league[1] * decay + 1.0
                    if isinstance(qid, str) and qid:
                        cur = qb.get(qid, [0.0, 0.0, 0.0])
                        cur[0] = cur[0] * decay + float(value)
                        cur[1] = cur[1] * decay + 1.0
                        cur[2] += 1.0
                        qb[qid] = cur

            # Snapshot AFTER this week is folded in, keyed by the week it
            # covers. A lookup for week W then takes the newest snapshot from
            # strictly before W, so it sees every completed game up to that
            # point and no game from W itself.
            #
            # An earlier version stored the state *before* each week and let a
            # missing key fall back to the nearest earlier one, which quietly
            # dropped a week of evidence whenever the target week had no games
            # in the table - exactly the case that arises when predicting a
            # slate that has not been played. The leakage test caught it by
            # comparing against a truncated history; the two disagreed, and the
            # truncated one was right.
            self._boundaries.append((int(year), int(week)))
            self._snapshots[(int(year), int(week))] = {
                "qb": {k: tuple(v) for k, v in qb.items()},
                "team_last": {k: (v[0], v[1]) for k, v in team_last.items()},
                "league": tuple(league),
            }

    # -- lookup --------------------------------------------------------------
    def _snapshot(self, year: int, week: int) -> dict | None:
        """State as of everything completed strictly before (year, week)."""
        import bisect
        i = bisect.bisect_left(self._boundaries, (year, week))
        if i == 0:
            return None
        return self._snapshots[self._boundaries[i - 1]]

    def lookup(self, year: int, week: int, team: str, qb_id) -> dict:
        out = dict(BLANK)
        snap = self._snapshot(int(year), int(week))
        if snap is None:
            return out

        qid = str(qb_id).strip() if qb_id is not None else ""
        if qid in ("", "nan", "None", "<NA>", "NA"):
            # No named starter. Continuity is unanswerable, and saying "0" here
            # would be a confident claim that nothing changed.
            return out

        out["qb_known"] = 1

        lg_sum, lg_w = snap["league"]
        league_mean = lg_sum / lg_w if lg_w > 0 else np.nan

        rec = snap["qb"].get(qid)
        if rec and rec[1] > 0 and np.isfinite(league_mean):
            own = rec[0] / rec[1]
            weight = rec[1] / (rec[1] + self.shrinkage)
            # Shrink toward the league mean, hard when he has barely played.
            # A rookie's first three good games are not evidence of a good
            # quarterback, and this is where that gets said in arithmetic.
            out["qb_value"] = float((own - league_mean) * weight)
            out["qb_starts"] = float(rec[2])

        prev = snap["team_last"].get(team)
        if prev is None:
            # No prior start on record for this team - genuinely unknown, not
            # evidence of a change.
            out["qb_new_starter"] = np.nan
            out["qb_team_starts"] = 0.0
        else:
            out["qb_new_starter"] = 0.0 if prev[0] == qid else 1.0
            out["qb_team_starts"] = float(prev[1]) if prev[0] == qid else 0.0

        return out

    def coverage(self) -> dict:
        """How often a starter was actually identified - worth logging."""
        s = self.starters
        if s.empty:
            return {"rows": 0, "identified": 0.0}
        return {
            "rows": int(len(s)),
            "identified": float(s["qb_id"].notna().mean()),
            "with_pass_epa": float(s["pass_epa"].notna().mean()),
            "distinct_qbs": int(s["qb_id"].nunique()),
        }
