"""Forecast weather for an upcoming slate.

The games file records temperature and wind for games already played, which is
what the model trained on. For a game about to kick off there is nothing, so
this fills the gap from Open-Meteo.

Two lessons carried over from the college build, both learned expensively:

* **Batch by date, not by venue.** Open-Meteo bills by variables x hours x
  locations, and one request per stadium turned into thousands of heavy calls
  and a rate-limit wall. One request per date, carrying every venue playing
  that date, is a handful of calls for a whole weekend.
* **A failed lookup stays missing.** Filling an unfetchable game with a mild
  70 degrees and no wind taught the model that the games it could not measure
  were calm afternoons. NaN is handled natively by the booster; a plausible
  default is a confident lie.

Indoor games never need a request at all - the roof already answers.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

import config
import venues

log = logging.getLogger(__name__)

MAX_LOCATIONS_PER_CALL = 40
MAX_ATTEMPTS = 3
DEFAULT_BUDGET_SECONDS = 120.0

UNKNOWN = {"temp_f": np.nan, "wind_mph": np.nan}
INDOORS = {"temp_f": 68.0, "wind_mph": 0.0}


class WeatherService:
    """Kickoff conditions for a slate, fetched once and looked up per game."""

    def __init__(self, budget_seconds: float = DEFAULT_BUDGET_SECONDS):
        self.budget = budget_seconds
        self.started = time.time()
        self.cache: dict[tuple, dict] = {}
        self.requested = 0
        self.failed = 0

    def _out_of_time(self) -> bool:
        return (time.time() - self.started) > self.budget

    # ------------------------------------------------------------------
    def prefetch(self, slate: pd.DataFrame) -> None:
        """One request per kickoff date, carrying every outdoor venue on it."""
        jobs: dict[date, list[tuple]] = {}

        for g in slate.itertuples(index=False):
            spot = venues.locate(g.home_team, getattr(g, "stadium", None),
                                 bool(getattr(g, "neutral_site", False)))
            if spot is None or spot[2] in venues.INDOOR:
                continue
            when = getattr(g, "kickoff", None)
            if when is None or pd.isna(when):
                continue
            day = pd.Timestamp(when).date()
            jobs.setdefault(day, []).append((round(spot[0], 3), round(spot[1], 3)))

        for day, spots in jobs.items():
            uniq = sorted(set(spots))
            for i in range(0, len(uniq), MAX_LOCATIONS_PER_CALL):
                if self._out_of_time():
                    log.warning("weather budget spent; %d locations unfetched",
                                len(uniq) - i)
                    return
                self._fetch(day, uniq[i:i + MAX_LOCATIONS_PER_CALL])

    def _fetch(self, day: date, spots: list[tuple]) -> None:
        params = {
            "latitude": ",".join(str(a) for a, _ in spots),
            "longitude": ",".join(str(b) for _, b in spots),
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": "UTC",
        }
        delay = 2.0
        for _ in range(MAX_ATTEMPTS):
            if self._out_of_time():
                return
            try:
                self.requested += 1
                resp = requests.get(config.OPEN_METEO_FORECAST, params=params,
                                    timeout=30)
            except requests.RequestException as exc:
                log.warning("weather request failed: %s", exc)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 200:
                self._store(day, spots, resp.json())
                return
            log.warning("weather HTTP %s", resp.status_code)
            time.sleep(delay)
            delay *= 2

        self.failed += 1

    def _store(self, day: date, spots: list[tuple], payload) -> None:
        blocks = payload if isinstance(payload, list) else [payload]
        for spot, block in zip(spots, blocks):
            hourly = (block or {}).get("hourly") or {}
            times = hourly.get("time") or []
            if not times:
                continue
            self.cache[(day, spot)] = {
                "time": times,
                "temp": hourly.get("temperature_2m") or [],
                "wind": hourly.get("wind_speed_10m") or [],
            }

    # ------------------------------------------------------------------
    def at_kickoff(self, home_team: str, stadium, kickoff, neutral=False) -> dict:
        spot = venues.locate(home_team, stadium, bool(neutral))
        if spot is None:
            return dict(UNKNOWN)
        if spot[2] in venues.INDOOR:
            return dict(INDOORS)
        if kickoff is None or pd.isna(kickoff):
            return dict(UNKNOWN)

        ts = pd.Timestamp(kickoff)
        key = (ts.date(), (round(spot[0], 3), round(spot[1], 3)))
        block = self.cache.get(key)
        if not block:
            return dict(UNKNOWN)

        # Nearest hour to kickoff.
        want = ts.tz_localize(None) if ts.tzinfo else ts
        stamps = pd.to_datetime(pd.Series(block["time"]), errors="coerce")
        if stamps.isna().all():
            return dict(UNKNOWN)
        i = int((stamps - want).abs().argmin())

        def pick(series):
            try:
                v = float(series[i])
                return np.nan if not np.isfinite(v) else v
            except (IndexError, TypeError, ValueError):
                return np.nan

        return {"temp_f": pick(block["temp"]), "wind_mph": pick(block["wind"])}

    def coverage(self) -> str:
        return (f"{len(self.cache)} venue-days cached from {self.requested} "
                f"requests, {self.failed} failed")


def describe(wx: dict, is_dome: float | None = None) -> str:
    """A short phrase for the card, or nothing when there is nothing to say."""
    if is_dome == 1.0:
        return "Indoors"
    t, w = wx.get("temp_f"), wx.get("wind_mph")
    bits = []
    if t is not None and not pd.isna(t):
        bits.append(f"{t:.0f}°F")
    if w is not None and not pd.isna(w):
        if w >= 18:
            bits.append(f"wind {w:.0f} mph")
        elif w >= 10:
            bits.append(f"breezy {w:.0f} mph")
    return ", ".join(bits)
