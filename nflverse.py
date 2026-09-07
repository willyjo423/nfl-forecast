"""Loading NFL data from nflverse.

No API key, no account, no rate limit. Two resources carry almost everything:

* **games.csv** - one row per game with the final score, the closing spread and
  total, rest days, roof and surface, temperature and wind, the starting
  quarterbacks and the coaches. In the college build that took four separate
  endpoints plus an entire weather service; here it is one file.
* **play-by-play** - one parquet or gzipped CSV per season, from which team
  efficiency is aggregated directly rather than trusting someone else's
  pre-aggregation.

Every fetch tries several URLs, because the GitHub mirror and the author's own
host fail independently, and caches to disk. Finished seasons never change, so
their play-by-play is cached permanently.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config
import schema

log = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """Every source for a resource failed."""


# ---------------------------------------------------------------- fetching
def _cache_path(url: str, suffix: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    stem = url.rsplit("/", 1)[-1].split("?")[0][:40] or "data"
    return config.CACHE / f"{stem}__{digest}{suffix}"


def _download(urls: list[str], ttl: int | None) -> bytes:
    """First URL that answers, cached. `ttl=None` means cache forever."""
    for url in urls:
        cache = _cache_path(url, ".bin")
        if cache.exists():
            fresh = ttl is None or (time.time() - cache.stat().st_mtime) < ttl
            if fresh:
                return cache.read_bytes()

        delay = 3.0
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=config.REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                log.warning("fetch failed (%s): %s", url, exc)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 200 and resp.content:
                cache.write_bytes(resp.content)
                log.info("fetched %s (%.1f MB)", url, len(resp.content) / 1e6)
                return resp.content

            if resp.status_code == 404:
                log.info("not published: %s", url)
                break  # try the next source, not the same one again
            log.warning("HTTP %s from %s", resp.status_code, url)
            time.sleep(delay)
            delay *= 2

    raise DataUnavailable(f"none of these could be fetched: {urls}")


def _read_table(blob: bytes, url: str, usecols=None) -> pd.DataFrame:
    """Parse parquet, gzipped CSV or plain CSV, whichever arrived."""
    if url.endswith(".parquet"):
        try:
            return pd.read_parquet(io.BytesIO(blob), columns=usecols)
        except Exception as exc:  # noqa: BLE001 - fall through to CSV source
            raise DataUnavailable(
                f"parquet unreadable ({exc}); install pyarrow or use the "
                f"csv.gz source") from exc
    if url.endswith(".gz"):
        with gzip.open(io.BytesIO(blob), "rb") as fh:
            return pd.read_csv(fh, low_memory=False, usecols=usecols)
    return pd.read_csv(io.BytesIO(blob), low_memory=False, usecols=usecols)


# ---------------------------------------------------------------- franchises
# A franchise that moves gets a new abbreviation, and its history would
# otherwise be orphaned: the Raiders' 2019 rating would not carry into 2020
# because "OAK" and "LV" look like different teams. Ratings track the
# franchise, so the abbreviations are folded onto one name per franchise.
TEAM_ALIASES = {
    "OAK": "LV",    # Oakland -> Las Vegas, 2020
    "SD": "LAC",    # San Diego -> Los Angeles, 2017
    "STL": "LA",    # St Louis -> Los Angeles, 2016
    "SL": "LA",
    "LAR": "LA",    # nflverse has used both spellings for the Rams
    "JAC": "JAX",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "WSH": "WAS",
}


def canonical_team(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.replace(TEAM_ALIASES)


# ---------------------------------------------------------------- games
def load_games_raw() -> pd.DataFrame:
    """The whole games file, exactly as published."""
    for url in config.GAMES_URLS:
        try:
            blob = _download([url], ttl=config.GAMES_TTL_SECONDS)
            df = _read_table(blob, url)
            log.info("games file: %d rows, %d columns, seasons %s-%s",
                     len(df), len(df.columns),
                     df["season"].min() if "season" in df else "?",
                     df["season"].max() if "season" in df else "?")
            return df
        except DataUnavailable:
            continue
    raise DataUnavailable("could not load the games file from any source")


def load_games(seasons: list[int] | None = None,
               regular_only: bool = False) -> pd.DataFrame:
    """Canonical game table, typed and filtered."""
    raw = load_games_raw()
    g = schema.normalise(raw, schema.GAME_FIELDS)

    for col in ("season", "week", "home_score", "away_score", "result", "total",
                "spread_line", "total_line", "home_moneyline", "away_moneyline",
                "home_rest", "away_rest", "div_game", "temp", "wind"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    g = g.dropna(subset=["season", "week"])
    g["season"] = g["season"].astype(int)
    g["week"] = g["week"].astype(int)

    for col in ("home_team", "away_team"):
        g[col] = canonical_team(g[col])

    g["kickoff"] = pd.to_datetime(
        g["gameday"].astype(str) + " " + g["gametime"].fillna("13:00").astype(str),
        errors="coerce", utc=False)

    if seasons:
        g = g[g["season"].isin(seasons)]
    if regular_only and "game_type" in g.columns:
        g = g[g["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]

    g["completed"] = g["home_score"].notna() & g["away_score"].notna()
    g["margin"] = g["home_score"] - g["away_score"]
    g["game_total"] = g["home_score"] + g["away_score"]

    # nflverse quotes spread_line from the HOME team's perspective as a
    # margin: positive means the home side is favoured by that many. That is
    # already the orientation this project uses, unlike the sportsbook
    # convention where a favourite is negative - so no sign flip here, and a
    # test pins it down rather than trusting the comment.
    g["market_margin"] = g["spread_line"]

    return g.sort_values(["season", "week", "kickoff"]).reset_index(drop=True)


# ---------------------------------------------------------------- play-by-play
def load_pbp(season: int, columns: list[str] | None = None) -> pd.DataFrame:
    """One season of play-by-play, trimmed to the columns we use."""
    ttl = None if season < config.IMMUTABLE_BEFORE_SEASON else config.GAMES_TTL_SECONDS
    wanted = columns or sorted({c for cands in schema.PBP_FIELDS.values()
                                for c in cands})

    last_error: Exception | None = None
    for url in config.pbp_urls(season):
        try:
            blob = _download([url], ttl=ttl)
            try:
                df = _read_table(blob, url, usecols=wanted)
            except (ValueError, KeyError):
                # usecols is strict about names that do not exist; take
                # everything and let the normaliser sort it out.
                df = _read_table(blob, url)
            log.info("play-by-play %s: %d plays", season, len(df))
            return df
        except DataUnavailable as exc:
            last_error = exc
            continue

    raise DataUnavailable(f"no play-by-play for {season}: {last_error}")


def team_game_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw plays into one row per team per game.

    Offence only at this stage - the defensive side of the same game is the
    opponent's offensive row, so pairing them up is the caller's job and keeps
    this function honest about what it measured.
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame()

    p = schema.normalise(pbp, schema.PBP_FIELDS)
    for col in ("epa", "success", "yards_gained", "pass_attempt",
                "rush_attempt", "down", "wp", "sack",
                "interception", "fumble_lost", "penalty", "season", "week"):
        p[col] = pd.to_numeric(p[col], errors="coerce")

    # Real scrimmage plays only: kneels, spikes and special teams would drown
    # the efficiency signal in noise.
    play = p["play_type"].astype(str).str.lower()
    scrimmage = play.isin(["pass", "run"])
    p = p.loc[scrimmage & p["posteam"].notna() & p["defteam"].notna()].copy()
    if p.empty:
        return pd.DataFrame()

    # Garbage time distorts everything; nflverse's win probability makes it
    # cheap to exclude, and we keep the raw count so the caller can see how
    # much of a team-game was thrown away.
    p["competitive"] = p["wp"].between(0.05, 0.95) | p["wp"].isna()

    p["explosive"] = (p["epa"] > 1.0).astype(float)
    p["is_pass"] = p["pass_attempt"].fillna(0)
    p["is_rush"] = p["rush_attempt"].fillna(0)
    p["turnover"] = (p["interception"].fillna(0) + p["fumble_lost"].fillna(0))

    raw_counts = (p.groupby(["game_id", "posteam"], dropna=True).size()
                  .rename("plays_raw").reset_index())

    comp = p.loc[p["competitive"]]
    base = comp if len(comp) > 0.4 * len(p) else p

    keys = ["game_id", "posteam", "defteam"]
    agg = base.groupby(keys, dropna=True).agg(
        season=("season", "first"),
        week=("week", "first"),
        plays=("epa", "size"),
        epa_per_play=("epa", "mean"),
        success_rate=("success", "mean"),
        explosive_rate=("explosive", "mean"),
        pass_rate=("is_pass", "mean"),
        sack_rate=("sack", "mean"),
        turnover_rate=("turnover", "mean"),
        yards_per_play=("yards_gained", "mean"),
    ).reset_index()

    # Passing and rushing efficiency separately. The probe showed `qb_epa` is
    # a near-duplicate of `epa` once averaged to the team-game - identical min
    # and max to three decimals - so carrying both was one real feature and one
    # copy of it. Splitting by play type gives two genuinely different numbers,
    # and the passing half is what the quarterback layer needs.
    for name, mask in (("pass_epa", base["is_pass"] > 0),
                       ("rush_epa", base["is_rush"] > 0)):
        side = (base.loc[mask].groupby(keys, dropna=True)["epa"]
                .agg(["mean", "size"]))
        side.columns = [name, f"{name}_n"]
        agg = agg.merge(side.reset_index(), on=keys, how="left")

    agg = agg.merge(raw_counts, on=["game_id", "posteam"], how="left")
    agg["garbage_share"] = 1.0 - agg["plays"] / agg["plays_raw"].replace(0, np.nan)

    agg = agg.rename(columns={"posteam": "team", "defteam": "opponent"})
    for col in ("team", "opponent"):
        agg[col] = canonical_team(agg[col])
    return agg


def load_efficiency(seasons: list[int]) -> pd.DataFrame:
    """Team-game efficiency across several seasons, skipping any that fail."""
    frames = []
    for season in seasons:
        try:
            frames.append(team_game_efficiency(load_pbp(season)))
        except DataUnavailable as exc:
            log.warning("efficiency for %s unavailable: %s", season, exc)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info("efficiency: %d team-games across %d seasons",
             len(out), out["season"].nunique())
    return out
