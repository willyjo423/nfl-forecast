"""Where each team plays, and whether the weather can reach the field.

The games file gives a stadium name but no coordinates, and a forecast needs a
latitude and longitude. Thirty-two teams is small enough and stable enough to
write down, which is better than another download that can fail on a Sunday
morning.

The coordinates are approximate - good to about a kilometre. That is far more
precision than a temperature and a wind speed require, and being off by a block
changes nothing. The roof flag is the part that matters: a closed roof means
the forecast is irrelevant, not merely mild.
"""
from __future__ import annotations

# team -> (latitude, longitude, roof)
# roof: "outdoor", "dome" (fixed), "retractable" (usually closed in bad weather)
STADIUMS: dict[str, tuple[float, float, str]] = {
    "ARI": (33.5276, -112.2626, "retractable"),
    "ATL": (33.7554, -84.4008, "retractable"),
    "BAL": (39.2780, -76.6227, "outdoor"),
    "BUF": (42.7738, -78.7870, "outdoor"),
    "CAR": (35.2258, -80.8528, "outdoor"),
    "CHI": (41.8623, -87.6167, "outdoor"),
    "CIN": (39.0955, -84.5161, "outdoor"),
    "CLE": (41.5061, -81.6995, "outdoor"),
    "DAL": (32.7473, -97.0945, "retractable"),
    "DEN": (39.7439, -105.0201, "outdoor"),
    "DET": (42.3400, -83.0456, "dome"),
    "GB":  (44.5013, -88.0622, "outdoor"),
    "HOU": (29.6847, -95.4107, "retractable"),
    "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3239, -81.6373, "outdoor"),
    "KC":  (39.0489, -94.4839, "outdoor"),
    "LA":  (33.9535, -118.3392, "dome"),   # SoFi - fixed roof, open sides
    "LAC": (33.9535, -118.3392, "dome"),   # shares SoFi
    "LV":  (36.0909, -115.1833, "dome"),
    "MIA": (25.9580, -80.2389, "outdoor"),
    "MIN": (44.9738, -93.2578, "dome"),
    "NE":  (42.0909, -71.2643, "outdoor"),
    "NO":  (29.9511, -90.0812, "dome"),
    "NYG": (40.8135, -74.0745, "outdoor"),  # shares MetLife
    "NYJ": (40.8135, -74.0745, "outdoor"),
    "PHI": (39.9008, -75.1675, "outdoor"),
    "PIT": (40.4468, -80.0158, "outdoor"),
    "SEA": (47.5952, -122.3316, "outdoor"),
    "SF":  (37.4033, -121.9694, "outdoor"),
    "TB":  (27.9759, -82.5033, "outdoor"),
    "TEN": (36.1665, -86.7713, "outdoor"),
    "WAS": (38.9077, -76.8645, "outdoor"),
}

# Neutral-site games are matched on the stadium name, since the home team's own
# coordinates would put a London kickoff in New Jersey. Matching is on a
# lowercase substring, because the file's spelling of these has changed more
# than once.
NEUTRAL_VENUES: list[tuple[str, float, float, str]] = [
    ("tottenham",       51.6043,  -0.0665, "outdoor"),
    ("wembley",         51.5560,  -0.2795, "outdoor"),
    ("allianz",         48.2188,  11.6247, "outdoor"),
    ("deutsche bank",   50.0686,   8.6455, "outdoor"),
    ("frankfurt",       50.0686,   8.6455, "outdoor"),
    ("azteca",          19.3029, -99.1505, "outdoor"),
    ("corinthians",    -23.5453, -46.4742, "outdoor"),
    ("neo quimica",    -23.5453, -46.4742, "outdoor"),
    ("croke",           53.3607,  -6.2512, "outdoor"),
    ("bernab",          40.4531,  -3.6883, "retractable"),
    ("olympiastadion",  52.5147,  13.2395, "outdoor"),
    ("twickenham",      51.4560,  -0.3415, "outdoor"),
]

INDOOR = {"dome", "retractable"}


def locate(home_team: str, stadium: str | None = None,
           neutral: bool = False) -> tuple[float, float, str] | None:
    """Coordinates and roof for a game, or None when we genuinely do not know.

    Returning None rather than guessing matters: a missing forecast leaves the
    weather features NaN, which the model handles, whereas a wrong location
    would hand it a confident number about the wrong continent.
    """
    name = (stadium or "").strip().lower()
    if name:
        for needle, lat, lon, roof in NEUTRAL_VENUES:
            if needle in name:
                return (lat, lon, roof)
    if neutral:
        return None
    return STADIUMS.get(str(home_team).strip().upper())


def is_indoor(home_team: str, stadium: str | None = None,
              neutral: bool = False) -> bool | None:
    got = locate(home_team, stadium, neutral)
    return None if got is None else got[2] in INDOOR
