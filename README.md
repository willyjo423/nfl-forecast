# NFL forecasting model

A port of the college football forecasting model to the NFL, built against a
schema confirmed by a live probe rather than assumed.

It forecasts three things per game — margin, total, and win probability — and
it does not make betting recommendations. The reason is in **What to expect**
below, and it is not modesty.

---

## The source: nflverse

No API key. No account. No rate limit. Nothing secret lives in this repo.

Two files carry nearly everything the college build needed five endpoints and a
weather service for:

| | |
|---|---|
| `games.csv` | 7,548 games, 1999–2026, 46 columns: score, closing spread and total, moneylines, rest days, divisional flag, roof, surface, temperature, wind, **the starting quarterbacks**, coaches, referee |
| `play_by_play_YYYY` | ~49,000 plays a season, aggregated here into team-game efficiency rather than trusting someone else's summary |

What the probe confirmed, over the 2006–2026 window:

- `home_rest` / `away_rest`, `div_game`, `roof`, `surface` — **100%** populated
- `home_qb_id` / `away_qb_id` — **96%**, and essentially complete for played games
- moneylines — 100% from 2007 on (the 71.6% figure in the first probe run was
  1999–2006 dragging it down, which is why coverage is now reported per window)
- `spread_line` orientation — **confirmed** against 7,276 results: positive
  means the home side is favoured, correlation +0.426, and home teams win 73.3%
  of the games where it exceeds 3

---

## Quick start

**On GitHub:** Actions → **Bootstrap** → *Run workflow*. It downloads twenty
seasons, measures each feature group, runs the walk-forward evaluation, trains
the model and commits it. Expect 30–60 minutes the first time and much less
afterwards, since finished seasons are cached.

**Probe only** (schema and coverage, a few minutes): Actions → **Probe**.

**Locally:**

```
pip install -r requirements.txt
python test_data.py && python test_model.py   # 114 offline checks, no network
python selftest.py --pbp 2024                 # what the live data actually contains
python build.py                               # measure and train
```

---

## What the model is made of

Four feature groups, each declared separately so each can be switched off and
measured.

**Strength.** Ridge-regularised least squares on capped scoring margin, refit at
every (season, week) using only games strictly before that week. The preseason
prior is last season's finish, shrunk by 45% — NFL team quality regresses hard,
and a prior that carried a 13-win season into September would be confidently
wrong about exactly the teams most likely to fall back.

**Situation.** Rest days, short weeks, long rest, divisional games, dome, turf,
temperature, wind. All of it arrives already computed.

**Efficiency.** Team offence and defence solved from raw play-by-play — EPA per
play, success rate, explosive rate, passing and rushing EPA, yards per play,
sack rate, turnover rate, pace — opponent-adjusted, weighted by how many plays
each team-game actually observed, and decayed across the season boundary so
week 1 has a prior instead of starting blind.

**Quarterbacks.** The thing college never had. Who is starting, how the offence
has passed in his starts (shrunk toward the league mean by how few there are),
and — cheaper and probably more valuable — whether he is the man who started
this team's last game. A backup's first start is worth real points and is
invisible to any rating built on scoring margin, because the margin that would
reveal it has not been played yet.

The closing line is **not** a feature. If the model trained on the spread it
would mostly learn to reproduce it, and "model versus market" would stop
meaning anything.

---

## Every feature group has to earn its place

`build.py` prints this before it trains anything:

```
group              cols     MAE    delta
strength             17  xx.xxx   +0.000
situation            32  xx.xxx   +x.xxx
efficiency           56  xx.xxx   +x.xxx
quarterback          66  xx.xxx   +x.xxx
```

Groups are added cumulatively to the ratings baseline, measured by walk-forward
MAE. A group that does not move that number is reported as a null and should be
removed, however good the story behind it is. The college build kept an entire
comparables engine that turned out to be worth 0.008 points, and the only
reason anyone found out is that it got measured instead of assumed.

The harness is itself tested. `test_model.py` builds the same fixture twice —
once where the starting quarterback is worth eight points and once where every
quarterback is identical — and requires the quarterback features to help in the
first case and not in the second. Without that second half, "the harness said
it helps" would be indistinguishable from "adding columns always helps a bit".

---

## What to expect

The probe measured the closing line's own accuracy over 7,276 games:

```
market margin MAE   10.27 pts
market total MAE    10.62 pts
```

That is the bar, and it is the hardest one in sport. The college model settled
about 1.5 points behind its market; the same gap or worse is the realistic
expectation here, on a third as many games per season. If a run ever reports
the model *beating* the closing line, treat it as a bug until a full season of
forward-only results says otherwise.

So this is a forecasting tool. It tells you what a game is likely to look like
and how uncertain that is. It does not tell you what to bet.

---

## Three things the first probe run corrected

Worth recording, because each was wrong in a way that would not have thrown an
error.

**The ridge strength was three times too strong.** `RIDGE_LAMBDA_BASE` started
at 18, reasoned from 32 teams playing 17 games being better determined than 133
playing 12. That confused how well-connected a schedule is with how informative
a single game is, and it is the second that sets the shrinkage. At 18, projected
margins had a standard deviation of 3.1 points against a real spread of 13.7 —
every game came out looking like a coin flip. The right value is roughly game
noise variance over team quality variance, which is about 5.

**`qb_epa` was a duplicate.** Averaged to the team-game it matched `epa` to
three decimals, identical minimum and maximum. It was one feature counted
twice. Passing and rushing EPA replaced it.

**Coverage over the whole file was misleading.** Moneylines read 71.6% and
temperature 69%, which sounds like sparse data and is not: the first is old
seasons, the second is indoor games where `roof` already says so. Coverage is
now reported for the training window alongside the total.

A fourth was caught by the tests rather than the probe: the quarterback engine's
snapshots were stored *before* each week and fell back to the nearest earlier
one, which silently dropped a week of evidence whenever the target week had no
games in the table — exactly the case that arises when predicting a slate that
has not been played yet. The leakage test compared against a truncated history,
the two disagreed, and the truncated one was right.

---

## Files

| file | what it does |
|---|---|
| `config.py` | every tunable, with the NFL-vs-college differences reasoned and the probe-measured ones marked |
| `schema.py` | canonical field names, candidate spellings, coverage and unmapped-column reporting |
| `nflverse.py` | fetching with multi-URL fallback and disk cache; game loading; franchise aliases; efficiency aggregation |
| `dataset.py` | situational columns — venue, rest, short weeks, weather |
| `ratings.py` | the leak-free ridge ratings solve |
| `efficiency.py` | opponent-adjusted play efficiency, week by week, across season boundaries |
| `qb.py` | quarterback value and continuity |
| `features.py` | the feature matrix, declared in switchable groups |
| `model.py` | fitting, scale and win-probability calibration, walk-forward evaluation, ablation |
| `build.py` | the bootstrap: fetch, measure, train |
| `selftest.py` | the live probe |
| `fixtures.py` | a synthetic nflverse with planted ground truth |
| `test_data.py` | 51 offline checks of the data layer |
| `test_model.py` | 63 offline checks of the modelling layers, including leakage |

---

## Leakage

Every engine answers "as of (season, week)" and is tested by deleting the future
and demanding the same answer. `test_model.py` does this for both the ratings
and the quarterback engine: it computes a value, deletes every game from that
week onward, recomputes, and requires the two to be bit-identical. A model that
has seen the future produces beautiful numbers, is worthless, and fails silently
— so it gets an explicit test rather than a careful comment.

## Still to port

The daily forecast job, the dashboard, and the results tracker, all of which are
straight ports from the college build once the numbers here are confirmed
against real data. The comparables engine is **not** on that list: it measured
as a null over 5,286 out-of-sample college games and there is no reason to
expect better in a smaller league.
