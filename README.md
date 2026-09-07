# NFL forecasting model — data layer

This is step one of porting the college football model to the NFL. It is the
data layer only: fetch, normalise, aggregate, and **report what actually
arrived**. No ratings, no features, no model yet — those get ported next,
against a schema this run confirms rather than one I guessed.

That sequencing is deliberate. Most of the time lost on the college build went
to assumed field names and assumed conventions that turned out to be wrong
after a 20-minute run. Here, one probe settles it up front.

---

## The source: nflverse

No API key. No account. No rate limit. Nothing to keep secret in this repo.

Two files carry nearly everything the college build needed five endpoints and
a weather service for:

| | |
|---|---|
| `games.csv` | one row per game: score, closing spread and total, moneylines, rest days, divisional flag, roof, surface, temperature, wind, starting quarterbacks, coaches, referee, stadium |
| `play_by_play_YYYY` | every play, from which team efficiency is aggregated directly rather than trusting someone else's summary |

Three things the NFL gives us that college never did:

1. **The starting quarterback, by name, for every game.** In college this was
   the largest unmeasured variable in the model.
2. **Rest days and short weeks**, already computed.
3. **Actual play-by-play**, so efficiency is measured rather than imported.

And one it takes away: 272 games a season against college football's ~800. The
training window is wider (2010) to compensate, and the ratings solve needs less
shrinkage because 32 teams playing 17 games each is far better determined than
133 teams playing 12.

---

## Run this first

**On GitHub:** Actions → **Probe** → *Run workflow*. Takes a few minutes.

**Locally:**

```
pip install -r requirements.txt
python selftest.py --pbp 2024
```

Then send me the whole output. Three parts of it decide what happens next:

- **The coverage table** — a column can exist and still be entirely empty.
  Coverage, not presence, is what makes a field worth building a feature on.
- **The unconfirmed-fields block** — `home_rest`, `away_rest`, `div_game`, the
  moneylines and the quarterback names were not in the published docs I could
  read. The probe answers each one yes/no with a percentage.
- **The spread orientation verdict** — see below.

Nothing in the probe is fatal. A missing field is a finding; the run finishes
and prints the summary either way.

---

## The one assumption under test

Everything downstream rests on this line in `nflverse.py`:

```python
g["market_margin"] = g["spread_line"]
```

nflverse appears to quote `spread_line` as a **home-perspective margin**:
positive means the home side is favoured by that many. That is already the
orientation this project uses. The sportsbook convention is the opposite sign —
a favourite is negative.

Getting it backwards would not crash anything. It would quietly invert every
prediction in the system, and the model would train happily on the inversion.
So the probe tests it against real results three ways (correlation, error as
written versus flipped, and how often big home favourites actually win) and
prints CONFIRMED, INVERTED or UNCLEAR. If it says INVERTED, the fix is one
character.

`test_data.py` also proves that check has teeth: it feeds the loader a
deliberately inverted file and asserts the test catches it. A test that passes
on both a correct and a broken input is not a test.

---

## Files

| file | what it does |
|---|---|
| `config.py` | every tunable, with the NFL-vs-college differences marked and reasoned |
| `schema.py` | canonical field names, candidate spellings, coverage and unmapped-column reporting |
| `nflverse.py` | fetching with multi-URL fallback and disk cache; game loading; efficiency aggregation |
| `selftest.py` | the live probe |
| `fixtures.py` | a synthetic nflverse with planted ground truth, so tests need no network |
| `test_data.py` | 48 offline checks |
| `storage.py` | parquet-or-pickle persistence, carried over from the college build |

---

## What the offline tests can and cannot prove

`python test_data.py` runs with no network at all and checks 48 things: that a
renamed column still resolves, that a missing column becomes NA instead of
raising, that garbage-time plays are actually excluded rather than silently
averaged in, that the efficiency aggregation recovers a signal planted in the
fixtures, and that an inverted spread file is caught.

What they cannot prove is that the real nflverse columns are spelled the way
the fixtures spell them. Only the live probe can, which is the entire reason it
exists.

---

## What comes next, once the probe reports back

Ported in this order, each measured rather than assumed:

1. **Ratings** — leak-free ridge solve refit at every (season, week), using
   only games strictly before that week.
2. **Efficiency features** — as matchup edges, the way the college build does.
3. **The model** — residual-from-baseline, because gradient boosters average
   leaves and cannot extrapolate; the trees only correct a rescaled linear
   ratings baseline.
4. **QB features** — the genuinely new one. Starter continuity, backup starts,
   and QB EPA carried at the player level rather than the team level.
5. **Rest, short weeks, division games, dome/outdoor** — free in this data.
6. **Comps, calibration, dashboard, results tracker** — straight ports.

Each feature group gets the same A/B treatment as the college build: it earns
its place by improving walk-forward MAE, or it gets reported as a null. The
college build's comps carried no signal on 5,286 out-of-sample games and were
reported as such rather than dressed up. The same standard applies here.

The realistic expectation, stated now rather than after: the NFL closing line
is the most efficient number in sports. The probe prints its MAE. Beating it is
not the goal — forecasting well is.
