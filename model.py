"""Model fitting, calibrated win probability, and walk-forward evaluation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error

import config
from features import FEATURE_COLUMNS

log = logging.getLogger(__name__)


class FeatureMismatchError(RuntimeError):
    """Saved model was fitted on a different feature set than the code builds."""


# The two ratings-derived baselines the trees correct rather than replace.
BASELINE_MARGIN = "proj_margin"
BASELINE_TOTAL = "proj_total"


def _regressor(**kw) -> HistGradientBoostingRegressor:
    params = dict(
        loss="absolute_error",   # margins are heavy-tailed; MAE is the honest loss
        max_iter=400,
        learning_rate=0.04,
        max_depth=None,
        # A third the training data of the college build, so the trees are
        # smaller and the leaves larger. Left as they were, they would fit
        # 5,000 games as confidently as 15,000, which is how a model ends up
        # memorising a decade of football.
        max_leaf_nodes=16,
        min_samples_leaf=60,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=config.RANDOM_SEED,
    )
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def _fit_line(x: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept, with a safe fallback.

    The ratings baseline is systematically compressed early in a season, when
    ridge shrinkage pulls every team toward its preseason prior. Rescaling it
    against reality before the trees see it means the model inherits a baseline
    that is on the right scale, not merely in the right order. In the college
    build this one step was worth several points of MAE, because a gradient
    booster can only average the leaves it has seen and cannot extrapolate past
    the largest mismatch in its training data.
    """
    ok = np.isfinite(x) & np.isfinite(target)
    if ok.sum() < 200 or np.std(x[ok]) < 1e-6:
        return 1.0, 0.0
    slope, intercept = np.polyfit(x[ok], target[ok], 1)
    if not np.isfinite(slope) or not (0.2 <= slope <= 5.0):
        return 1.0, 0.0
    return float(slope), float(intercept)


@dataclass
class NFLModel:
    """Margin, total, and win probability, fitted together."""

    margin: HistGradientBoostingRegressor | None = None
    total: HistGradientBoostingRegressor | None = None
    margin_line: tuple[float, float] = (1.0, 0.0)
    total_line: tuple[float, float] = (1.0, 0.0)
    # Slope and intercept applied to the finished margin, fitted on genuinely
    # held-out seasons. See `_fit_scale`.
    margin_calibration: tuple[float, float] = (1.0, 0.0)
    wp_coef: float = 0.0
    wp_intercept: float = 0.0
    margin_sigma: float = 13.5
    total_sigma: float = 10.5
    # Residual spread as a function of how many games the teams have played.
    # A week-2 forecast leaning on last season's carryover deserves less
    # confidence than a week-12 one, and one global sigma cannot say so.
    sigma_by_played: list = field(default_factory=list)
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_seasons: list[int] = field(default_factory=list)

    # -- baselines ----------------------------------------------------------
    def _baseline(self, X: pd.DataFrame, which: str) -> np.ndarray:
        col, (a, b) = ((BASELINE_MARGIN, self.margin_line) if which == "margin"
                       else (BASELINE_TOTAL, self.total_line))
        raw = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        return a * np.nan_to_num(raw, nan=0.0) + b

    # -- fitting ------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "NFLModel":
        """Fit the trees on the *residual* from a rescaled ratings baseline."""
        X = X[self.features]

        self.margin_line = _fit_line(
            X[BASELINE_MARGIN].to_numpy(dtype=float),
            y["margin"].to_numpy(dtype=float))
        self.total_line = _fit_line(
            X[BASELINE_TOTAL].to_numpy(dtype=float),
            y["total"].to_numpy(dtype=float))
        log.info("baseline scaling: margin %.2fx%+.2f, total %.2fx%+.2f",
                 *self.margin_line, *self.total_line)

        margin_base = self._baseline(X, "margin")
        total_base = self._baseline(X, "total")

        self.margin = _regressor().fit(X, y["margin"].to_numpy() - margin_base)
        self.total = _regressor().fit(X, y["total"].to_numpy() - total_base)

        # One held-out pass serves two purposes: it says whether the finished
        # forecasts are on the right scale, and it supplies the margins the
        # win-probability curve is fitted to.
        shadow = self._shadow(X, y)
        self.margin_calibration = self._fit_scale(shadow)

        resid = y["margin"].to_numpy() - self._apply_scale(
            margin_base + self.margin.predict(X))
        # 1.4826 * MAD is a robust sd estimate, less swayed by 40-point games.
        self.margin_sigma = float(max(
            8.0, 1.4826 * np.median(np.abs(resid - np.median(resid)))))
        tresid = y["total"].to_numpy() - (total_base + self.total.predict(X))
        self.total_sigma = float(max(
            6.0, 1.4826 * np.median(np.abs(tresid - np.median(tresid)))))

        self.sigma_by_played = self._fit_sigma_curve(X, resid)
        self.trained_seasons = sorted(y["season"].unique().tolist())
        self._fit_win_probability(X, y, shadow)
        return self

    # -- scale ---------------------------------------------------------------
    def _shadow(self, X: pd.DataFrame, y: pd.DataFrame):
        """Refit on early seasons and predict the last two. Honest or nothing.

        Anything measured on the training rows themselves is measured on games
        the trees have already memorised, and would report a model far better
        calibrated than it is. If there is not enough history to hold seasons
        back, this returns None and the callers fall back rather than pretend.
        """
        seasons = sorted(y["season"].dropna().unique())
        if len(seasons) < 4:
            return None
        cutoff = seasons[-2]
        train = (y["season"] < cutoff).to_numpy()
        held = (y["season"] >= cutoff).to_numpy()
        if train.sum() < 600 or held.sum() < 200:
            return None

        line = _fit_line(X.loc[train, BASELINE_MARGIN].to_numpy(dtype=float),
                         y.loc[train, "margin"].to_numpy(dtype=float))

        def base(sel):
            raw = X.loc[sel, BASELINE_MARGIN].to_numpy(dtype=float)
            return line[0] * np.nan_to_num(raw, nan=0.0) + line[1]

        trees = _regressor().fit(X.loc[train],
                                 y.loc[train, "margin"].to_numpy() - base(train))
        return {
            "pred": base(held) + trees.predict(X.loc[held]),
            "actual": y.loc[held, "margin"].to_numpy(dtype=float),
            "win": y.loc[held, "home_win"].to_numpy(),
        }

    @staticmethod
    def _fit_scale(shadow) -> tuple[float, float]:
        """How much to stretch or shrink the finished margin.

        Regressing the real outcome on the forecast answers a specific
        question: when this model says seven points, what do such games
        actually finish at? A slope of 1 means the forecasts are on the right
        scale. Above 1 means they are systematically too timid - the college
        build's compression bug, where the answer was 3.28. Below 1 means the
        opposite, which is what the trees do when they have too little history
        and start fitting noise as though it were signal.

        The correction is clamped. A slope this far from 1 is more likely to be
        a small held-out sample talking than a real property of the model, and
        a calibration step that can reverse or triple a forecast is a bigger
        hazard than the miscalibration it is fixing.
        """
        if not shadow:
            return (1.0, 0.0)
        pred, actual = shadow["pred"], shadow["actual"]
        ok = np.isfinite(pred) & np.isfinite(actual)
        if ok.sum() < 200 or np.std(pred[ok]) < 1e-6:
            return (1.0, 0.0)
        slope, intercept = np.polyfit(pred[ok], actual[ok], 1)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return (1.0, 0.0)
        slope = float(np.clip(slope, 0.55, 1.45))
        intercept = float(np.clip(intercept, -3.0, 3.0))
        if abs(slope - 1.0) > 0.05 or abs(intercept) > 0.5:
            log.info("margin scale calibration: %.3fx %+.2f "
                     "(fitted on %d held-out games)", slope, intercept, int(ok.sum()))
        return (slope, intercept)

    def _apply_scale(self, margin: np.ndarray) -> np.ndarray:
        a, b = self.margin_calibration
        return a * margin + b

    # -- uncertainty as a function of evidence -------------------------------
    def _fit_sigma_curve(self, X: pd.DataFrame, resid) -> list:
        """Measure residual spread separately for thin and thick evidence.

        Buckets are tighter than the college version because an NFL season is
        17 games, not 12 plus a bye week: by game six a team has played a third
        of its schedule, where a college team is barely halfway to that.
        """
        if "min_played" not in X.columns:
            return []
        played = pd.to_numeric(X["min_played"], errors="coerce").to_numpy()
        r = np.asarray(resid, dtype=float)

        curve = []
        for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 99)):
            sel = (played >= lo) & (played < hi) & np.isfinite(r)
            if sel.sum() < 150:
                continue
            block = r[sel]
            sd = float(1.4826 * np.median(np.abs(block - np.median(block))))
            curve.append({"lo": float(lo), "hi": float(hi),
                          "n": int(sel.sum()), "sigma": max(sd, 6.0)})

        if len(curve) < 2:
            return []

        # More evidence cannot make a forecast less certain, so impose that
        # rather than letting bucket noise invert it. Without this the fitted
        # curve can come back tighter for thin evidence, and the adjustment
        # would then make week-1 predictions *more* confident - the exact
        # opposite of the point.
        for i in range(len(curve) - 2, -1, -1):
            curve[i]["sigma"] = max(curve[i]["sigma"], curve[i + 1]["sigma"])

        log.info("residual spread by games played: %s",
                 ", ".join(f"{c['lo']:.0f}-{c['hi']:.0f}: {c['sigma']:.1f}"
                           f" (n={c['n']:,})" for c in curve))
        return curve

    def sigma_for(self, min_played) -> np.ndarray:
        m = np.asarray(min_played, dtype=float)
        out = np.full(m.shape, self.margin_sigma, dtype=float)
        for c in (self.sigma_by_played or []):
            out = np.where((m >= c["lo"]) & (m < c["hi"]), c["sigma"], out)
        return np.nan_to_num(out, nan=self.margin_sigma)

    def _fit_win_probability(self, X: pd.DataFrame, y: pd.DataFrame,
                             shadow=None) -> None:
        """Map predicted margin to win probability with a one-variable logistic.

        Isotonic regression was the wrong tool here in the college build: fitted
        on a few hundred held-out games it produced flat plateaus, so a coin
        flip came out at 61% and a near-certainty got dragged down to 90%. A
        logistic in the predicted margin has two parameters, is monotonic by
        construction, and keeps rising sensibly past the largest mismatch in the
        training data.
        """
        wmask = y["home_win"].notna().to_numpy()
        if wmask.sum() < 400:
            self.wp_coef, self.wp_intercept = 1.0 / self.margin_sigma, 0.0
            return

        margins, wins = None, None
        if shadow is not None:
            # Fit the mapping on genuinely out-of-sample margins, so it
            # reflects how confident the model deserves to be on games it has
            # not seen - and on the *calibrated* margins, since those are what
            # the caller will actually be handed.
            keep = np.isfinite(shadow["pred"]) & pd.notna(shadow["win"])
            if keep.sum() >= 250:
                margins = self._apply_scale(shadow["pred"][keep])
                wins = shadow["win"][keep]

        if margins is None:
            margins = self._apply_scale(
                self._baseline(X, "margin") + self.margin.predict(X))[wmask]
            wins = y.loc[wmask, "home_win"].to_numpy()

        clf = LogisticRegression(C=1e6, solver="lbfgs")
        clf.fit(margins.reshape(-1, 1), wins.astype(int))
        self.wp_coef = float(clf.coef_[0][0])
        self.wp_intercept = float(clf.intercept_[0])
        log.info("win probability: p = sigmoid(%.4f * margin %+.4f) "
                 "-> 1 pt of margin is worth %.1f%% at the coin flip",
                 self.wp_coef, self.wp_intercept, 25 * self.wp_coef)

    # -- prediction ---------------------------------------------------------
    def _win_prob(self, margin: np.ndarray,
                  min_played: np.ndarray | None = None) -> np.ndarray:
        if min_played is None or not self.sigma_by_played:
            scale = 1.0
        else:
            # Capped at 1: this may only widen a probability toward a coin
            # flip, never sharpen one. A bucket whose residuals happen to come
            # back tighter than average is far more likely to be noise than a
            # licence for extra confidence, and overconfidence on thin evidence
            # is the costlier mistake.
            scale = np.minimum(1.0, self.margin_sigma / self.sigma_for(min_played))

        if self.wp_coef <= 0:
            sigma = (self.margin_sigma if min_played is None
                     else self.sigma_for(min_played))
            return norm.cdf(margin / sigma)
        z = np.clip(self.wp_coef * margin * scale + self.wp_intercept, -12, 12)
        return 1.0 / (1.0 + np.exp(-z))

    def check_compatible(self, X: pd.DataFrame) -> None:
        """Fail loudly if the saved model predates the current feature set."""
        missing = [c for c in self.features if c not in X.columns]
        if not missing:
            return
        raise FeatureMismatchError(
            f"The saved model expects features that this code no longer "
            f"produces: {missing}. The model file is out of date with the "
            f"feature pipeline.\n\n"
            f"Fix: re-run the Bootstrap workflow to retrain. "
            f"(Model was fitted on seasons "
            f"{min(self.trained_seasons, default='?')}-"
            f"{max(self.trained_seasons, default='?')}.)")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.check_compatible(X)
        Xf = X[self.features]
        margin = self._apply_scale(
            self._baseline(Xf, "margin") + self.margin.predict(Xf))
        total = self._baseline(Xf, "total") + self.total.predict(Xf)
        # A total below a floor is nonsense and would poison the score split.
        total = np.clip(total, 20.0, 90.0)
        played = (Xf["min_played"].to_numpy(dtype=float)
                  if "min_played" in Xf.columns else None)
        prob = np.clip(self._win_prob(margin, played), 0.005, 0.995)
        return pd.DataFrame({
            "pred_margin": margin,
            "pred_total": total,
            "home_win_prob": prob,
            "margin_sigma": (self.sigma_for(played) if played is not None
                             else np.full(len(Xf), self.margin_sigma)),
            "pred_home_points": (total + margin) / 2,
            "pred_away_points": (total - margin) / 2,
        }, index=X.index)


# -- evaluation -------------------------------------------------------------
def walk_forward(feat: pd.DataFrame, min_train_seasons: int = 5,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """Refit at each season boundary and predict the next season only.

    This is the only evaluation that reflects how the model will be used. A
    random train/test split would let it see later games from the same season
    and would flatter it badly.
    """
    from features import training_matrix

    X_all, y_all = training_matrix(feat, columns=columns)
    seasons = sorted(y_all["season"].unique())
    out = []

    for i, season in enumerate(seasons):
        if i < min_train_seasons:
            continue
        train = y_all["season"] < season
        test = y_all["season"] == season
        if train.sum() < 500 or test.sum() == 0:
            continue

        model = NFLModel(features=list(X_all.columns))
        model.fit(X_all.loc[train.values], y_all.loc[train])
        preds = model.predict(X_all.loc[test.values])
        block = y_all.loc[test].join(preds)
        block["season_tested"] = season
        out.append(block)
        log.info("walk-forward %s: trained on %d, tested on %d",
                 season, int(train.sum()), int(test.sum()))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(oos: pd.DataFrame) -> dict:
    """Metrics that matter, including the ones that are unflattering."""
    if oos.empty:
        return {}

    m = {"n_games": int(len(oos))}
    m["margin_mae"] = float(mean_absolute_error(oos["margin"], oos["pred_margin"]))
    m["total_mae"] = float(mean_absolute_error(oos["total"], oos["pred_total"]))

    wp = oos.dropna(subset=["home_win"])
    if len(wp) > 50:
        m["win_logloss"] = float(log_loss(wp["home_win"], wp["home_win_prob"]))
        m["win_accuracy"] = float(
            ((wp["home_win_prob"] > 0.5).astype(int) == wp["home_win"]).mean())
        m["win_brier"] = float(np.mean((wp["home_win_prob"] - wp["home_win"]) ** 2))

        bins = pd.cut(wp["home_win_prob"], np.linspace(0, 1, 11))
        grouped = wp.groupby(bins, observed=True).agg(
            pred=("home_win_prob", "mean"),
            obs=("home_win", "mean"),
            n=("home_win", "size"))
        m["win_ece"] = float(
            (grouped["pred"] - grouped["obs"]).abs()
            .mul(grouped["n"]).sum() / grouped["n"].sum())
        m["calibration_table"] = [
            {"bin": str(idx), "predicted": round(float(r["pred"]), 4),
             "observed": round(float(r["obs"]), 4), "n": int(r["n"])}
            for idx, r in grouped.iterrows()]

    # --- the real benchmark: the closing line ---
    # market_margin is already a home-perspective margin; the first live probe
    # confirmed the orientation against 7,276 results rather than assuming it.
    lined = oos.dropna(subset=["market_margin", "margin"]).copy()
    if len(lined) > 100:
        m["market_margin_mae"] = float(
            mean_absolute_error(lined["margin"], lined["market_margin"]))
        m["model_margin_mae_on_lined"] = float(
            mean_absolute_error(lined["margin"], lined["pred_margin"]))
        m["mae_vs_market"] = m["model_margin_mae_on_lined"] - m["market_margin_mae"]

        lined["edge"] = lined["pred_margin"] - lined["market_margin"]
        lined["ats_correct"] = np.where(
            lined["edge"] > 0,
            lined["margin"] > lined["market_margin"],
            lined["margin"] < lined["market_margin"])
        push = np.isclose(lined["margin"], lined["market_margin"])
        graded = lined.loc[~push]
        m["ats_n"] = int(len(graded))
        m["ats_win_pct"] = (float(graded["ats_correct"].mean())
                            if len(graded) else None)

        tiers = {}
        for threshold, label in config.EDGE_TIERS:
            sel = graded.loc[graded["edge"].abs() >= threshold]
            if len(sel) >= 30:
                tiers[label] = {"threshold": threshold, "n": int(len(sel)),
                                "ats_win_pct": float(sel["ats_correct"].mean())}
        m["ats_by_tier"] = tiers
        m["breakeven_at_minus_110"] = 0.5238

    totals = oos.dropna(subset=["total_line", "total"]).copy()
    if len(totals) > 100:
        m["market_total_mae"] = float(
            mean_absolute_error(totals["total"], totals["total_line"]))
        m["model_total_mae_on_lined"] = float(
            mean_absolute_error(totals["total"], totals["pred_total"]))

    by_season = (oos.groupby("season_tested")
                 .apply(lambda d: mean_absolute_error(d["margin"], d["pred_margin"]),
                        include_groups=False)
                 .round(3).to_dict())
    m["margin_mae_by_season"] = {int(k): float(v) for k, v in by_season.items()}
    return m


def ablation(feat: pd.DataFrame, groups: dict[str, list[str]],
             base_group: str = "strength") -> dict:
    """Measure what each feature group is actually worth.

    Every group has to earn its place by improving walk-forward MAE. The
    college build kept a whole comparables engine that turned out to be worth
    +0.008 points, and the only reason that was ever discovered is that it got
    measured instead of assumed. This runs the same test here, before anyone
    gets attached to the quarterback layer.
    """
    results = {}
    base_cols = list(groups[base_group])

    baseline = evaluate(walk_forward(feat, columns=base_cols))
    if not baseline:
        return {}
    results[base_group] = {"columns": len(base_cols),
                           "margin_mae": baseline["margin_mae"],
                           "delta": 0.0}
    log.info("ablation %-14s %d cols -> MAE %.3f",
             base_group, len(base_cols), baseline["margin_mae"])

    cumulative = list(base_cols)
    for name, cols in groups.items():
        if name == base_group:
            continue
        cumulative = cumulative + [c for c in cols if c not in cumulative]
        got = evaluate(walk_forward(feat, columns=cumulative))
        if not got:
            continue
        delta = got["margin_mae"] - results[base_group]["margin_mae"]
        results[name] = {"columns": len(cumulative),
                         "margin_mae": got["margin_mae"],
                         "delta": delta}
        log.info("ablation +%-13s %d cols -> MAE %.3f (%+.3f vs %s alone)",
                 name, len(cumulative), got["margin_mae"], delta, base_group)
    return results


def summarize(metrics: dict) -> str:
    if not metrics:
        return "No metrics (empty evaluation set)."
    lines = [
        f"Games evaluated out-of-sample : {metrics['n_games']:,}",
        f"Margin MAE                    : {metrics['margin_mae']:.2f} pts",
        f"Total MAE                     : {metrics['total_mae']:.2f} pts",
    ]
    if "win_accuracy" in metrics:
        lines += [
            f"Win-pick accuracy             : {metrics['win_accuracy']*100:.1f}%",
            f"Win-prob log loss             : {metrics['win_logloss']:.4f}",
            f"Brier score                   : {metrics['win_brier']:.4f}",
        ]
    if "market_margin_mae" in metrics:
        delta = metrics["mae_vs_market"]
        verdict = "better than" if delta < 0 else "worse than"
        lines += [
            "",
            f"Closing-line margin MAE       : {metrics['market_margin_mae']:.2f} pts",
            f"Model margin MAE (same games) : {metrics['model_margin_mae_on_lined']:.2f} pts",
            f"  -> model is {abs(delta):.2f} pts {verdict} the market",
            f"ATS record (all picks)        : {metrics['ats_win_pct']*100:.1f}% "
            f"on {metrics['ats_n']:,} games (break-even 52.4%)",
        ]
        for label, t in (metrics.get("ats_by_tier") or {}).items():
            lines.append(f"  {label:<10} (edge >= {t['threshold']:.1f}) : "
                         f"{t['ats_win_pct']*100:.1f}% on {t['n']:,}")
    return "\n".join(lines)


def save_metrics(metrics: dict, path) -> None:
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
