"""
Triple-Barrier Labeling + Meta-Labeling — mlfinlab replacement.

López de Prado ("Advances in Financial Machine Learning", Ch. 3 & 4):

Triple-Barrier Labeling
    For each CUSUM event, define three barriers:
      Upper (+1):    profit-taking at +pt_sl × daily_vol (long wins)
      Lower (-1):    stop-loss at −pt_sl × daily_vol (long loses)
      Vertical (0):  time expiry after max_hold bars (inconclusive)
    The barrier touched first determines the label.

Meta-Labeling (Ch. 4)
    A secondary binary classifier (RandomForest by default) is trained to
    predict whether the primary signal will be correct.  The meta-label is:
        1 → take the primary bet (expected to win)
        0 → skip this signal (primary model is likely wrong)
    This reduces false-positive rate without changing the primary strategy logic.

Usage:
    from signals.cusum import cusum_filter, dynamic_threshold
    from signals.triple_barrier import label_events, MetaLabeler

    events = cusum_filter(close, h)
    labels = label_events(close, events, pt_sl=1.0, max_hold=5)

    ml = MetaLabeler()
    ml.fit(feature_df, labels)
    meta = ml.predict(feature_df)   # {timestamp: 0 or 1}
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Triple-barrier labeling
# ---------------------------------------------------------------------------

def label_events(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: float = 1.0,
    max_hold: int = 10,
    vol_window: int = 20,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to a set of CUSUM event timestamps.

    For each event, barriers are placed at:
      Upper (take-profit):  close[t] × (1 + pt_sl × daily_vol[t])
      Lower (stop-loss):    close[t] × (1 − pt_sl × daily_vol[t])
      Vertical (time-out):  close[t + max_hold]

    Args:
        close:      Close price series with DatetimeIndex.
        events:     CUSUM event timestamps (from cusum_filter).
        pt_sl:      Profit/stop multiplier in units of daily vol (default 1.0).
        max_hold:   Maximum holding bars before forced close (default 10).
        vol_window: Rolling window for daily volatility estimate (default 20).
        min_ret:    Minimum absolute return to count as a win/loss rather than 0
                    (default 0.0 — any non-zero move counts).

    Returns:
        DataFrame indexed by event timestamp with columns:
            t1      — actual close timestamp (when a barrier was hit)
            ret     — return achieved between event and t1
            label   — {+1: upper hit, -1: lower hit, 0: vertical/time}
            barrier — {'upper', 'lower', 'vertical'}
    """
    daily_vol = (
        close.pct_change()
             .rolling(vol_window)
             .std()
    )
    idx = close.index

    rows = []
    for t0 in events:
        if t0 not in idx:
            continue
        loc0  = idx.get_loc(t0)
        p0    = float(close.iloc[loc0])
        vol0  = float(daily_vol.iloc[loc0]) if not np.isnan(daily_vol.iloc[loc0]) else 0.01

        upper  = p0 * (1 + pt_sl * vol0)
        lower  = p0 * (1 - pt_sl * vol0)
        t_end  = min(loc0 + max_hold, len(idx) - 1)

        label   = 0
        barrier = "vertical"
        t1      = idx[t_end]
        ret     = 0.0

        for j in range(loc0 + 1, t_end + 1):
            p = float(close.iloc[j])
            r = (p - p0) / p0

            if p >= upper:
                label, barrier, t1, ret = 1,  "upper",    idx[j], r; break
            elif p <= lower:
                label, barrier, t1, ret = -1, "lower",    idx[j], r; break
        else:
            # Vertical barrier: use return at max_hold
            ret = (float(close.iloc[t_end]) - p0) / p0
            if abs(ret) >= min_ret:
                label = 1 if ret > 0 else -1

        rows.append({"t0": t0, "t1": t1, "ret": round(ret, 6),
                     "label": label, "barrier": barrier})

    if not rows:
        return pd.DataFrame(columns=["t1", "ret", "label", "barrier"])

    df = pd.DataFrame(rows).set_index("t0")
    df.index.name = "t0"
    return df


# ---------------------------------------------------------------------------
# Meta-labeling
# ---------------------------------------------------------------------------

class MetaLabeler:
    """
    Secondary binary classifier that predicts whether the primary signal
    will be correct.

    The meta-label is 1 if the primary model is expected to be right,
    0 if it should be skipped.  Fitting this on top of the primary labels
    (from triple-barrier) reduces the false-positive rate.

    Uses RandomForestClassifier by default (robust, no hyperparameter tuning
    needed for a first-pass model).
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 4):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        primary_agrees: Optional[pd.Series] = None,
    ) -> "MetaLabeler":
        """
        Train the meta-labeler.

        Args:
            features:       Feature DataFrame (rows = event timestamps, cols = features).
                            Rows not present in `labels` are ignored.
            labels:         Output of label_events() — must contain a 'label' column.
            primary_agrees: Optional boolean Series aligned to labels.index:
                            True where primary signal agrees with barrier direction.
                            If None, all events are treated as primary-agrees=True.

        Returns:
            self (for chaining).
        """
        aligned = features.reindex(labels.index).dropna()
        y_raw   = labels.loc[aligned.index, "label"]

        # Meta-label: 1 if the trade was profitable (label == +1), else 0
        y = (y_raw == 1).astype(int)

        # Optionally filter to events where the primary signal fires
        if primary_agrees is not None:
            mask = primary_agrees.reindex(aligned.index).fillna(False)
            aligned = aligned[mask]
            y = y[mask]

        if len(aligned) < 10:
            print("  [MetaLabeler] Not enough samples to train — skipping.")
            return self

        self.model.fit(aligned.values, y.values)
        self._fitted = True
        print(f"  [MetaLabeler] Trained on {len(aligned)} samples "
              f"(positive rate: {y.mean():.1%})")
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict meta-labels for new events.

        Args:
            features: Feature DataFrame (rows = event timestamps).

        Returns:
            pd.Series of {0, 1} indexed by event timestamp.
            All 1s if the model is not yet fitted (pass-through).
        """
        if not self._fitted:
            return pd.Series(1, index=features.index, dtype=int)

        proba = self.model.predict_proba(features.values)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        return pd.Series(pred, index=features.index, name="meta_label")

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Return probability of meta-label = 1."""
        if not self._fitted:
            return pd.Series(1.0, index=features.index)
        proba = self.model.predict_proba(features.values)[:, 1]
        return pd.Series(proba, index=features.index, name="meta_proba")

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        """Return feature importances sorted descending."""
        if not self._fitted:
            return pd.Series(dtype=float)
        imp = pd.Series(self.model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
