"""Mechanical look-ahead checks.

1. Truncation: features computed on prices up to date T must equal the same rows
   of features computed on the full history (nothing after T can matter).
2. Perturbation: scrambling every price after T must leave features on dates <= T
   unchanged, while the target on the last pre-T days *does* change (it is the only
   forward-looking column, by design).
3. Purge: in every walk-forward fold, the last train/validation date plus the
   horizon is strictly before the first test date.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import feature_columns, make_features
from src.validation import walk_forward_folds


@pytest.fixture(scope="module")
def px():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2000-01-03", "2012-12-31")
    n, k = len(dates), 6
    logp = np.cumsum(rng.normal(0.0003, 0.02, size=(n, k)), axis=0) + 3
    px = pd.DataFrame(np.exp(logp), index=dates, columns=[f"T{i}" for i in range(k)])
    px.iloc[:400, 5] = np.nan          # a late-listed ticker
    px.index.name = "date"
    return px


def test_truncation_invariance(px):
    h = 1
    full = make_features(px, h)
    cutoff = px.index[len(px) // 2]
    trunc = make_features(px.loc[:cutoff], h)
    feats = feature_columns(full)
    pd.testing.assert_frame_equal(full.loc[trunc.index, feats], trunc[feats])


def test_future_perturbation(px):
    h = 1
    cutoff = px.index[len(px) // 2]
    base = make_features(px, h)
    rng = np.random.default_rng(1)
    px2 = px.copy()
    after = px2.index > cutoff
    px2.loc[after] = px2.loc[after] * np.exp(rng.normal(0, 0.5, size=(after.sum(), px.shape[1])))
    pert = make_features(px2, h)
    feats = feature_columns(base)
    past = base.index[base.index.get_level_values("date") <= cutoff]
    pd.testing.assert_frame_equal(base.loc[past, feats], pert.loc[past, feats])
    # the target on the last day before the cutoff must change (it looks h days ahead)
    last = base.index[base.index.get_level_values("date") == cutoff]
    assert not np.allclose(base.loc[last, "target"], pert.loc[last, "target"])


def test_purge_gap(px):
    for h in (1, 5):
        df = make_features(px, h)
        dates = df.index.get_level_values("date").unique()
        folds = walk_forward_folds(dates, [2008, 2009, 2010], purge=h, min_train_years=3)
        assert folds, "no folds built"
        for f in folds:
            # a sample on date d uses prices up to d + h; that must stay before the test period
            all_dates = pd.DatetimeIndex(sorted(dates))
            pos = all_dates.get_loc(f.val_dates[-1])
            assert all_dates[pos + h] < f.test_dates[0]
            pos = all_dates.get_loc(f.train_dates[-1])
            assert all_dates[pos + h] < f.val_dates[0]
            assert f.train_dates[-1] < f.val_dates[0] < f.test_dates[0]
