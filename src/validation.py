"""Walk-forward (rolling-origin) validation by calendar year.

For test year T:  train = all years < T-1,  validation = year T-1,  test = year T.

Purge: a sample dated d carries a target that spans d .. d+horizon.  The last
``purge`` (= horizon) days of the validation year therefore have targets that
reach into the test year, and the last ``purge`` days of the training set have
targets that reach into the validation year.  Both are dropped, so the model
and the early-stopping / tuning step never see a test-period price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    test_year: int
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def walk_forward_folds(dates, test_years, purge: int = 1, min_train_years: int = 5) -> list[Fold]:
    dates = pd.DatetimeIndex(sorted(set(dates)))
    years = dates.year
    folds = []
    for ty in test_years:
        vy = ty - 1
        train = dates[years < vy]
        val = dates[years == vy]
        test = dates[years == ty]
        if purge:
            train = train[:-purge]
            val = val[:-purge]
        if len(np.unique(train.year)) < min_train_years or len(val) == 0 or len(test) == 0:
            continue
        folds.append(Fold(ty, train, val, test))
    return folds
