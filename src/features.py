"""Point-in-time feature construction.

Every feature on date t is a function of prices on dates <= t only.  The single
column that looks ahead is ``target`` (the forward ``horizon``-day log return),
which is the quantity being predicted.  ``tests/test_no_lookahead.py`` checks
this property mechanically by truncating / perturbing future prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOM_WINDOWS = (5, 20, 60)
VOL_WINDOWS = (20, 60)


def make_features(px: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return a long DataFrame indexed by (date, ticker) with feature columns
    and a ``target`` column.  Rows with any missing value are dropped."""
    logp = np.log(px)
    r1 = logp.diff()

    f: dict[str, pd.DataFrame] = {"r1": r1}
    for w in MOM_WINDOWS:
        f[f"mom_{w}"] = logp - logp.shift(w)                 # w-day log return ending at t
    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = r1.rolling(w, min_periods=w).std()   # realised vol
    f["r1_z"] = r1 / f["vol_20"]                             # vol-normalised return
    f["vol_ratio"] = f["vol_20"] / f["vol_60"]               # short vs long vol
    for w in (20, 60):
        f[f"ma_gap_{w}"] = px / px.rolling(w, min_periods=w).mean() - 1
    f["hi_gap_60"] = px / px.rolling(60, min_periods=60).max() - 1

    # Equal-weight universe return: known at the close of day t.
    mkt = r1.mean(axis=1)
    f["mkt_r1"] = pd.DataFrame({c: mkt for c in px.columns})
    f["mkt_mom_20"] = pd.DataFrame({c: mkt.rolling(20, min_periods=20).sum() for c in px.columns})
    f["mkt_vol_20"] = pd.DataFrame({c: mkt.rolling(20, min_periods=20).std() for c in px.columns})

    # Target: forward log return from t to t+horizon.  The ONLY forward-looking column.
    f["target"] = logp.shift(-horizon) - logp

    wide = pd.concat(f, axis=1)          # columns: (feature, ticker)
    long = wide.stack(level=1)           # index: (date, ticker)
    long.index.names = ["date", "ticker"]

    dow = long.index.get_level_values("date").dayofweek
    for d in range(4):                   # Mon..Thu dummies (Fri = baseline)
        long[f"dow_{d}"] = (dow == d).astype("float64")

    long = long.dropna().sort_index()
    return long


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "target"]
