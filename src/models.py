"""Model wrappers with one interface.

    model.fit(df, feats, train_dates, val_dates)
    model.predict(df, feats, dates) -> np.ndarray aligned with rows of df on `dates`

``df`` is the full long feature frame (already standardised with training-fold
statistics), so that the LSTM can look back before the first test date.
Hyper-parameters are chosen on the validation year only; the test year is
never touched during fitting.  Targets are trained in percent (x100) purely for
numerical convenience and rescaled on prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

Y_SCALE = 100.0  # train on percent log-returns


def rows_on(df: pd.DataFrame, dates) -> pd.DataFrame:
    return df[df.index.get_level_values("date").isin(dates)]


def spearman_ic(pred, y) -> float:
    pred = np.asarray(pred, dtype=float)
    if np.nanstd(pred) == 0:
        return 0.0
    return float(spearmanr(pred, y)[0])


class Scaler:
    """Per-feature z-score with statistics from the training rows, clipped."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip

    def fit(self, X: pd.DataFrame):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = sd.where(sd > 1e-12, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return ((X - self.mu) / self.sd).clip(-self.clip, self.clip)


# --------------------------------------------------------------------------- baselines
class ZeroModel:
    """Random-walk benchmark: expected return = 0."""
    name = "zero"
    info: dict = {}

    def fit(self, df, feats, train_dates, val_dates):
        return self

    def predict(self, df, feats, dates):
        return np.zeros(len(rows_on(df, dates)))


class RuleModel:
    """One-line, non-ML rule: prediction = sign * (a single feature)."""

    def __init__(self, name: str, feature: str, sign: float = 1.0):
        self.name, self.feature, self.sign, self.info = name, feature, sign, {}

    def fit(self, df, feats, train_dates, val_dates):
        return self

    def predict(self, df, feats, dates):
        return self.sign * rows_on(df, dates)[self.feature].to_numpy(dtype=float)


# --------------------------------------------------------------------------- ridge
class RidgeModel:
    name = "ridge"

    def __init__(self, alphas=(1, 10, 100, 1e3, 1e4, 1e5)):
        self.alphas = alphas

    def fit(self, df, feats, train_dates, val_dates):
        tr, va = rows_on(df, train_dates), rows_on(df, val_dates)
        best = None
        for a in self.alphas:
            m = Ridge(alpha=a).fit(tr[feats], tr["target"] * Y_SCALE)
            ic = spearman_ic(m.predict(va[feats]), va["target"])
            if best is None or ic > best[0]:
                best = (ic, a, m)
        self.val_ic, self.alpha, self.model = best
        self.info = {"alpha": self.alpha, "val_ic": round(self.val_ic, 4)}
        return self

    def predict(self, df, feats, dates):
        return self.model.predict(rows_on(df, dates)[feats]) / Y_SCALE


# --------------------------------------------------------------------------- lightgbm
class LGBMModel:
    name = "lgbm"

    def __init__(self, leaves=(7, 31), lr=0.05, n_estimators=1000, es_rounds=50, seed=0):
        self.leaves, self.lr, self.n_estimators, self.es_rounds, self.seed = leaves, lr, n_estimators, es_rounds, seed

    def fit(self, df, feats, train_dates, val_dates):
        import lightgbm as lgb

        tr, va = rows_on(df, train_dates), rows_on(df, val_dates)
        best = None
        for nl in self.leaves:
            m = lgb.LGBMRegressor(
                num_leaves=nl, learning_rate=self.lr, n_estimators=self.n_estimators,
                min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=5.0, random_state=self.seed, verbose=-1, n_jobs=1,
            )
            cb = [lgb.early_stopping(self.es_rounds, verbose=False)]
            try:  # lightgbm >= 4.7
                m.fit(tr[feats], tr["target"] * Y_SCALE,
                      eval_X=[va[feats]], eval_y=[va["target"] * Y_SCALE], callbacks=cb)
            except TypeError:  # older lightgbm
                m.fit(tr[feats], tr["target"] * Y_SCALE,
                      eval_set=[(va[feats], va["target"] * Y_SCALE)], callbacks=cb)
            ic = spearman_ic(m.predict(va[feats]), va["target"])
            if best is None or ic > best[0]:
                best = (ic, nl, m)
        self.val_ic, self.num_leaves, self.model = best
        self.info = {"num_leaves": self.num_leaves, "best_iter": int(self.model.best_iteration_),
                     "val_ic": round(self.val_ic, 4)}
        self.importance = pd.Series(self.model.booster_.feature_importance("gain"), index=feats)
        return self

    def predict(self, df, feats, dates):
        return self.model.predict(rows_on(df, dates)[feats]) / Y_SCALE


# --------------------------------------------------------------------------- lstm
class LSTMModel:
    """Stacked LSTM over the last `seq_len` days of the (standardised) feature vector.

    Sequences are built per ticker from contiguous rows; a row's sequence ends on
    that row's date, so a test-date sequence only ever contains past features.
    Rows without `seq_len` days of history get prediction 0 (no view)."""
    name = "lstm"

    def __init__(self, seq_len=20, units=(32, 16), dropout=0.2, lr=1e-3, epochs=12,
                 batch=512, patience=2, seed=0, verbose=0):
        self.seq_len, self.units, self.dropout, self.lr = seq_len, units, dropout, lr
        self.epochs, self.batch, self.patience, self.seed, self.verbose = epochs, batch, patience, seed, verbose

    def _sequences(self, df, feats, dates):
        want = pd.DatetimeIndex(dates)
        Xs, ys, idx = [], [], []
        for tkr, g in df.groupby(level="ticker", sort=True):
            g = g.droplevel("ticker")
            if len(g) <= self.seq_len:
                continue
            F = g[feats].to_numpy(np.float32)
            y = g["target"].to_numpy(np.float32)
            W = sliding_window_view(F, (self.seq_len, F.shape[1]))[:, 0]   # (T-L+1, L, nfeat)
            end = g.index[self.seq_len - 1:]
            m = end.isin(want)
            if not m.any():
                continue
            Xs.append(W[m])
            ys.append(y[self.seq_len - 1:][m])
            idx.append(pd.MultiIndex.from_arrays([end[m], np.repeat(tkr, int(m.sum()))],
                                                 names=["date", "ticker"]))
        index = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
        return np.concatenate(Xs), np.concatenate(ys), index

    def _build(self, nfeat):
        import keras
        from keras import layers

        inp = keras.Input((self.seq_len, nfeat))
        x = inp
        for i, u in enumerate(self.units):
            x = layers.LSTM(u, return_sequences=(i < len(self.units) - 1))(x)
            x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1)(x)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0), loss="mse")
        return model

    def fit(self, df, feats, train_dates, val_dates):
        import keras

        keras.utils.set_random_seed(self.seed)
        Xtr, ytr, _ = self._sequences(df, feats, train_dates)
        Xva, yva, _ = self._sequences(df, feats, val_dates)
        self.model = self._build(Xtr.shape[2])
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                           restore_best_weights=True)
        hist = self.model.fit(Xtr, ytr * Y_SCALE, validation_data=(Xva, yva * Y_SCALE),
                              epochs=self.epochs, batch_size=self.batch, shuffle=True,
                              callbacks=[es], verbose=self.verbose)
        pva = self.model.predict(Xva, batch_size=4096, verbose=0).ravel()
        vl = hist.history["val_loss"]
        self.info = {"epochs_run": len(vl), "best_epoch": int(np.argmin(vl)) + 1,
                     "val_ic": round(spearman_ic(pva, yva), 4), "n_train_seq": int(len(Xtr))}
        return self

    def predict(self, df, feats, dates):
        X, _, index = self._sequences(df, feats, dates)
        p = self.model.predict(X, batch_size=4096, verbose=0).ravel() / Y_SCALE
        s = pd.Series(p, index=index)
        target_rows = rows_on(df, dates).index
        return s.reindex(target_rows).fillna(0.0).to_numpy()
