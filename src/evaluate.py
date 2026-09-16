"""Evaluation: signal quality, cost-aware backtest, and plots.

Signal metrics
  ic_pooled   Spearman corr(prediction, realised return) over all test rows.
  cs_ic_mean  mean of the daily cross-sectional Spearman IC; cs_ic_t = its t-stat.
  hit_rate    P(sign(pred) == sign(realised)), Wilson 95% CI, binomial p vs 0.5.
  up_rate     share of up days = hit rate of an "always long" rule (the real bar).
  r2_oos      1 - SSE(model) / SSE(zero forecast)   (Campbell-Thompson OOS R^2).

Backtest (daily rebalanced at the close, held one day)
  sign  : w_i = sign(pred_i) / N   (equal-weight long/short by predicted sign)
  rank  : w_i ∝ rank(pred_i) - mean rank, scaled so sum |w_i| = 1 (dollar-neutral)
  cost  : cost_bps * turnover, turnover_t = sum_i |w_t,i - w_t-1,i|
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANN = 252


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def signal_metrics(pred, y, dates) -> dict:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    out = {"n": int(len(y))}
    const = np.std(pred) == 0

    out["ic_pooled"] = np.nan if const else float(spearmanr(pred, y)[0])

    if const:
        out["cs_ic_mean"], out["cs_ic_t"] = np.nan, np.nan
    else:
        tmp = pd.DataFrame({"p": pred, "y": y, "d": np.asarray(dates)})
        ics = []
        for _, g in tmp.groupby("d"):
            if len(g) >= 5 and g["p"].std() > 0:
                ics.append(spearmanr(g["p"], g["y"])[0])
        ics = np.array([v for v in ics if not np.isnan(v)])
        out["cs_ic_mean"] = float(ics.mean()) if len(ics) else np.nan
        out["cs_ic_t"] = float(ics.mean() / ics.std() * np.sqrt(len(ics))) if len(ics) > 1 else np.nan

    nz = y != 0
    hits = (y[nz] > 0) if const else (np.sign(pred[nz]) == np.sign(y[nz]))
    k, n = int(hits.sum()), int(nz.sum())
    out["hit_rate"] = k / n if n else np.nan
    out["hit_ci_lo"], out["hit_ci_hi"] = wilson_ci(k, n)
    out["hit_p_vs_50"] = binomtest(k, n, 0.5).pvalue if n else np.nan
    out["up_rate"] = float((y[nz] > 0).mean()) if n else np.nan
    # the honest bar: does the model beat the unconditional 'always long' hit rate?
    out["hit_p_vs_uprate"] = binomtest(k, n, out["up_rate"]).pvalue if n else np.nan
    out["share_long"] = float((pred > 0).mean())
    out["r2_oos"] = float(1 - np.sum((y - pred) ** 2) / np.sum(y ** 2))
    return out


# ------------------------------------------------------------------ backtest
def make_weights(pred_df: pd.DataFrame, model: str, strategy: str) -> pd.DataFrame:
    p = pred_df[model].unstack("ticker")
    if strategy == "sign":
        w = np.sign(p)
        n = w.abs().sum(axis=1)
        w = w.div(n.where(n > 0, np.nan), axis=0)
    elif strategy == "rank":
        r = p.rank(axis=1)
        r = r.sub(r.mean(axis=1), axis=0)
        g = r.abs().sum(axis=1)
        w = r.div(g.where(g > 0, np.nan), axis=0)
    else:
        raise ValueError(strategy)
    return w.fillna(0.0)


def backtest(pred_df: pd.DataFrame, model: str, strategy: str, cost_bps: float) -> pd.DataFrame:
    w = make_weights(pred_df, model, strategy)
    R = pred_df["ret_fwd"].unstack("ticker").reindex(index=w.index, columns=w.columns).fillna(0.0)
    gross = (w * R).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})


def buy_and_hold(pred_df: pd.DataFrame) -> pd.Series:
    """Equal-weight, daily-rebalanced long-only universe (transaction costs ignored)."""
    return pred_df["ret_fwd"].unstack("ticker").mean(axis=1)


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann_ret = r.mean() * ANN
    ann_vol = r.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {"ann_ret": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
            "max_dd": float(dd.min()), "total_ret": float(eq.iloc[-1] - 1)}


def backtest_table(pred_df, models, strategies=("sign", "rank"), costs=(0, 2, 5, 10, 20)) -> pd.DataFrame:
    rows = []
    bh_ret = buy_and_hold(pred_df)
    for m in models:
        for s in strategies:
            w = make_weights(pred_df, m, s)
            share_long = float((w > 0).sum(axis=1).sum() / max((w != 0).sum(axis=1).sum(), 1))
            for c in costs:
                bt = backtest(pred_df, m, s, c)
                st = perf_stats(bt["net"])
                st.update({"model": m, "strategy": s, "cost_bps": c,
                           "avg_turnover": float(bt["turnover"].mean()),
                           "corr_bh": float(bt["net"].corr(bh_ret)), "share_long": share_long})
                rows.append(st)
    bh = perf_stats(buy_and_hold(pred_df))
    bh.update({"model": "buy_and_hold_ew", "strategy": "long", "cost_bps": 0, "avg_turnover": np.nan,
               "corr_bh": 1.0, "share_long": 1.0})
    rows.append(bh)
    cols = ["model", "strategy", "cost_bps", "ann_ret", "ann_vol", "sharpe", "max_dd", "total_ret",
            "avg_turnover", "corr_bh", "share_long"]
    return pd.DataFrame(rows)[cols]


# ------------------------------------------------------------------ plots
def plot_by_year(metrics: pd.DataFrame, models, col, path, title, ylabel, baseline_col=None):
    years = sorted(metrics["test_year"].unique())
    x = np.arange(len(years))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(models):
        sub = metrics[metrics["model"] == m].set_index("test_year").reindex(years)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, sub[col].to_numpy(), width, label=m)
    if baseline_col is not None:
        base = metrics[metrics["model"] == models[0]].set_index("test_year").reindex(years)[baseline_col]
        ax.plot(x, base.to_numpy(), "k--", lw=1.2, label=baseline_col)
    ax.axhline(0 if col != "hit_rate" else 0.5, color="grey", lw=0.8)
    ax.set_xticks(x, years)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=min(len(models) + 1, 4), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_equity(curves: dict[str, pd.Series], path, title):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for name, r in curves.items():
        eq = (1 + r.dropna()).cumprod()
        ax.plot(eq.index, eq.to_numpy(), label=name, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_drawdown(curves: dict[str, pd.Series], path, title):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, r in curves.items():
        eq = (1 + r.dropna()).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.to_numpy(), label=name, lw=1.1)
    ax.set_ylabel("drawdown")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sharpe_vs_cost(table: pd.DataFrame, models, path, strategy="sign"):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for m in models:
        sub = table[(table["model"] == m) & (table["strategy"] == strategy)].sort_values("cost_bps")
        ax.plot(sub["cost_bps"], sub["sharpe"], marker="o", label=m)
    bh = table[table["model"] == "buy_and_hold_ew"]["sharpe"].iloc[0]
    ax.axhline(bh, color="k", ls="--", lw=1, label="buy & hold (EW)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("one-way transaction cost (bps of traded notional)")
    ax.set_ylabel("annualised Sharpe (net)")
    ax.set_title(f"Sharpe vs. cost — {strategy} long/short, daily rebalance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(imp: pd.Series, path, title="LightGBM gain importance (avg over folds)"):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.to_numpy())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
