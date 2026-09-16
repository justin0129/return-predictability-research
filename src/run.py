"""Run the walk-forward experiment end to end.

    python -m src.run                                   # reproducible v1 (GitHub CSV, 20 stocks)
    python -m src.run --source yfinance --tickers AAPL MSFT ... --start 2005-01-01 --test-years 2015-2025

Outputs (in --out, default results/):
    predictions.csv        every out-of-sample prediction, per model, with realised return
    fold_log.csv           chosen hyper-parameters + validation IC per fold and model
    metrics_by_year.csv    IC / hit-rate / OOS-R2 per model and test year (+ pooled "ALL")
    metrics_by_ticker.csv  pooled IC / hit-rate per model and ticker
    backtest.csv           net performance per model x strategy x cost level
    plots/*.png
    summary.md             tables ready to paste into the README
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

from . import evaluate as ev
from .data import load_prices
from .features import feature_columns, make_features
from .models import LGBMModel, LSTMModel, RidgeModel, RuleModel, Scaler, ZeroModel, rows_on, spearman_ic
from .validation import walk_forward_folds

ML = ["ridge", "lgbm", "lstm"]


def parse_years(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def build_model(name: str, args):
    return {
        "zero": ZeroModel(),
        "rev1": RuleModel("rev1", "r1", sign=-1.0),      # 1-day reversal rule
        "mom20": RuleModel("mom20", "mom_20", sign=1.0),  # 20-day momentum rule
        "ridge": RidgeModel(),
        "lgbm": LGBMModel(seed=args.seed),
        "lstm": LSTMModel(seq_len=args.seq_len, units=tuple(args.units), epochs=args.epochs,
                          seed=args.seed, verbose=args.verbose),
    }[name]


def fmt(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    """Minimal markdown table writer (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row):
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="github", choices=["github", "yfinance"])
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--test-years", default="2006-2017")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=["zero", "rev1", "mom20", "ridge", "lgbm", "lstm"])
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--units", type=int, nargs="*", default=[32, 16])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t0 = time.time()
    px = load_prices(args.source, args.tickers, args.start, args.end)
    print(f"prices: {px.shape[0]} days x {px.shape[1]} tickers, "
          f"{px.index[0].date()} .. {px.index[-1].date()}", flush=True)
    df = make_features(px, args.horizon)
    feats = feature_columns(df)
    print(f"feature rows: {len(df):,}   features ({len(feats)}): {feats}", flush=True)

    folds = walk_forward_folds(df.index.get_level_values("date").unique(),
                               parse_years(args.test_years), purge=args.horizon)
    print("test years:", [f.test_year for f in folds], flush=True)

    out_dir = args.out
    fold_dir = os.path.join(out_dir, "folds")
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(fold_dir, exist_ok=True)

    preds, fold_log, importances = [], [], []
    for fold in folds:
        tf0 = time.time()
        ck = os.path.join(fold_dir, f"{fold.test_year}.csv")
        ck_log = os.path.join(fold_dir, f"{fold.test_year}_log.json")
        if os.path.exists(ck) and os.path.exists(ck_log):      # resume support
            saved = pd.read_csv(ck, index_col=["date", "ticker"], parse_dates=["date"])
            if all(m in saved.columns for m in args.models):
                preds.append(saved[["target", "ret_fwd", "test_year"] + args.models])
                with open(ck_log) as fh:
                    log = json.load(fh)
                fold_log.extend(log["fold_log"])
                if "importance" in log:
                    importances.append(pd.Series(log["importance"]))
                print(f"fold {fold.test_year} loaded from checkpoint", flush=True)
                continue
        tr = rows_on(df, fold.train_dates)
        scaler = Scaler().fit(tr[feats])          # training-fold statistics only
        dfs = df.copy()
        dfs[feats] = scaler.transform(df[feats])
        te = rows_on(dfs, fold.test_dates)
        out = te[["target"]].copy()
        out["ret_fwd"] = np.expm1(out["target"])  # simple forward return for P&L
        out["test_year"] = fold.test_year
        for name in args.models:
            m = build_model(name, args).fit(dfs, feats, fold.train_dates, fold.val_dates)
            p = m.predict(dfs, feats, fold.test_dates)
            out[name] = p
            rec = {"test_year": fold.test_year, "model": name,
                   "test_ic": round(spearman_ic(p, out["target"]), 4), **m.info}
            fold_log.append(rec)
            if name == "lgbm":
                importances.append(m.importance)
            print(f"  {fold.test_year} {name:6s} test_ic={rec['test_ic']:+.4f}  {json.dumps(m.info)}", flush=True)
        preds.append(out)
        out.to_csv(ck)
        log = {"fold_log": [r for r in fold_log if r["test_year"] == fold.test_year]}
        if importances and "lgbm" in args.models:
            log["importance"] = {k: float(v) for k, v in importances[-1].items()}
        with open(ck_log, "w") as fh:
            json.dump(log, fh)
        print(f"fold {fold.test_year} done in {time.time() - tf0:.0f}s  "
              f"(train {fold.train_dates[0].year}-{fold.train_dates[-1].year}, val {fold.val_dates[0].year}, "
              f"n_train={len(tr):,}, n_test={len(te):,})", flush=True)

    pred_df = pd.concat(preds).sort_index()
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"))
    pd.DataFrame(fold_log).to_csv(os.path.join(out_dir, "fold_log.csv"), index=False)

    # ---------------- signal metrics
    rows = []
    for name in args.models:
        for ty, g in pred_df.groupby("test_year"):
            rows.append({"model": name, "test_year": int(ty),
                         **ev.signal_metrics(g[name], g["target"], g.index.get_level_values("date"))})
        rows.append({"model": name, "test_year": "ALL",
                     **ev.signal_metrics(pred_df[name], pred_df["target"], pred_df.index.get_level_values("date"))})
    metrics = pd.DataFrame(rows)
    # OOS R^2 is only meaningful for models that predict a return level (rules predict a z-score)
    metrics.loc[metrics["model"].isin(["rev1", "mom20"]), "r2_oos"] = np.nan
    metrics.to_csv(os.path.join(out_dir, "metrics_by_year.csv"), index=False)

    rows = []
    for name in args.models:
        for tk, g in pred_df.groupby(level="ticker"):
            rows.append({"model": name, "ticker": tk,
                         **ev.signal_metrics(g[name], g["target"], g.index.get_level_values("date"))})
    by_ticker = pd.DataFrame(rows)
    by_ticker.to_csv(os.path.join(out_dir, "metrics_by_ticker.csv"), index=False)

    # ---------------- backtests
    bt_models = [m for m in args.models if m != "zero"]
    table = ev.backtest_table(pred_df, bt_models)
    table.to_csv(os.path.join(out_dir, "backtest.csv"), index=False)

    # ---------------- plots
    ml = [m for m in ML if m in args.models]
    yr = metrics[metrics["test_year"] != "ALL"].copy()
    yr["test_year"] = yr["test_year"].astype(int)
    P = lambda f: os.path.join(out_dir, "plots", f)  # noqa: E731
    ev.plot_by_year(yr, ml, "ic_pooled", P("ic_by_year.png"),
                    "Out-of-sample Spearman IC by test year", "IC")
    ev.plot_by_year(yr, ml, "hit_rate", P("hit_rate_by_year.png"),
                    "Directional accuracy by test year (dashed = share of up days, i.e. 'always long')",
                    "hit rate", baseline_col="up_rate")
    curves = {f"{m} (sign L/S, net 5 bps)": ev.backtest(pred_df, m, "sign", 5)["net"] for m in bt_models}
    curves["buy & hold (EW)"] = ev.buy_and_hold(pred_df)
    ev.plot_equity(curves, P("equity_net_5bps.png"),
                   "Cumulative return: sign long/short net of 5 bps vs. equal-weight buy & hold")
    ev.plot_sharpe_vs_cost(table, bt_models, P("sharpe_vs_cost.png"))
    ev.plot_sharpe_vs_cost(table, bt_models, P("sharpe_vs_cost_rank.png"), strategy="rank")
    rank_curves = {f"{m} (rank L/S, net 5 bps)": ev.backtest(pred_df, m, "rank", 5)["net"] for m in bt_models}
    rank_curves["buy & hold (EW)"] = curves["buy & hold (EW)"]
    ev.plot_equity(rank_curves, P("equity_rank_net_5bps.png"),
                   "Cumulative return: dollar-neutral rank long/short net of 5 bps vs. equal-weight buy & hold")
    sel = table[(table["strategy"] == "sign") & (table["cost_bps"] == 5) & (table["model"].isin(ml))]
    best = sel.sort_values("sharpe").iloc[-1]["model"]
    ev.plot_drawdown({f"{best} (sign L/S, net 5 bps)": curves[f"{best} (sign L/S, net 5 bps)"],
                      "buy & hold (EW)": curves["buy & hold (EW)"]},
                     P("drawdown.png"), f"Drawdown: {best} long/short (net 5 bps) vs. buy & hold")
    if importances:
        imp = pd.concat(importances, axis=1).mean(axis=1)
        imp = imp / imp.sum()
        ev.plot_importance(imp, P("lgbm_importance.png"))

    # ---------------- summary markdown
    pooled = metrics[metrics["test_year"] == "ALL"].drop(columns=["test_year"])
    pooled = pooled[["model", "n", "ic_pooled", "cs_ic_mean", "cs_ic_t", "hit_rate", "hit_ci_lo",
                     "hit_ci_hi", "up_rate", "hit_p_vs_uprate", "share_long", "r2_oos"]]
    ic_wide = yr.pivot(index="test_year", columns="model", values="ic_pooled")[[m for m in args.models if m != "zero"]]
    hit_wide = yr.pivot(index="test_year", columns="model", values="hit_rate")[args.models]
    bt_sign = table[(table["strategy"] == "sign") & (table["cost_bps"].isin([0, 5, 10]))]
    bt_rank = table[(table["strategy"] == "rank") & (table["cost_bps"].isin([0, 5, 10]))]
    bh = table[table["model"] == "buy_and_hold_ew"]
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write(f"Run: source={args.source}, tickers={px.shape[1]}, {px.index[0].date()}..{px.index[-1].date()}, "
                 f"horizon={args.horizon}, test years {folds[0].test_year}-{folds[-1].test_year}, "
                 f"seq_len={args.seq_len}, units={args.units}, seed={args.seed}\n\n")
        fh.write("## Pooled out-of-sample signal metrics\n\n" + fmt(pooled) + "\n\n")
        fh.write("## Pooled IC by test year\n\n" + fmt(ic_wide.reset_index(), "{:+.3f}") + "\n\n")
        fh.write("## Hit rate by test year\n\n" + fmt(hit_wide.reset_index(), "{:.3f}") + "\n\n")
        fh.write("## Backtest — sign long/short, daily rebalance\n\n" + fmt(pd.concat([bt_sign, bh]), "{:.3f}") + "\n\n")
        fh.write("## Backtest — rank (dollar-neutral) long/short\n\n" + fmt(pd.concat([bt_rank, bh]), "{:.3f}") + "\n\n")
        fh.write("## Chosen hyper-parameters per fold\n\n" + fmt(pd.DataFrame(fold_log).fillna(""), "{:.4g}") + "\n")
    print(f"\nall done in {(time.time() - t0) / 60:.1f} min -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
