"""Price data loading.

Two sources:
  * ``github``   – a static CSV of daily *adjusted* closes for 20 large US stocks
                   (1989-12-29 .. 2018-04-11) hosted in the PyPortfolioOpt repo.
                   Fully reproducible; used for the v1 run in this repo.
  * ``yfinance`` – live download of adjusted closes for any ticker list, so the
                   study can be re-run through the present day.

Both return a wide DataFrame: index = trading date, columns = tickers, values =
adjusted close. Missing values are allowed (a ticker that was not yet listed).
"""
from __future__ import annotations

import os
import urllib.request

import pandas as pd

GITHUB_URL = (
    "https://raw.githubusercontent.com/robertmartin8/PyPortfolioOpt/"
    "master/tests/resources/stock_prices.csv"
)


def load_prices(
    source: str = "github",
    tickers: list[str] | None = None,
    start: str = "1995-01-01",
    end: str | None = None,
    cache_dir: str = "data",
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)

    if source == "github":
        path = os.path.join(cache_dir, "stock_prices_github.csv")
        if not os.path.exists(path):
            urllib.request.urlretrieve(GITHUB_URL, path)
        px = pd.read_csv(path, parse_dates=["date"], index_col="date")

    elif source == "yfinance":
        import yfinance as yf  # optional dependency

        if not tickers:
            raise ValueError("--tickers is required with --source yfinance")
        path = os.path.join(cache_dir, f"stock_prices_yf_{len(tickers)}.csv")
        if not os.path.exists(path):
            raw = yf.download(list(tickers), start=start, end=end,
                              auto_adjust=True, progress=False)
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            close.to_csv(path)
        px = pd.read_csv(path, parse_dates=[0], index_col=0)

    else:
        raise ValueError(f"unknown source {source!r}")

    px = px.sort_index()
    px.index.name = "date"
    if tickers:
        px = px[[t for t in tickers if t in px.columns]]
    px = px.loc[start:end]
    px = px.dropna(axis=1, how="all")
    return px
