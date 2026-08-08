"""
The real fix for the "no volatility edge" problem: price options using
historical VIX (real market implied vol) instead of realized vol, and
compare an unconditional short-straddle seller against a VRP-conditional
one that only sells when VIX is rich relative to trailing realized vol.

REQUIRES INTERNET ACCESS (Yahoo Finance) -- run this on your own machine,
not in a sandboxed environment. Falls back to a synthetic "rich IV" series
if VIX data isn't reachable, purely so the script always runs end-to-end,
but the whole point of this script is to use REAL VIX history -- if you see
the fallback message, fix your connectivity before trusting the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from backtest import (
    BacktestEngine, DeltaHedgedStraddle, VRPConditionalStraddle,
    compute_stats, print_stats,
)
from options_pricer.data import get_vix_history

TICKER = "SPY"
PERIOD = "5y"


def load_data():
    try:
        spy_hist = yf.Ticker(TICKER).history(period=PERIOD)
        if spy_hist.empty:
            raise RuntimeError("empty SPY history")
        spy_prices = spy_hist["Close"]
        spy_prices.index = spy_prices.index.tz_localize(None)

        vix = get_vix_history(period=PERIOD)

        # align to common trading days
        common_idx = spy_prices.index.intersection(vix.index)
        spy_prices = spy_prices.loc[common_idx].sort_index()
        vix = vix.loc[common_idx].sort_index()
        return spy_prices, vix, "real SPY + VIX data"
    except Exception as e:
        print(f"[info] Live data unavailable ({e}); using synthetic fallback "
              f"(NOTE: this defeats the purpose of this script -- fix connectivity to get a real result).")
        np.random.seed(11)
        n_days = 500
        dt = 1 / 252
        mu, sigma_true = 0.08, 0.16
        rets = np.random.normal((mu - 0.5 * sigma_true ** 2) * dt, sigma_true * np.sqrt(dt), n_days)
        prices = 100 * np.exp(np.cumsum(rets))
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
        spy_prices = pd.Series(prices, index=dates)
        # synthetic VIX-like series: realized vol * a premium factor + noise, floored sensibly
        log_ret = np.log(spy_prices / spy_prices.shift(1))
        rv = log_ret.rolling(20).std() * np.sqrt(252)
        synthetic_vix = (rv * 1.20 + np.random.normal(0, 0.01, n_days)).clip(lower=0.08).dropna()
        return spy_prices, synthetic_vix, "synthetic fallback"


def main():
    spy_prices, vix, source = load_data()
    print(f"Data source: {source}")
    print(f"Range: {spy_prices.index[0].date()} to {spy_prices.index[-1].date()}, "
          f"{len(spy_prices)} trading days")

    # descriptive check: how often is VIX actually rich vs trailing realized vol?
    log_ret = np.log(spy_prices / spy_prices.shift(1))
    realized_vol = log_ret.rolling(20).std() * np.sqrt(252)
    vrp = (vix - realized_vol).dropna()
    print(f"\nVIX - trailing realized vol: mean={vrp.mean()*100:.2f} vol pts, "
          f"median={vrp.median()*100:.2f}, pct of days VIX > RV: {(vrp > 0).mean()*100:.1f}%")

    initial_cash = spy_prices.iloc[0] * 100 * 1.5

    strategies = [
        (DeltaHedgedStraddle(n_contracts=1, dte_days=30), "Unconditional Short Straddle (priced at real VIX)"),
        (VRPConditionalStraddle(n_contracts=1, dte_days=30, vrp_threshold=0.03),
         "VRP-Conditional Short Straddle (only sells when VIX rich by 3+ vol pts)"),
    ]

    curves = {}
    for strat, name in strategies:
        engine = BacktestEngine(spy_prices, strat, initial_cash=initial_cash,
                                 external_iv_series=vix)
        curve = engine.run()
        curves[name] = curve["equity"]
        stats = compute_stats(curve["equity"])
        print_stats(name, stats)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, curve in curves.items():
        ax.plot(curve.index, curve.values, label=name)
    ax.set_title(f"Volatility Risk Premium Strategies -- {source}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("vrp_backtest_comparison.png", dpi=150)
    print("\nSaved vrp_backtest_comparison.png")


if __name__ == "__main__":
    main()
