"""
Runs and compares three strategies over a price history:
  - Buy & Hold                (benchmark)
  - Covered Call                (income / vol-selling, static)
  - Delta-Hedged Short Straddle  (income / vol-selling, dynamically hedged)

Uses real historical prices via yfinance if available; falls back to a
synthetic GBM path otherwise (e.g. no internet access), so this script
always runs end-to-end even without live data.

IMPORTANT METHODOLOGY NOTE: options in this backtest are priced with
Black-Scholes using a rolling REALIZED volatility estimate, not real market
implied vol (real historical options chains aren't freely available). This
means these strategies have no built-in "volatility risk premium" edge --
in reality, implied vol tends to run richer than subsequent realized vol,
which is the actual source of edge that makes short-vol strategies
profitable on average over time. This backtest will roughly show that
edge-free, cost-adjusted behavior: see the zero-edge sanity check in the
README. Treat these results as a methodology demonstration, not a claim
that these strategies are profitable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest import BacktestEngine, BuyAndHold, CoveredCall, DeltaHedgedStraddle, compute_stats, print_stats

TICKER = "SPY"
LOOKBACK_DAYS = 500  # roughly 2 years of trading days


def get_price_series() -> tuple[pd.Series, str]:
    """Try real data first, fall back to synthetic GBM. Returns (series, source_label)."""
    try:
        from options_pricer.data import get_spot_price  # noqa: F401 -- just to test connectivity
        import yfinance as yf
        hist = yf.Ticker(TICKER).history(period="2y")
        if hist.empty:
            raise RuntimeError("empty history")
        series = hist["Close"]
        series.index = series.index.tz_localize(None)
        return series, f"real {TICKER} data (yfinance)"
    except Exception as e:
        print(f"[info] Live data unavailable ({e}); falling back to synthetic GBM path.")
        np.random.seed(7)
        n_days = LOOKBACK_DAYS
        dt = 1 / 252
        mu, sigma_true = 0.08, 0.18
        rets = np.random.normal((mu - 0.5 * sigma_true ** 2) * dt, sigma_true * np.sqrt(dt), n_days)
        prices = 100 * np.exp(np.cumsum(rets))
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
        return pd.Series(prices, index=dates), "synthetic GBM path"


def main():
    price_series, source = get_price_series()
    print(f"Price source: {source}")
    print(f"Range: {price_series.index[0].date()} to {price_series.index[-1].date()}, "
          f"{price_series.iloc[0]:.2f} -> {price_series.iloc[-1]:.2f}")

    initial_cash = price_series.iloc[0] * 100 * 1.5  # 100-share position + buffer for margin/premiums

    strategies = [
        (BuyAndHold(shares=100), "Buy & Hold"),
        (CoveredCall(shares=100, otm_pct=0.05, dte_days=30), "Covered Call"),
        (DeltaHedgedStraddle(n_contracts=1, dte_days=30), "Delta-Hedged Short Straddle"),
    ]

    curves = {}
    all_stats = {}
    for strat, name in strategies:
        engine = BacktestEngine(price_series, strat, initial_cash=initial_cash)
        curve = engine.run()
        curves[name] = curve["equity"]
        stats = compute_stats(curve["equity"])
        all_stats[name] = stats
        print_stats(name, stats)

    # ---- Comparison table ----
    summary = pd.DataFrame({
        name: {
            "Total Return": f"{s.total_return*100:.2f}%",
            "CAGR": f"{s.cagr*100:.2f}%",
            "Ann. Vol": f"{s.annualized_vol*100:.2f}%",
            "Sharpe": f"{s.sharpe:.2f}",
            "Max DD": f"{s.max_drawdown*100:.2f}%",
        }
        for name, s in all_stats.items()
    }).T
    print("\n=== Summary ===")
    print(summary.to_string())

    # ---- Equity curve plot ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, curve in curves.items():
        ax.plot(curve.index, curve.values, label=name)
    ax.set_title(f"Strategy Comparison -- {source}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("backtest_comparison.png", dpi=150)
    print("\nSaved backtest_comparison.png")


if __name__ == "__main__":
    main()
