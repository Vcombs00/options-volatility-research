"""
Sweeps the VRP entry threshold across several values and compares them
against the unconditional straddle, so you're not hand-editing the number
and re-running one at a time. Also runs a trade-level P&L breakdown on the
default-threshold conditional strategy, which is the more informative test
for understanding WHY a threshold helps or doesn't -- an aggregate Sharpe
ratio hides whether a strategy is winning steadily or winning big a few
times and losing steadily the rest.

REQUIRES INTERNET ACCESS (real SPY + VIX data). Falls back to synthetic
data if unreachable, same as vrp_backtest.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from backtest import BacktestEngine, DeltaHedgedStraddle, VRPConditionalStraddle, compute_stats
from options_pricer.data import get_vix_history

TICKER = "SPY"
PERIOD = "5y"
THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08]  # vol points, e.g. 0.03 = 3 vol points


def load_data():
    try:
        spy_hist = yf.Ticker(TICKER).history(period=PERIOD)
        if spy_hist.empty:
            raise RuntimeError("empty SPY history")
        spy_prices = spy_hist["Close"]
        spy_prices.index = spy_prices.index.tz_localize(None)
        vix = get_vix_history(period=PERIOD)
        common_idx = spy_prices.index.intersection(vix.index)
        return spy_prices.loc[common_idx].sort_index(), vix.loc[common_idx].sort_index(), "real SPY + VIX data"
    except Exception as e:
        print(f"[info] Live data unavailable ({e}); using synthetic fallback.")
        np.random.seed(11)
        n_days = 500
        dt = 1 / 252
        mu, sigma_true = 0.08, 0.16
        rets = np.random.normal((mu - 0.5 * sigma_true ** 2) * dt, sigma_true * np.sqrt(dt), n_days)
        prices = 100 * np.exp(np.cumsum(rets))
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
        spy_prices = pd.Series(prices, index=dates)
        log_ret = np.log(spy_prices / spy_prices.shift(1))
        rv = log_ret.rolling(20).std() * np.sqrt(252)
        synthetic_vix = (rv * 1.20 + np.random.normal(0, 0.01, n_days)).clip(lower=0.08).dropna()
        return spy_prices, synthetic_vix, "synthetic fallback"


def main():
    spy_prices, vix, source = load_data()
    print(f"Data source: {source}\n")

    initial_cash = spy_prices.iloc[0] * 100 * 1.5

    # --- Unconditional baseline ---
    uncond = DeltaHedgedStraddle(n_contracts=1, dte_days=30)
    engine = BacktestEngine(spy_prices, uncond, initial_cash=initial_cash, external_iv_series=vix)
    uncond_curve = engine.run()
    uncond_stats = compute_stats(uncond_curve["equity"])

    results = [{
        "threshold_vol_pts": "unconditional",
        "n_trades": len(uncond.trade_log),
        "total_return_pct": uncond_stats.total_return * 100,
        "sharpe": uncond_stats.sharpe,
        "max_dd_pct": uncond_stats.max_drawdown * 100,
    }]

    # --- Sweep thresholds ---
    conditional_results = {}
    for thresh in THRESHOLDS:
        strat = VRPConditionalStraddle(n_contracts=1, dte_days=30, vrp_threshold=thresh)
        engine = BacktestEngine(spy_prices, strat, initial_cash=initial_cash, external_iv_series=vix)
        curve = engine.run()
        stats = compute_stats(curve["equity"])
        conditional_results[thresh] = (strat, curve)
        results.append({
            "threshold_vol_pts": thresh * 100,
            "n_trades": len(strat.trade_log),
            "total_return_pct": stats.total_return * 100,
            "sharpe": stats.sharpe,
            "max_dd_pct": stats.max_drawdown * 100,
        })

    summary = pd.DataFrame(results)
    print("=== Threshold sweep ===")
    print(summary.to_string(index=False))

    # --- Plot: return and Sharpe vs threshold ---
    numeric = summary[summary["threshold_vol_pts"] != "unconditional"].copy()
    numeric["threshold_vol_pts"] = numeric["threshold_vol_pts"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(numeric["threshold_vol_pts"], numeric["total_return_pct"], marker="o")
    axes[0].axhline(uncond_stats.total_return * 100, color="gray", linestyle="--", label="Unconditional")
    axes[0].set_xlabel("VRP threshold (vol points)")
    axes[0].set_ylabel("Total Return (%)")
    axes[0].set_title("Return vs Threshold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(numeric["threshold_vol_pts"], numeric["sharpe"], marker="o", color="tab:orange")
    axes[1].axhline(uncond_stats.sharpe, color="gray", linestyle="--", label="Unconditional")
    axes[1].set_xlabel("VRP threshold (vol points)")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_title("Sharpe vs Threshold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"VRP Threshold Sweep -- {source}")
    fig.tight_layout()
    fig.savefig("vrp_threshold_sweep.png", dpi=150)
    print("\nSaved vrp_threshold_sweep.png")

    # --- Trade-level P&L breakdown for the default threshold (0.03) ---
    default_strat, default_curve = conditional_results[0.03]
    trades = pd.DataFrame(default_strat.trade_log)
    if not trades.empty:
        print(f"\n=== Trade-level breakdown (threshold=3 vol pts, {len(trades)} trades) ===")
        win_rate = (trades["pnl"] > 0).mean()
        print(f"Win rate: {win_rate*100:.1f}%")
        print(f"Average P&L per trade: {trades['pnl'].mean():.2f}")
        print(f"Median P&L per trade:  {trades['pnl'].median():.2f}")
        print(f"Best trade:  {trades['pnl'].max():.2f}")
        print(f"Worst trade: {trades['pnl'].min():.2f}")
        print(f"Std dev of trade P&L: {trades['pnl'].std():.2f}")

        # is P&L correlated with how rich VRP was at entry? (should be positive if signal has value)
        corr = trades["vrp_at_entry"].corr(trades["pnl"])
        print(f"Correlation(VRP at entry, trade P&L): {corr:.3f}  "
              f"(positive would support 'richer entry -> better trade')")

        trades.to_csv("vrp_trade_log.csv", index=False)
        print("Saved full trade log to vrp_trade_log.csv")


if __name__ == "__main__":
    main()
