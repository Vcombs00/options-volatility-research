"""
Diagnostic: does the VRP-conditional strategy's trade timing cluster around
volatility spikes (which would explain why higher thresholds raised average
trade P&L but hurt Sharpe/drawdown -- see README)?

Reads vrp_trade_log.csv (saved by vrp_threshold_sweep.py for the 3-vol-point
threshold case) and overlays trade entries on the actual VIX history, so you
can see directly whether richer-VRP trades bunch up in turbulent periods.

REQUIRES: vrp_trade_log.csv in the same folder (run vrp_threshold_sweep.py
first if you don't have it). Also pulls real VIX history for the background
context plot -- requires internet; falls back to using entry_vol from the
trade log itself (still informative, just without the between-trade context)
if unreachable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from options_pricer.data import get_vix_history

TRADE_LOG_PATH = "vrp_trade_log.csv"


def main():
    trades = pd.read_csv(TRADE_LOG_PATH, parse_dates=["open_date", "close_date"])
    if trades.empty:
        print("Trade log is empty -- nothing to analyze.")
        return

    print(f"Loaded {len(trades)} trades from {TRADE_LOG_PATH}")
    print(f"Date range: {trades['open_date'].min().date()} to {trades['close_date'].max().date()}")

    # --- Try to get full VIX history for background context ---
    try:
        start = trades["open_date"].min() - pd.Timedelta(days=30)
        vix = get_vix_history(period="5y")
        vix = vix[vix.index >= start]
        have_vix_context = True
    except Exception as e:
        print(f"[info] Couldn't fetch VIX context ({e}); plotting trade markers only.")
        vix = None
        have_vix_context = False

    # --- Split trades into top/bottom half by VRP at entry ---
    median_vrp = trades["vrp_at_entry"].median()
    trades["vrp_bucket"] = np.where(trades["vrp_at_entry"] >= median_vrp, "High VRP", "Low VRP")

    print(f"\nMedian VRP at entry: {median_vrp*100:.2f} vol pts")
    for bucket, grp in trades.groupby("vrp_bucket"):
        print(f"  {bucket}: {len(grp)} trades, avg pnl={grp['pnl'].mean():.2f}, "
              f"win rate={ (grp['pnl']>0).mean()*100:.1f}%")

    # --- Plot 1: trade entries over time, colored by P&L, positioned at vol level ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    if have_vix_context:
        ax.plot(vix.index, vix.values * 100, color="lightgray", linewidth=1, label="VIX", zorder=1)
        y_vals = [vix.asof(d) * 100 for d in trades["open_date"]]
        ax.set_ylabel("VIX level")
    else:
        y_vals = trades["entry_vol"] * 100
        ax.set_ylabel("Entry vol (from trade log)")

    pnl_abs_max = trades["pnl"].abs().max()
    sc = ax.scatter(trades["open_date"], y_vals,
                     c=trades["pnl"], cmap="RdYlGn", s=90, edgecolor="black", zorder=3,
                     vmin=-pnl_abs_max, vmax=pnl_abs_max)
    fig.colorbar(sc, ax=ax, label="Trade P&L ($)")
    ax.set_title("Trade Entries Over Time (color = P&L, position = vol level at entry)")
    if have_vix_context:
        ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # --- Plot 2: VRP at entry over time, colored by bucket ---
    ax2 = axes[1]
    colors = trades["vrp_bucket"].map({"High VRP": "tab:red", "Low VRP": "tab:blue"})
    ax2.scatter(trades["open_date"], trades["vrp_at_entry"] * 100, c=colors, s=70, edgecolor="black")
    ax2.axhline(median_vrp * 100, color="gray", linestyle="--", label=f"Median ({median_vrp*100:.1f} pts)")
    ax2.set_ylabel("VRP at entry (vol pts)")
    ax2.set_xlabel("Date")
    ax2.set_title("VRP at Entry Over Time (red = high-VRP half, blue = low-VRP half)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("trade_timeline_diagnostic.png", dpi=150)
    print("\nSaved trade_timeline_diagnostic.png")

    # --- Direct test: is high-VRP trade timing concentrated (clustered) vs spread out? ---
    # Compare the spread (std dev) of entry dates (as ordinal numbers) between buckets --
    # a smaller spread for High VRP trades is evidence of clustering into specific windows.
    trades["open_date_ordinal"] = trades["open_date"].map(pd.Timestamp.toordinal)
    spread_by_bucket = trades.groupby("vrp_bucket")["open_date_ordinal"].std()
    print("\nSpread (std dev, in days) of trade entry dates by bucket:")
    print(spread_by_bucket.to_string())
    print("(A noticeably smaller spread for High VRP suggests those trades cluster in time --")
    print(" i.e. the threshold is picking out specific volatile windows rather than being spread evenly.)")


if __name__ == "__main__":
    main()
