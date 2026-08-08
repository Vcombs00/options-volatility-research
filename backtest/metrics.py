"""
Standard performance metrics computed off an equity curve. Nothing exotic --
these are the numbers any research write-up or interview conversation about
a backtest is expected to include.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PerformanceStats:
    total_return: float
    cagr: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    max_drawdown_date: object


def compute_stats(equity_curve: pd.Series, r_f: float = 0.05) -> PerformanceStats:
    equity_curve = equity_curve.dropna()
    daily_ret = equity_curve.pct_change().dropna()

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    ann_vol = daily_ret.std() * np.sqrt(252)
    ann_ret = daily_ret.mean() * 252
    sharpe = (ann_ret - r_f) / ann_vol if ann_vol > 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    return PerformanceStats(
        total_return=total_return,
        cagr=cagr,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        max_drawdown_date=max_dd_date,
    )


def print_stats(name: str, stats: PerformanceStats):
    print(f"\n--- {name} ---")
    print(f"Total return:     {stats.total_return * 100:6.2f}%")
    print(f"CAGR:             {stats.cagr * 100:6.2f}%")
    print(f"Annualized vol:   {stats.annualized_vol * 100:6.2f}%")
    print(f"Sharpe ratio:     {stats.sharpe:6.2f}")
    print(f"Max drawdown:     {stats.max_drawdown * 100:6.2f}%  (on {stats.max_drawdown_date.date()})")
