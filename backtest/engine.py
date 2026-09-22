import numpy as np
import pandas as pd

from .portfolio import Portfolio


class BacktestEngine:
    def __init__(self, price_series: pd.Series, strategy, initial_cash: float = 100_000,
                 r: float = 0.05, vol_window: int = 20, transaction_cost_bps: float = 5.0,
                 external_iv_series: pd.Series = None):
        self.prices = price_series.sort_index()
        self.strategy = strategy
        self.r = r
        self.vol_window = vol_window
        self.external_iv_series = external_iv_series
        self.portfolio = Portfolio(cash=initial_cash, transaction_cost_bps=transaction_cost_bps)
        self.equity_curve = None

    def _realized_vol_series(self) -> pd.Series:
        log_ret = np.log(self.prices / self.prices.shift(1))
        rolling_std = log_ret.rolling(self.vol_window).std()
        return rolling_std * np.sqrt(252)

    def run(self) -> pd.DataFrame:
        realized_vol_series = self._realized_vol_series()
        records = []
        warned_missing_iv = False

        for date, spot in self.prices.items():
            realized_vol = realized_vol_series.loc[date]

            if self.external_iv_series is not None and date in self.external_iv_series.index:
                pricing_vol = self.external_iv_series.loc[date]
            elif self.external_iv_series is not None:
                pricing_vol = realized_vol
                if not warned_missing_iv and not pd.isna(realized_vol):
                    print(f"[warn] no external IV for {date.date()}, falling back to realized vol")
                    warned_missing_iv = True
            else:
                pricing_vol = realized_vol

            if pd.isna(realized_vol) or pd.isna(pricing_vol):
                records.append({"date": date, "spot": spot, "pricing_vol": np.nan,
                                 "realized_vol": np.nan,
                                 "equity": self.portfolio.total_equity(spot, self.r, 0.2, date)})
                continue

            self.strategy.on_bar(self, date, spot, pricing_vol, realized_vol)
            equity = self.portfolio.total_equity(spot, self.r, pricing_vol, date)
            records.append({"date": date, "spot": spot, "pricing_vol": pricing_vol,
                             "realized_vol": realized_vol, "equity": equity})

        self.equity_curve = pd.DataFrame(records).set_index("date")
        return self.equity_curve
