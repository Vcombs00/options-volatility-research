"""
Portfolio: tracks cash, a stock position, and any number of open option
positions, and marks the whole thing to market using the existing
Black-Scholes pricer. This is the accounting core the backtest engine and
every strategy write against.

Design choice: options are marked to market using Black-Scholes (European)
even though real listed options are American. This is a standard, explicit
simplification -- see README for the justification (early exercise of a
call is essentially never optimal absent dividends, which we already proved
in demo.py; puts are the case where this simplification is weakest).
"""

from dataclasses import dataclass, field
from options_pricer import price as bs_price


@dataclass
class OptionPosition:
    option_type: str    # 'call' or 'put'
    strike: float
    expiry_date: object  # pandas.Timestamp
    qty: int             # positive = long, negative = short (in CONTRACTS)
    entry_price: float   # premium received/paid per share at entry
    multiplier: int = 100  # shares per contract, standard US equity option

    def is_expired(self, current_date) -> bool:
        return current_date >= self.expiry_date

    def intrinsic_value(self, spot: float) -> float:
        if self.option_type == "call":
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def mark_to_market(self, spot: float, r: float, vol: float, current_date) -> float:
        """Value of this position (positive = asset, negative = liability)."""
        if self.is_expired(current_date):
            px = self.intrinsic_value(spot)
        else:
            T = max((self.expiry_date - current_date).days, 0) / 365.0
            if T <= 0:
                px = self.intrinsic_value(spot)
            else:
                px = bs_price(spot, self.strike, T, r, vol, self.option_type)
        return px * self.qty * self.multiplier


@dataclass
class Portfolio:
    cash: float
    stock_qty: int = 0
    options: list = field(default_factory=list)
    transaction_cost_bps: float = 5.0  # round-trip cost, in bps of notional; crude but standard first-pass friction model

    def trade_stock(self, qty: int, price: float):
        """Positive qty = buy, negative = sell. Applies a simple bps cost."""
        notional = abs(qty) * price
        cost = notional * self.transaction_cost_bps / 10_000
        self.cash -= qty * price
        self.cash -= cost
        self.stock_qty += qty

    def open_option(self, option_type: str, strike: float, expiry_date,
                     qty: int, premium: float, multiplier: int = 100):
        """qty > 0 = buy (pay premium), qty < 0 = sell (receive premium)."""
        notional = abs(qty) * premium * multiplier
        cost = notional * self.transaction_cost_bps / 10_000
        self.cash -= qty * premium * multiplier
        self.cash -= cost
        self.options.append(OptionPosition(option_type, strike, expiry_date, qty, premium, multiplier))

    def settle_expired(self, current_date, spot: float):
        """
        Cash-settle any option positions that have expired as of current_date.
        Removes them from the book after crediting/debiting intrinsic value.
        """
        still_open = []
        for opt in self.options:
            if opt.is_expired(current_date):
                intrinsic = opt.intrinsic_value(spot)
                self.cash += intrinsic * opt.qty * opt.multiplier
            else:
                still_open.append(opt)
        self.options = still_open

    def net_option_delta(self, spot: float, r: float, vol: float, current_date) -> float:
        """
        Portfolio delta from options only, in SHARE-equivalent units
        (i.e. already multiplied by contract size), used by delta-hedging
        strategies to know how much stock to trade to stay delta-neutral.
        """
        from options_pricer import greeks as bs_greeks
        total = 0.0
        for opt in self.options:
            if opt.is_expired(current_date):
                continue
            T = max((opt.expiry_date - current_date).days, 0) / 365.0
            if T <= 0:
                continue
            g = bs_greeks(spot, opt.strike, T, r, vol, opt.option_type)
            total += g.delta * opt.qty * opt.multiplier
        return total

    def total_equity(self, spot: float, r: float, vol: float, current_date) -> float:
        stock_val = self.stock_qty * spot
        option_val = sum(opt.mark_to_market(spot, r, vol, current_date) for opt in self.options)
        return self.cash + stock_val + option_val
