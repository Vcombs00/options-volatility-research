"""
Strategies plug into the backtest engine via a single on_bar() callback.
Each strategy owns its own state (e.g. "do I currently have an open
straddle?") and decides what trades to make given the current date, spot
price, the vol used for PRICING options (real market IV if the engine was
given one, else realized vol), and the trailing REALIZED vol of the
underlying (always the true rolling realized vol, regardless of pricing
source) -- so a strategy can compare the two.

Four strategies are provided as a spectrum of complexity:
  BuyAndHold                benchmark, no options at all
  CoveredCall                static income strategy, sell 1 OTM call per cycle
  DeltaHedgedStraddle          sell vol every cycle, hedge delta daily, unconditionally
  VRPConditionalStraddle         only sell vol when pricing_vol is rich vs realized_vol
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    name: str = "Strategy"

    @abstractmethod
    def on_bar(self, engine, date: pd.Timestamp, spot: float, vol: float, realized_vol: float = None):
        """
        Called once per trading day. Mutate engine.portfolio via its trade
        methods. `vol` is the pricing vol (real IV if provided to the
        engine, else realized vol); `realized_vol` is always the trailing
        realized vol of the underlying, for strategies that want to compare
        the two (e.g. only sell vol when it's rich).
        """
        raise NotImplementedError


class BuyAndHold(Strategy):
    """Benchmark: buy N shares on day 1, never trade again."""
    name = "Buy & Hold"

    def __init__(self, shares: int = 100):
        self.shares = shares
        self._bought = False

    def on_bar(self, engine, date, spot, vol, realized_vol=None):
        if not self._bought:
            engine.portfolio.trade_stock(self.shares, spot)
            self._bought = True


class CoveredCall(Strategy):
    """
    Classic income strategy: hold 100 shares, continuously sell one
    out-of-the-money call against them. On each expiry, the option either
    expires worthless (keep the premium, sell a new call) or gets assigned
    (shares called away at the strike -- we model this as: collect strike
    proceeds, then immediately re-buy the shares to keep the position
    running, so the strategy is directly comparable to buy-and-hold on the
    same share count).
    """
    name = "Covered Call"

    def __init__(self, shares: int = 100, otm_pct: float = 0.05,
                 dte_days: int = 30, r: float = 0.05):
        self.shares = shares
        self.otm_pct = otm_pct
        self.dte_days = dte_days
        self.r = r
        self._initialized = False

    def on_bar(self, engine, date, spot, vol, realized_vol=None):
        pf = engine.portfolio

        if not self._initialized:
            pf.trade_stock(self.shares, spot)
            self._initialized = True

        pf.settle_expired(date, spot)

        if pf.stock_qty < self.shares:
            pf.trade_stock(self.shares - pf.stock_qty, spot)

        has_open_call = any(o.option_type == "call" and not o.is_expired(date) for o in pf.options)
        if not has_open_call:
            strike = round(spot * (1 + self.otm_pct), 0)
            expiry = date + pd.Timedelta(days=self.dte_days)
            T = self.dte_days / 365.0
            premium = _safe_bs_price(spot, strike, T, self.r, vol, "call")
            n_contracts = self.shares // 100
            if n_contracts >= 1 and premium > 0:
                pf.open_option("call", strike, expiry, qty=-n_contracts, premium=premium)


class DeltaHedgedStraddle(Strategy):
    """
    Short-vol strategy: sell an at-the-money straddle (call + put) each
    cycle, then delta-hedge daily by trading the underlying so net delta
    stays near zero. Sells UNCONDITIONALLY every cycle regardless of
    whether pricing_vol is rich or cheap relative to realized_vol -- if the
    engine is given real IV via external_iv_series, this strategy will
    passively capture (or lose to) the volatility risk premium depending on
    what actually happens; if not, it's edge-free by construction (see
    README zero-edge validation).

    Tracks a trade_log (list of dicts) with one entry per completed cycle:
    open_date, close_date, entry_vol (pricing vol at entry), entry_realized_vol,
    vrp_at_entry, and pnl (mark-to-market equity change over the cycle,
    including hedging trades) -- useful for post-hoc analysis of which
    trades drove overall performance.
    """
    name = "Delta-Hedged Short Straddle"

    def __init__(self, n_contracts: int = 1, dte_days: int = 30,
                 r: float = 0.05, hedge_band: float = 5.0):
        self.n_contracts = n_contracts
        self.dte_days = dte_days
        self.r = r
        self.hedge_band = hedge_band
        self._cycle_active = False
        self._cycle_entry_equity = None
        self._cycle_entry_date = None
        self._cycle_entry_vol = None
        self._cycle_entry_realized_vol = None
        self.trade_log = []

    def on_bar(self, engine, date, spot, vol, realized_vol=None):
        pf = engine.portfolio

        pf.settle_expired(date, spot)
        still_open = any(not o.is_expired(date) for o in pf.options)

        if not still_open and self._cycle_active:
            if pf.stock_qty != 0:
                pf.trade_stock(-pf.stock_qty, spot)
            close_equity = pf.total_equity(spot, self.r, vol, date)
            self.trade_log.append({
                "open_date": self._cycle_entry_date,
                "close_date": date,
                "entry_vol": self._cycle_entry_vol,
                "entry_realized_vol": self._cycle_entry_realized_vol,
                "vrp_at_entry": (self._cycle_entry_vol - self._cycle_entry_realized_vol
                                 if self._cycle_entry_realized_vol is not None else None),
                "pnl": close_equity - self._cycle_entry_equity,
            })
            self._cycle_active = False

        if not self._cycle_active:
            strike = round(spot, 0)
            expiry = date + pd.Timedelta(days=self.dte_days)
            T = self.dte_days / 365.0
            call_px = _safe_bs_price(spot, strike, T, self.r, vol, "call")
            put_px = _safe_bs_price(spot, strike, T, self.r, vol, "put")
            self._cycle_entry_equity = pf.total_equity(spot, self.r, vol, date)
            self._cycle_entry_date = date
            self._cycle_entry_vol = vol
            self._cycle_entry_realized_vol = realized_vol
            if call_px > 0:
                pf.open_option("call", strike, expiry, qty=-self.n_contracts, premium=call_px)
            if put_px > 0:
                pf.open_option("put", strike, expiry, qty=-self.n_contracts, premium=put_px)
            self._cycle_active = True

        if self._cycle_active:
            option_delta = pf.net_option_delta(spot, self.r, vol, date)
            target_stock_qty = -round(option_delta)
            drift = target_stock_qty - pf.stock_qty
            if abs(drift) >= self.hedge_band:
                pf.trade_stock(drift, spot)


class VRPConditionalStraddle(Strategy):
    """
    Signal-driven variant of the short straddle: only initiates a new cycle
    when pricing_vol (real IV, if provided) is meaningfully rich relative to
    trailing realized_vol -- i.e. only sells volatility when there's a
    documented reason to expect an edge (the volatility risk premium). When
    the signal isn't present, the strategy holds no position at all rather
    than selling unconditionally.

    This mirrors real practice: desks and vol-selling funds condition
    entries on some measure of "is IV rich" (IV rank, IV percentile, IV
    minus RV, etc.) rather than being permanently short vol.
    """
    name = "VRP-Conditional Short Straddle"

    def __init__(self, n_contracts: int = 1, dte_days: int = 30,
                 r: float = 0.05, hedge_band: float = 5.0,
                 vrp_threshold: float = 0.03):
        """
        vrp_threshold: minimum (pricing_vol - realized_vol) required, in
                       annualized vol points (e.g. 0.03 = 3 vol points), to
                       initiate a new short straddle cycle.
        """
        self.n_contracts = n_contracts
        self.dte_days = dte_days
        self.r = r
        self.hedge_band = hedge_band
        self.vrp_threshold = vrp_threshold
        self._cycle_active = False
        self._cycle_entry_equity = None
        self._cycle_entry_date = None
        self._cycle_entry_vol = None
        self._cycle_entry_realized_vol = None
        self.trade_log = []

    def on_bar(self, engine, date, spot, vol, realized_vol=None):
        pf = engine.portfolio

        pf.settle_expired(date, spot)
        still_open = any(not o.is_expired(date) for o in pf.options)

        if not still_open and self._cycle_active:
            if pf.stock_qty != 0:
                pf.trade_stock(-pf.stock_qty, spot)
            close_equity = pf.total_equity(spot, self.r, vol, date)
            self.trade_log.append({
                "open_date": self._cycle_entry_date,
                "close_date": date,
                "entry_vol": self._cycle_entry_vol,
                "entry_realized_vol": self._cycle_entry_realized_vol,
                "vrp_at_entry": (self._cycle_entry_vol - self._cycle_entry_realized_vol
                                 if self._cycle_entry_realized_vol is not None else None),
                "pnl": close_equity - self._cycle_entry_equity,
            })
            self._cycle_active = False

        vrp = vol - (realized_vol if realized_vol is not None else vol)
        signal_present = vrp >= self.vrp_threshold

        if not self._cycle_active and signal_present:
            strike = round(spot, 0)
            expiry = date + pd.Timedelta(days=self.dte_days)
            T = self.dte_days / 365.0
            call_px = _safe_bs_price(spot, strike, T, self.r, vol, "call")
            put_px = _safe_bs_price(spot, strike, T, self.r, vol, "put")
            self._cycle_entry_equity = pf.total_equity(spot, self.r, vol, date)
            self._cycle_entry_date = date
            self._cycle_entry_vol = vol
            self._cycle_entry_realized_vol = realized_vol
            if call_px > 0:
                pf.open_option("call", strike, expiry, qty=-self.n_contracts, premium=call_px)
            if put_px > 0:
                pf.open_option("put", strike, expiry, qty=-self.n_contracts, premium=put_px)
            self._cycle_active = True

        if self._cycle_active:
            option_delta = pf.net_option_delta(spot, self.r, vol, date)
            target_stock_qty = -round(option_delta)
            drift = target_stock_qty - pf.stock_qty
            if abs(drift) >= self.hedge_band:
                pf.trade_stock(drift, spot)


def _safe_bs_price(S, K, T, r, sigma, option_type):
    """BS price with a floor on T and sigma to avoid div-by-zero at the edges."""
    T = max(T, 1 / 365.0)
    sigma = max(sigma, 0.01)
    from options_pricer import price as bs_price
    return bs_price(S, K, T, r, sigma, option_type)
