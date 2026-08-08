from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import yfinance as yf


@dataclass
class ChainSnapshot:
    ticker: str
    spot: float
    as_of: datetime
    calls: pd.DataFrame
    puts: pd.DataFrame
    expiry: str
    T: float


def get_spot_price(ticker: str) -> float:
    tk = yf.Ticker(ticker)
    return float(tk.fast_info["last_price"])


def list_expirations(ticker: str) -> list[str]:
    tk = yf.Ticker(ticker)
    return list(tk.options)


def get_chain(ticker: str, expiry: str) -> ChainSnapshot:
    tk = yf.Ticker(ticker)
    spot = get_spot_price(ticker)
    chain = tk.option_chain(expiry)

    now = datetime.now()
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    T = max((expiry_dt - now).days, 1) / 365.0

    return ChainSnapshot(
        ticker=ticker,
        spot=spot,
        as_of=now,
        calls=chain.calls,
        puts=chain.puts,
        expiry=expiry,
        T=T,
    )


def get_all_chains(ticker: str, max_expirations: int = 6) -> list[ChainSnapshot]:
    expirations = list_expirations(ticker)[:max_expirations]
    return [get_chain(ticker, exp) for exp in expirations]


def get_vix_history(period: str = "5y") -> pd.Series:
    tk = yf.Ticker("^VIX")
    hist = tk.history(period=period)
    if hist.empty:
        raise RuntimeError("No VIX history returned -- check connectivity.")
    series = hist["Close"] / 100.0
    series.index = series.index.tz_localize(None)
    series.name = "VIX"
    return series


def mid_price(row: pd.Series) -> float:
    bid, ask = row.get("bid", 0.0), row.get("ask", 0.0)
    if bid and ask and ask >= bid:
        return (bid + ask) / 2
    return row.get("lastPrice", float("nan"))
