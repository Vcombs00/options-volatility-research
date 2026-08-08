"""
Implied volatility surface: back out sigma from real market option prices
across a grid of strikes and expiries, then visualize the smile/skew.

This is the payoff of building implied_vol() properly in black_scholes.py --
apply it row-by-row across a real chain and you get the market's own view
of volatility, which is the actual object of study in options research
(the "surface" is the primary object, not any single option's price).
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from .black_scholes import implied_vol
from .data import ChainSnapshot, mid_price


@dataclass
class SurfacePoint:
    expiry: str
    T: float
    strike: float
    market_price: float
    iv: float
    option_type: str


def build_surface(chains: list[ChainSnapshot], r: float = 0.05,
                   q: float = 0.0, option_type: str = "call",
                   min_price: float = 0.05,
                   moneyness_range: tuple = (0.7, 1.3)) -> pd.DataFrame:
    """
    Build a long-format DataFrame of implied vols across strike x expiry.

    Filters applied (both matter for data quality):
      - min_price: drops near-worthless quotes where bid/ask is noisy and
        implied vol becomes numerically unstable (tiny price changes imply
        huge vol swings).
      - moneyness_range: restricts to strikes within +/-30% of spot by
        default, since far OTM/ITM strikes are illiquid and their quoted
        prices are unreliable/stale.
    """
    rows = []
    for snap in chains:
        df = snap.calls if option_type == "call" else snap.puts
        for _, row in df.iterrows():
            K = row["strike"]
            moneyness = K / snap.spot
            if not (moneyness_range[0] <= moneyness <= moneyness_range[1]):
                continue

            px = mid_price(row)
            if pd.isna(px) or px < min_price:
                continue

            try:
                iv = implied_vol(px, snap.spot, K, snap.T, r, option_type, q)
            except (ValueError, RuntimeError):
                continue

            # Sanity bound: implied_vol's bisection fallback can return its
            # search boundary (5.0) when it fails to converge -- drop those.
            if not (0.01 < iv < 3.0):
                continue

            rows.append(SurfacePoint(
                expiry=snap.expiry, T=snap.T, strike=K,
                market_price=px, iv=iv, option_type=option_type,
            ))

    return pd.DataFrame([vars(p) for p in rows])


def plot_smile(surface_df: pd.DataFrame, expiry: str):
    """Plot the vol smile/skew for a single expiry: IV vs strike."""
    sub = surface_df[surface_df["expiry"] == expiry].sort_values("strike")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["strike"], sub["iv"] * 100, marker="o", markersize=3)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title(f"Volatility Smile -- expiry {expiry}")
    ax.grid(alpha=0.3)
    return fig


def plot_surface_3d(surface_df: pd.DataFrame):
    """
    Full 3D surface: strike x time-to-expiry x implied vol.

    Uses a scatter rather than a fitted mesh since real chains have uneven
    strike spacing per expiry -- interpolating to a smooth mesh is a
    reasonable next step (e.g., via scipy.interpolate.griddata) but adds
    modeling choices worth doing deliberately rather than by default.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        surface_df["strike"], surface_df["T"], surface_df["iv"] * 100,
        c=surface_df["iv"] * 100, cmap="viridis", s=15,
    )
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to Expiry (years)")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title("Implied Volatility Surface")
    fig.colorbar(sc, shrink=0.5, label="IV (%)")
    return fig
