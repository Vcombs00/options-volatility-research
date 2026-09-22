
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

    sub = surface_df[surface_df["expiry"] == expiry].sort_values("strike")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["strike"], sub["iv"] * 100, marker="o", markersize=3)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title(f"Volatility Smile -- expiry {expiry}")
    ax.grid(alpha=0.3)
    return fig


def plot_surface_3d(surface_df: pd.DataFrame):

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
