import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import minimize


def L1_NS(x, eps=1e-8):
    """
    Compute (1 - e^{-x}) / x safely, using Taylor approx near zero for stability
    As x -> 0, (1 - e^{-x}) / x -> 1 - x/2 + x^2/6
    """
    x = np.array(x, dtype=float)
    out = np.empty_like(x)
    
    # mask for where we use the Taylor approximation
    approx = np.abs(x) < eps
    exact = ~approx
    
    out[exact] = -np.expm1(-x[exact]) / x[exact] # perform exact calculation
    xx = x[approx]
    out[approx] = 1.0 - xx/2.0 + (xx**2)/6.0 # Taylor expansion: 1 - x/2 + x^2/6
    
    return out


def L2_NS(x, eps=1e-8):
    """(1 - (1+x)e^{-x}) / x -> 1/2 as x->0"""
    x = np.array(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < eps
    large = ~small
    out[large] = (1.0 - (1.0 + x[large]) * np.exp(-x[large])) / x[large]
    # Taylor: 1/2 - x/6 + x^2/24  (from series expansion)
    xx = x[small]
    out[small] = 0.5 - xx/6.0 + (xx**2)/24.0
    return out

def NS_rate(maturities: np.ndarray, params: tuple) -> np.ndarray:
    """
    This function computes the Nelson-Siegel rate for given maturities M and parameters.
    
    Formula: y(τ) = β₀ + β₁·L₁(τ/λ) + β₂·L₂(τ/λ)
    
    Parameters:
    - maturities: array-like, maturities (τ)
    - params: tuple of (beta0, beta1, beta2, lambda)
      - beta0: level factor
      - beta1: slope factor (loading on L1)
      - beta2: curvature factor (loading on L2)
      - lambda: decay parameter (τ/λ)
    
    Returns:
    - array-like: NS rates for each maturity
    """
    beta0, beta1, beta2, lam = params
    
    maturities = np.array(maturities, dtype=float)
    x = maturities / lam
    
    return beta0 + beta1 * L1_NS(x) + beta2 * L2_NS(x)


def NS_residuals(params, M, prices, verbosity: bool = False):
    log_prices = np.log(np.maximum(prices, 1e-8))  # avoid log(0)
    modeled_log_prices = NS_rate(M, params)
    residuals = np.sum((modeled_log_prices - log_prices) ** 2)
    if verbosity:
        print(f"Parameters: {params}")
        print("Modeled log prices:", modeled_log_prices)
        print("Actual log prices:", log_prices)
        print(f"Residuals: {residuals:,.2f}")
    return residuals


def NS_estimation(maturities, prices, initial_guess=(5.5, -2.0, 1.0, 50), min_maturity=10, max_maturity=500, verbosity=False):
    """
    Estimate 4-parameter Nelson-Siegel model using least squares optimization.
    
    Parameters:
    - maturities: array of time-to-maturity values
    - prices: array of futures prices
    - initial_guess: tuple of (beta0, beta1, beta2, lambda) starting values
    - min_maturity: minimum maturity to include in fit
    - max_maturity: maximum maturity to include in fit
    - verbosity: if True, print detailed results
    
    Returns:
    - params: array of [beta0, beta1, beta2, lambda] fitted parameters
    """
    # Lambda (decay parameter) must be positive
    bounds = [(None, None), (None, None), (None, None), (1e-6, 500)]

    # Filter maturities to avoid contracts too close to expiry or too far in time
    mask = (maturities >= min_maturity) & (maturities <= max_maturity)
    maturities = maturities[mask]
    prices = prices[mask]
    
    result = minimize(NS_residuals, initial_guess,
                      args=(maturities, prices, verbosity),
                      method='L-BFGS-B', bounds=bounds)
    if verbosity:
        print("Sum of squared residuals:", result.fun)
        print(f"Fitted parameters:")
        print(f"  β₀ (level):     {result.x[0]:,.4f}")
        print(f"  β₁ (slope):     {result.x[1]:,.4f}")
        print(f"  β₂ (curvature): {result.x[2]:,.4f}")
        print(f"  λ (decay):      {result.x[3]:,.4f}")
    return result.x

def plot_ns_fits(
    futures_data: pd.DataFrame,
    dates: list,
    col_price: str = "price",
    min_maturity: int = 10,
    max_maturity: int = 500,
    show_actual: bool = True,
    grid_step: int = 5,      # spacing in business days for the NS curve
    # alternatively, you could add n_grid: int = 100 and use linspace
):
    """
    Plot NS fitted curves for several dates on the same plot.

    Parameters:
    - futures_data: DataFrame with futures data, indexed by date.
                    Must contain 'time_to_maturity' and price column.
    - dates: list of date strings to plot
    - col_price: column name for futures prices in df
    - min_maturity: minimum maturity threshold for fitting
    - max_maturity: maximum maturity threshold for fitting
    - show_actual: if True, plot actual futures prices as markers
    - grid_step: step (in business days) for evaluating the NS curve

    Futures prices are 'X' markers and share the same color as their NS fit.
    """

    fig = go.Figure()

    # A deterministic color cycle — add more if needed
    color_cycle = [
        "#d62728",  # red
        "#2ca02c",  # green
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]

    # Grid of maturities (business days) where we evaluate the NS curve
    grid_maturities = np.arange(min_maturity, max_maturity + grid_step, grid_step)

    for i, date_str in enumerate(dates):

        color = color_cycle[i % len(color_cycle)]

        # --- Check date existence ---
        if date_str not in futures_data.index:
            raise ValueError(f"{date_str} not found in DataFrame index.")

        # --- Filter by threshold & sort by maturity ---
        day_data = futures_data.loc[date_str]
        day_data = day_data[
            (day_data["time_to_maturity"] >= min_maturity)
            & (day_data["time_to_maturity"] <= max_maturity)
        ].sort_values("time_to_maturity")

        maturities = day_data["time_to_maturity"].to_numpy()
        prices = day_data[col_price].to_numpy()

        if len(maturities) == 0:
            continue  # nothing to fit for this date

        # --- NS estimation on observed points ---
        est = NS_estimation(maturities, prices, verbosity=False)

        # --- Smooth NS curve on the regular grid ---
        modeled_grid = np.exp(NS_rate(grid_maturities, est))

        fig.add_trace(
            go.Scatter(
                x=grid_maturities,
                y=modeled_grid,
                mode="lines",
                line=dict(color=color),
                name=f"NS Fit {date_str}",
            )
        )

        # --- Futures prices in 'X' markers (same color) ---
        if show_actual:
            fig.add_trace(
                go.Scatter(
                    x=maturities,
                    y=prices,
                    mode="markers",
                    marker=dict(symbol="x", size=9, color=color),
                    name=f"Actual {date_str}",
                )
            )

    fig.update_layout(
        title="NS Model Fits Across Dates",
        xaxis_title="Time to Maturity (Business Days)",
        yaxis_title="Futures Price",
        legend_title="Curves",
    )

    try:
        renderer = "vscode" if "vscode" in pio.renderers else "browser"
        fig.show(renderer=renderer)
    except Exception:
        fig.show()
