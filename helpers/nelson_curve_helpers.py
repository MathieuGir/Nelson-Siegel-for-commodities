import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import minimize


def L1_NSS(x, eps=1e-8):
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


def L2_NSS(x, eps=1e-8):
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

def NSS_rate(maturities: np.ndarray, params: tuple) -> np.ndarray:
    """
    This function computes the Nelson-Siegel-Svensson rate for given maturities M and parameters.
    Parameters:
    - M: array-like, maturities
    - beta0, beta1, beta2, beta3: float, NSS parameters
    - theta1, theta2: float, NSS parameters
    Returns:
    - array-like: NSS rates for each maturity in M
    """
    beta0, beta1, beta2, beta3, theta1, theta2 = params
    
    maturities = np.array(maturities, dtype=float)

    x1 = maturities / theta1
    x2 = maturities / theta2
    return (beta0
            + beta1 * L1_NSS(x1)
            + beta2 * L2_NSS(x1)
            + beta3 * L2_NSS(x2))


def NSS_residuals(params, M, prices, verbosity: bool = False):
    log_prices = np.log(np.maximum(prices, 1e-8))  # avoid log(0)
    modeled_log_prices = NSS_rate(M, params)
    residuals = np.sum((modeled_log_prices - log_prices) ** 2)
    if verbosity:
        print(f"Parameters: {params}")
        print("Modeled log prices:", modeled_log_prices)
        print("Actual log prices:", log_prices)
        print(f"Residuals: {residuals:,.2f}")
    return residuals


def NSS_estimation(maturities,prices, initial_guess=(5.5, -2.0, 1.0, 0.5, 100, 200), min_maturity=10, max_maturity = 500, verbosity = False):
    bounds = [(None,None), (None,None), (None,None), (None,None), (1e-6,500), (1e-6,500)]   # theta1 and theta2 positive

    # Filter maturities to avoid dealing with contracts too close to expiry, or too far in time, exposed to liquidity issues 
    mask = (maturities >= min_maturity) & (maturities <= max_maturity)
    maturities = maturities[mask]
    prices = prices[mask]
    
    result = minimize(NSS_residuals, initial_guess,
                      args=(maturities, prices, verbosity),
                      method='L-BFGS-B', bounds=bounds)
    if verbosity:
        print("Sum of squared residuals:", result.fun)
        print(f"Results are: beta0={result.x[0]:,.4f}, beta1={result.x[1]:,.4f}, beta2={result.x[2]:,.4f}, beta3={result.x[3]:,.4f}, theta1={result.x[4]:,.4f}, theta2={result.x[5]:,.4f}")
    return result.x

def plot_nss_fits(
    futures_data: pd.DataFrame,
    dates: list,
    col_price: str = "price",
    min_maturity: int = 10,
    max_maturity: int = 500,
    show_actual: bool = True,
    grid_step: int = 5,      # spacing in business days for the NSS curve
    # alternatively, you could add n_grid: int = 100 and use linspace
):
    """
    Plot NSS fitted curves for several dates on the same plot.

    Parameters:
    - futures_data: DataFrame with futures data, indexed by date.
                    Must contain 'time_to_maturity' and price column.
    - dates: list of date strings to plot
    - col_price: column name for futures prices in df
    - min_maturity: minimum maturity threshold for fitting
    - max_maturity: maximum maturity threshold for fitting
    - show_actual: if True, plot actual futures prices as markers
    - grid_step: step (in business days) for evaluating the NSS curve

    Futures prices are 'X' markers and share the same color as their NSS fit.
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

    # Grid of maturities (business days) where we evaluate the NSS curve
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

        # --- NSS estimation on observed points ---
        est = NSS_estimation(maturities, prices, verbosity=False)

        # --- Smooth NSS curve on the regular grid ---
        modeled_grid = np.exp(NSS_rate(grid_maturities, est))

        fig.add_trace(
            go.Scatter(
                x=grid_maturities,
                y=modeled_grid,
                mode="lines",
                line=dict(color=color),
                name=f"NSS Fit {date_str}",
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
        title="NSS Model Fits Across Dates",
        xaxis_title="Time to Maturity (Business Days)",
        yaxis_title="Futures Price",
        legend_title="Curves",
    )

    try:
        renderer = "vscode" if "vscode" in pio.renderers else "browser"
        fig.show(renderer=renderer)
    except Exception:
        fig.show()
