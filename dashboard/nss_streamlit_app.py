import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Ensure project root is on path
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ------------------------------------------------------------
# Local helpers
# ------------------------------------------------------------
from helpers.nelson_curve_helpers import NSS_estimation, NSS_rate
from helpers.data_helpers import load_calibration_data


# ------------------------------------------------------------
# Plot for ONE date
# ------------------------------------------------------------
def nss_figure_for_date(
    futures_data: pd.DataFrame,
    date: pd.Timestamp,
    min_maturity: int = 10,
    max_maturity: int = 500,
    grid_step: int = 15,
) -> Tuple[go.Figure, Optional[np.ndarray]]:

    fig = go.Figure()
    date = pd.to_datetime(date)

    try:
        day = futures_data.loc[date]
    except KeyError:
        fig.add_annotation(
            text=f"No data for {date.date()}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig, None

    if isinstance(day, pd.Series):
        day = day.to_frame().T

    day = day[
        (day["time_to_maturity"] >= min_maturity) &
        (day["time_to_maturity"] <= max_maturity)
    ].sort_values("time_to_maturity")

    if day.empty:
        fig.add_annotation(
            text=f"No contracts in window on {date.date()}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig, None

    m = day["time_to_maturity"].to_numpy()
    p = day["price"].to_numpy()

    params = NSS_estimation(m, p, verbosity=False)

    grid_start = max(min_maturity, int(np.min(m)))
    grid = np.arange(grid_start, max_maturity + grid_step, grid_step)
    modeled = np.exp(NSS_rate(grid, params))

    fig.add_trace(go.Scatter(x=grid, y=modeled, mode="lines", name="NSS Fit"))
    fig.add_trace(go.Scatter(
        x=m, y=p, mode="markers",
        marker=dict(symbol="x", size=9),
        name="Actual"
    ))

    fig.update_layout(
        title=f"NSS Term Structure — {date.date()}",
        xaxis_title="Time to Maturity (Business Days)",
        yaxis_title="Futures Price",
        template="plotly_white",
        margin=dict(t=60, l=60, r=30, b=60),
    )

    return fig, params


# ------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="NSS Explorer", layout="wide")
    st.title("NSS Term Structure Explorer")

    @st.cache_data(show_spinner=False)
    def load_data():
        return load_calibration_data(
            commo_ticker="CT",
            start_date="1980-01-01",
            end_date="2025-12-31",
        )

    futures_data = load_data()
    if futures_data.empty:
        st.error("No data loaded.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def get_dates(df):
        return pd.DatetimeIndex(pd.to_datetime(df.index)).unique().sort_values()

    dates = get_dates(futures_data)
    n_dates = len(dates)

    # ---------------- State ----------------
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False

    # ---------------- Controls ----------------
    st.sidebar.header("Playback Controls")

    speed_ms = st.sidebar.slider("Speed (ms)", 50, 2000, 200, 50)
    step = st.sidebar.slider("Step (#dates)", 1, 50, 5, 1)

    c1, c2, c3 = st.sidebar.columns(3)

    if c1.button("◀ Prev"):
        st.session_state.current_idx = max(0, st.session_state.current_idx - step)
        st.session_state.playing = False

    play_label = "⏸ Pause" if st.session_state.playing else "▶ Play"
    if c2.button(play_label):
        st.session_state.playing = not st.session_state.playing

    if c3.button("Next ▶"):
        st.session_state.current_idx = min(n_dates - 1, st.session_state.current_idx + step)
        st.session_state.playing = False

    # ---------------- Slider ----------------
    slider_value = st.slider(
        "Date index",
        0,
        n_dates - 1,
        value=st.session_state.current_idx,
    )
    
    # Update current_idx from slider if user moved it
    if slider_value != st.session_state.current_idx:
        st.session_state.current_idx = slider_value
        st.session_state.playing = False

    # ---------------- AUTO PLAY ----------------
    if st.session_state.playing:
        time.sleep(speed_ms / 1000.0)
        next_idx = st.session_state.current_idx + step
        if next_idx >= n_dates:
            st.session_state.playing = False
            st.session_state.current_idx = n_dates - 1
        else:
            st.session_state.current_idx = next_idx
        st.rerun()

    # ---------------- Plot ----------------
    d = dates[int(st.session_state.current_idx)]
    st.subheader(f"Date: {d.date()}")

    @st.cache_data(show_spinner=False)
    def cached_plot(date_val):
        return nss_figure_for_date(futures_data, date_val)

    fig, params = cached_plot(d)

    left, right = st.columns([1, 3])
    with left:
        st.markdown("### NSS Parameters")
        if params is not None:
            b0, b1, b2, b3, t1, t2 = params
            st.write(
                f"β₀: {b0:.4f}\n\n"
                f"β₁: {b1:.4f}\n\n"
                f"β₂: {b2:.4f}\n\n"
                f"β₃: {b3:.4f}\n\n"
                f"θ₁: {t1:.2f}\n\n"
                f"θ₂: {t2:.2f}"
            )

    with right:
        st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
