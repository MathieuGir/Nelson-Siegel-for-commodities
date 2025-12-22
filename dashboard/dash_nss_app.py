import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Memory

from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context

# --------------------------
# Path setup (same as you did)
# --------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helpers.nelson_curve_helpers import NSS_estimation, NSS_rate
from helpers.data_helpers import load_calibration_data


# --------------------------
# Config
# --------------------------
AVAILABLE_TICKERS = ["CT", "KC", "SB", "CC"]
START_DATE = "1980-01-01"
END_DATE = "2025-12-31"

MIN_MATURITY = 10
MAX_MATURITY = 300
GRID_STEP = 15

# Persistent cache on disk for data + params
CACHE_DIR = ROOT_DIR / ".cache" / "dash_nss"
memory = Memory(location=str(CACHE_DIR), verbose=0)


# --------------------------
# Cached data loaders
# --------------------------
@memory.cache
def load_dataset(ticker: str):
    df = load_calibration_data(
        commo_ticker=ticker,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    if df.empty:
        raise RuntimeError(f"Loaded futures_data is empty for {ticker}. Check load_calibration_data().")
    dates = pd.DatetimeIndex(pd.to_datetime(df.index)).unique().sort_values()

    # y-range robust stats per ticker
    p = df["price"].astype(float).to_numpy()
    p = p[np.isfinite(p) & (p > 0)]
    y_lo = float(np.quantile(p, 0.005))
    y_hi = float(np.quantile(p, 0.995))
    pad = 0.15 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    y_range = (y_lo - pad, y_hi + pad)
    cap_low = y_lo - 2 * pad
    cap_high = y_hi + 2 * pad

    return df, dates, y_range, (cap_low, cap_high)


# --------------------------
# Cache NSS params per (ticker, date index)
# --------------------------
@memory.cache
def fit_params_for_idx_cached(ticker: str, date: pd.Timestamp):
    df, dates, _, _ = load_dataset(ticker)
    day = slice_day(df, date)
    if day.empty:
        return None
    m = day["time_to_maturity"].to_numpy()
    pr = day["price"].to_numpy()
    return NSS_estimation(m, pr, verbosity=False)


# --------------------------
# Fast per-date slice helper
# --------------------------
def slice_day(df: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    day = df.loc[d]
    if isinstance(day, pd.Series):
        day = day.to_frame().T
    day = day[
        (day["time_to_maturity"] >= MIN_MATURITY) &
        (day["time_to_maturity"] <= MAX_MATURITY)
    ].sort_values("time_to_maturity")
    return day


# --------------------------
# Cache NSS params per date index (big speedup)
# --------------------------
@lru_cache(maxsize=5000)
def fit_params_for_idx(i: int, ticker: str):
    df, dates, _, _ = load_dataset(ticker)
    d = dates[i]
    day = slice_day(df, d)
    if day.empty:
        return None
    m = day["time_to_maturity"].to_numpy()
    pr = day["price"].to_numpy()
    params = NSS_estimation(m, pr, verbosity=False)
    return params


def make_figure(ticker: str, i: int) -> go.Figure:
    df, dates, y_range, caps = load_dataset(ticker)
    cap_low, cap_high = caps
    d = dates[i]
    day = slice_day(df, d)

    fig = go.Figure()

    if day.empty:
        fig.add_annotation(
            text=f"No data on {d.date()} in maturity window",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    else:
        m = day["time_to_maturity"].to_numpy()
        pr = day["price"].to_numpy()

        params = fit_params_for_idx(i, ticker)
        if params is not None:
            # Start NSS curve from GRID_STEP
            grid = np.arange(GRID_STEP, MAX_MATURITY + GRID_STEP, GRID_STEP)
            modeled = np.exp(NSS_rate(grid, params))

            # Clip extreme values to keep curve visible and prevent axis jumping
            modeled = np.clip(modeled, cap_low, cap_high)

            fig.add_trace(go.Scatter(x=grid, y=modeled, mode="lines", name="NSS Fit"))

        fig.add_trace(go.Scatter(
            x=m, y=pr,
            mode="markers",
            marker=dict(symbol="x", size=9),
            name="Actual"
        ))

    fig.update_layout(
        title=f"Commodity {ticker} — {d.date()}",
        template="plotly_white",
        xaxis=dict(title="Time to Maturity (Business Days)", range=[MIN_MATURITY, MAX_MATURITY]),
        yaxis=dict(title="Futures Price", range=list(y_range)),
        margin=dict(t=60, l=60, r=30, b=60),
        showlegend=True,
    )

    return fig


# --------------------------
# Dash app
# --------------------------
app = Dash(__name__)
app.title = "Commodity NSS Viewer"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H2("Commodity NSS Curves"),

        # Commodity selector
        html.Div(
            style={"marginBottom": "12px"},
            children=[
                html.Label("Commodity", style={"marginRight": "8px"}),
                dcc.Dropdown(
                    id="commodity",
                    options=[{"label": t, "value": t} for t in AVAILABLE_TICKERS],
                    value=AVAILABLE_TICKERS[0],
                    clearable=False,
                    style={"width": "200px"},
                ),
            ],
        ),

        # Flex container: parameters on left, graph on right
        html.Div(
            style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
            children=[
                # Left: Parameter box
                html.Div(
                    id="params-box",
                    style={
                        "minWidth": "200px",
                        "padding": "20px",
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                        "fontSize": "14px",
                    },
                    children=[html.Div("No data")],
                ),
                # Right: Graph
                html.Div(
                    style={"flex": "1"},
                    children=[
                        dcc.Graph(id="graph", config={"displayModeBar": True}),
                    ],
                ),
            ],
        ),

        html.Div(
            style={
                "display": "flex", 
                "gap": "12px", 
                "alignItems": "center",
                "marginTop": "20px",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            },
            children=[
                html.Button(
                    "Prev", 
                    id="prev", 
                    n_clicks=0,
                    style={
                        "padding": "8px 16px",
                        "borderRadius": "20px",
                        "border": "1px solid #ddd",
                        "backgroundColor": "#fff",
                        "cursor": "pointer",
                        "fontSize": "14px",
                    },
                ),
                html.Button(
                    "Play / Pause", 
                    id="toggle", 
                    n_clicks=0,
                    style={
                        "padding": "8px 16px",
                        "borderRadius": "20px",
                        "border": "1px solid #ddd",
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "cursor": "pointer",
                        "fontSize": "14px",
                    },
                ),
                html.Button(
                    "Next", 
                    id="next", 
                    n_clicks=0,
                    style={
                        "padding": "8px 16px",
                        "borderRadius": "20px",
                        "border": "1px solid #ddd",
                        "backgroundColor": "#fff",
                        "cursor": "pointer",
                        "fontSize": "14px",
                    },
                ),

                html.Div(style={"width": "20px"}),

                html.Div("Speed (ms):", style={"fontSize": "14px"}),
                dcc.Input(
                    id="speed", 
                    type="number", 
                    min=10, 
                    max=5000, 
                    step=10, 
                    value=80,
                    style={
                        "padding": "6px 12px",
                        "borderRadius": "12px",
                        "border": "1px solid #ddd",
                        "width": "80px",
                    },
                ),

                html.Div(style={"width": "20px"}),

                html.Div("Step:", style={"fontSize": "14px"}),
                dcc.Input(
                    id="step", 
                    type="number", 
                    min=1, 
                    max=200, 
                    step=1, 
                    value=5,
                    style={
                        "padding": "6px 12px",
                        "borderRadius": "12px",
                        "border": "1px solid #ddd",
                        "width": "60px",
                    },
                ),
            ],
        ),

        html.Div(style={"height": "12px"}),

        dcc.Slider(
            id="slider",
            min=0,
            max=1,
            step=1,
            value=0,
            marks={},
            tooltip={"placement": "bottom", "always_visible": True},
        ),

        html.Div(id="date-label", style={"marginTop": "8px", "color": "#444"}),

        # stores app state
        dcc.Store(id="playing", data=False),

        # timer
        dcc.Interval(id="timer", interval=80, n_intervals=0, disabled=True),
    ],
)

# Update slider bounds/marks when commodity changes
@app.callback(
    Output("slider", "max"),
    Output("slider", "marks"),
    Output("slider", "value", allow_duplicate=True),
    Input("commodity", "value"),
    prevent_initial_call=True,
)
def update_slider_bounds(ticker):
    _, dates, _, _ = load_dataset(ticker)
    n_dates = len(dates)
    if n_dates == 0:
        return 0, {}, 0
    # marks: first, middle, last to avoid clutter
    mid = n_dates // 2
    marks = {
        0: str(dates[0].date()),
        mid: str(dates[mid].date()),
        n_dates - 1: str(dates[-1].date()),
    }
    return n_dates - 1, marks, 0


# Update timer speed + enabled/disabled based on playing
@app.callback(
    Output("timer", "interval"),
    Output("timer", "disabled"),
    Input("speed", "value"),
    Input("playing", "data"),
)
def update_timer(speed, playing):
    speed = int(speed) if speed else 80
    playing = bool(playing)
    return speed, (not playing)


# Toggle play/pause
@app.callback(
    Output("playing", "data"),
    Input("toggle", "n_clicks"),
    State("playing", "data"),
    prevent_initial_call=True,
)
def toggle_play(n, playing):
    return not bool(playing)


# Prev/Next buttons move slider
@app.callback(
    Output("slider", "value", allow_duplicate=True),
    Input("prev", "n_clicks"),
    Input("next", "n_clicks"),
    State("slider", "value"),
    State("step", "value"),
    State("slider", "max"),
    prevent_initial_call=True,
)
def prev_next(prev, nxt, current, step, slider_max):
    step = int(step) if step else 1
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    slider_max = int(slider_max) if slider_max is not None else 0
    current = int(current)

    if trig == "prev":
        return max(0, current - step)
    if trig == "next":
        return min(slider_max, current + step)
    return no_update


# Auto-advance when timer ticks (only when playing)
@app.callback(
    Output("slider", "value", allow_duplicate=True),
    Input("timer", "n_intervals"),
    State("slider", "value"),
    State("slider", "max"),
    State("step", "value"),
    State("playing", "data"),
    prevent_initial_call=True,
)
def tick(n_intervals, current, slider_max, step, playing):
    if not playing:
        return no_update
    step = int(step) if step else 1
    slider_max = int(slider_max) if slider_max is not None else 0
    nxt = int(current) + step
    if nxt >= slider_max:
        return slider_max
    return nxt


# Update figure whenever slider moves or ticker changes
@app.callback(
    Output("graph", "figure"),
    Output("params-box", "children"),
    Output("date-label", "children"),
    Input("slider", "value"),
    Input("commodity", "value"),
)
def update_graph(i, ticker):
    df, dates, _, _ = load_dataset(ticker)
    n_dates = len(dates)
    if n_dates == 0:
        return go.Figure(), [html.Div("No data")], ""

    i = min(max(int(i), 0), n_dates - 1)
    fig = make_figure(ticker, i)
    d = dates[i].date()
    params = fit_params_for_idx(i, ticker)

    if params is None:
        params_display = [
            html.H4("NSS Parameters", style={"marginTop": "0", "marginBottom": "16px", "color": "#333"}),
            html.Div("Not available", style={"color": "#888"}),
        ]
    else:
        b0, b1, b2, b3, t1, t2 = params
        params_display = [
            html.H4("NSS Parameters", style={"marginTop": "0", "marginBottom": "16px", "color": "#333", "fontWeight": "600"}),
            html.Div([html.Strong("β₀: "), f"{b0:.4f}"], style={"marginBottom": "8px", "color": "#555"}),
            html.Div([html.Strong("β₁: "), f"{b1:.4f}"], style={"marginBottom": "8px", "color": "#555"}),
            html.Div([html.Strong("β₂: "), f"{b2:.4f}"], style={"marginBottom": "8px", "color": "#555"}),
            html.Div([html.Strong("β₃: "), f"{b3:.4f}"], style={"marginBottom": "8px", "color": "#555"}),
            html.Div([html.Strong("θ₁: "), f"{t1:.2f}"], style={"marginBottom": "8px", "color": "#555"}),
            html.Div([html.Strong("θ₂: "), f"{t2:.2f}"], style={"color": "#555"}),
        ]

    date_text = f"Selected date: {d} (index {i})"
    return fig, params_display, date_text


if __name__ == "__main__":
    app.run(debug=True)
