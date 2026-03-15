"""
Interactive Nelson-Siegel Curve Parameter Explorer
Adjust beta0, beta1, beta2, and lambda to see how the curve changes in real-time

To run, simply do:
streamlit run dashboard/interactive_ns_explorer.py

"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helpers.nelson_curve_helpers import NS_rate, L1_NS, L2_NS

# Page config
st.set_page_config(page_title="Nelson-Siegel Explorer", layout="wide")
st.title("🎛️ Nelson-Siegel Curve Interactive Explorer")
st.markdown("""
Adjust the parameters to see how the Nelson-Siegel curve changes in real-time.

**Formula:** y(τ) = β₀ + β₁·L₁(τ/λ) + β₂·L₂(τ/λ)
""")

# Create columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Parameters")
    
    # Create sliders for parameters
    beta0 = st.slider(
        "β₀ (Level factor)",
        min_value=-5.0,
        max_value=10.0,
        value=5.5,
        step=0.1,
        help="Controls the overall level of the curve (long-term rate)"
    )
    
    beta1 = st.slider(
        "β₁ (Slope factor)",
        min_value=-5.0,
        max_value=5.0,
        value=-2.0,
        step=0.1,
        help="Controls the slope (difference between short and long-term rates)"
    )
    
    beta2 = st.slider(
        "β₂ (Curvature factor)",
        min_value=-5.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Controls the curvature/hump in the middle of the curve"
    )
    
    lambda_param = st.slider(
        "λ (Decay parameter)",
        min_value=1.0,
        max_value=200.0,
        value=50.0,
        step=1.0,
        help="Controls how quickly the loadings decay (impacts where hump occurs)"
    )
    
    st.divider()
    st.subheader("📏 Plot Settings")
    
    max_maturity = st.slider(
        "Maximum maturity to plot (days)",
        min_value=50.0,
        max_value=500.0,
        value=500.0,
        step=10.0,
        help="Set the maximum time-to-maturity displayed on the curve"
    )
    
    st.divider()
    st.subheader("📊 Statistics")
    
    # Display parameter info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("β₀", f"{beta0:.3f}", "Level")
        st.metric("β₂", f"{beta2:.3f}", "Curvature")
    with col_info2:
        st.metric("β₁", f"{beta1:.3f}", "Slope")
        st.metric("λ", f"{lambda_param:.1f}", "Decay")

with col2:
    # Generate maturities for the curve
    maturities = np.linspace(0.1, max_maturity, 500)
    
    # Calculate rates using the NS formula
    rates = NS_rate(maturities, (beta0, beta1, beta2, lambda_param))
    
    # Calculate individual components
    x = maturities / lambda_param
    level_component = np.full_like(maturities, beta0)
    slope_component = beta1 * L1_NS(x)
    curvature_component = beta2 * L2_NS(x)
    
    # Create the main plot
    fig = go.Figure()
    
    # Add the main NS curve
    fig.add_trace(go.Scatter(
        x=maturities,
        y=rates,
        mode='lines',
        name='Nelson-Siegel Curve',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>Maturity:</b> %{x:.1f}<br><b>Rate:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Add components (optional - can toggle)
    fig.add_trace(go.Scatter(
        x=maturities,
        y=level_component,
        mode='lines',
        name='β₀ (Level)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        visible='legendonly',
        hovertemplate='<b>Maturity:</b> %{x:.1f}<br><b>Level:</b> %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=slope_component,
        mode='lines',
        name='β₁·L₁ (Slope)',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        visible='legendonly',
        hovertemplate='<b>Maturity:</b> %{x:.1f}<br><b>Slope:</b> %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=curvature_component,
        mode='lines',
        name='β₂·L₂ (Curvature)',
        line=dict(color='#d62728', width=2, dash='dash'),
        visible='legendonly',
        hovertemplate='<b>Maturity:</b> %{x:.1f}<br><b>Curvature:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Nelson-Siegel Yield Curve',
        xaxis_title='Maturity (days)',
        yaxis_title='Rate / Yield',
        template='plotly_white',
        hovermode='x unified',
        height=600,
        showlegend=True,
        xaxis=dict(range=[0, max_maturity]),
        yaxis=dict(range=[-0.5, 10]),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Additional analysis section
st.subheader("📈 Curve Analysis")

col_analysis1, col_analysis2, col_analysis3 = st.columns(3)

with col_analysis1:
    # Short-term rate
    short_maturity = min(10.0, max_maturity * 0.2)
    short_rate = NS_rate(np.array([short_maturity]), (beta0, beta1, beta2, lambda_param))[0]
    st.metric(f"{short_maturity:.0f}-day Rate", f"{short_rate:.4f}")

with col_analysis2:
    # Medium-term rate
    medium_maturity = min(100.0, max_maturity * 0.4)
    medium_rate = NS_rate(np.array([medium_maturity]), (beta0, beta1, beta2, lambda_param))[0]
    st.metric(f"{medium_maturity:.0f}-day Rate", f"{medium_rate:.4f}")

with col_analysis3:
    # Long-term rate
    long_maturity = max_maturity * 0.9
    long_rate = NS_rate(np.array([long_maturity]), (beta0, beta1, beta2, lambda_param))[0]
    st.metric(f"{long_maturity:.0f}-day Rate", f"{long_rate:.4f}")

# Slope and curvature metrics
col_metrics1, col_metrics2 = st.columns(2)

with col_metrics1:
    slope = long_rate - short_rate
    st.metric("Curve Slope", f"{slope:.4f}", 
              delta="Normal" if slope > 0 else "Inverted",
              delta_color="normal" if slope > 0 else "inverse")

with col_metrics2:
    # Hump location (where L2 is maximum, roughly at maturity = 1.5*lambda)
    hump_location = 1.5 * lambda_param
    st.metric("Hump Location (approx)", f"{hump_location:.0f} days")

st.divider()

# Information section
with st.expander("ℹ️ About the Nelson-Siegel Model"):
    st.markdown("""
    ### Nelson-Siegel Model
    
    The Nelson-Siegel model is a parametric model for yield curves. It's widely used in finance 
    to model commodity futures curves, interest rates, and other term structures.
    
    **Formula:** y(τ) = β₀ + β₁·L₁(τ/λ) + β₂·L₂(τ/λ)
    
    Where:
    - **τ** is time to maturity
    - **β₀** (Level): Controls the long-term level of the curve. Higher β₀ → higher overall rates
    - **β₁** (Slope): Controls the initial slope. Negative β₁ → downward slope (normal), Positive → upward slope
    - **β₂** (Curvature): Controls the hump/hump. Positive β₂ → hump in the middle, Negative → inverted hump
    - **λ** (Decay): Controls how quickly the loadings decay. Lower λ → hump closer to short end, Higher λ → hump farther out
    
    ### Loading Functions
    - L₁(x) = (1 - e^(-x)) / x → 1 as x→0, → 0 as x→∞
    - L₂(x) = (1 - (1+x)e^(-x)) / x → 0.5 as x→0, → 0 as x→∞
    
    ### Typical Interpretations
    - **Normal yield curve**: β₁ < 0 (upward slope from short to long), β₂ > 0 (slight hump)
    - **Inverted curve**: β₁ > 0 (downward slope)
    - **Flat curve**: β₁ ≈ 0, small β₂
    """)
