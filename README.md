# Nelson-Siegel Model for Commodity Futures

A quantitative framework for modeling and trading commodity futures curves using the **Nelson-Siegel term structure model**.

## Project Overview

This project adapts the Nelson-Siegel (1987) model—originally designed for government bond yield curves—to **commodity futures curves**. The core idea is to represent the entire price term structure of a commodity with just **four parameters** (β₀, β₁, β₂, λ), then use **daily changes** in these parameters as **cross-sectional trading signals** across multiple commodities.

### Commodities

- **KC**: Coffee (Arabica) — ICE Futures US
- **CC**: Cocoa — ICE Futures US  
- **SB**: Sugar — ICE Futures US
- **CT**: Cotton — ICE Futures US

---

## Directory Structure

```
Nelson Siegel for commodities/
├── README.md                           # This file
├── main.ipynb                          # Main analysis & backtest pipeline
├── pyproject.toml                      # Project dependencies
├── helpers/                            # Core utility modules
│   ├── nelson_curve_helpers.py         # NS fitting & evaluation
│   ├── data_helpers.py                 # Data loading & preprocessing
│   └── plot_helpers.py                 # Visualization utilities
├── dashboard/                          # Interactive tools
│   ├── interactive_ns_explorer.py      # Parameter exploration (Streamlit)
│   ├── nss_streamlit_app.py           # Dashboard application
│   └── dash_nss_app.py                 # Alternative Dash app
├── data/                               # Processed data (not in GitHub)
│   ├── {TICKER}.csv                    # Raw futures prices
│   ├── {TICKER}_NS_price_spreads.csv   # Fitted NS parameters
│   └── DTB3.csv                        # Daily risk-free rate
└── videos/                             # Generated animations
```

---

## Core Components

### 1. **Data Layer** (`helpers/data_helpers.py`)

Loads and preprocesses commodity futures data:
- Reads daily settlement prices for all listed contracts
- Computes **time-to-maturity** using commodity expiry conventions (3rd Wednesday of delivery month)
- Aligns with daily risk-free rates (DTB3 when available)
- Filters contracts (10–500 business days to maturity for fitting)

**Output**: Clean panel of (ticker, date, contract, price, maturity) tuples

### 2. **Nelson-Siegel Fitting** (`helpers/nelson_curve_helpers.py`)

Fits the NS model on each commodity-date pair:

**Model:**
$$\ln P(\tau) = \beta_0 + \beta_1 \cdot L_1\left(\frac{\tau}{\lambda}\right) + \beta_2 \cdot L_2\left(\frac{\tau}{\lambda}\right)$$

Where:
- **β₀ (Level)**: Long-term level of the curve
- **β₁ (Slope)**: Initial slope (contango vs backwardation)
- **β₂ (Curvature)**: Hump/valley in the middle of the curve
- **λ (Decay)**: Controls where the hump occurs

**Algorithm:**
- Non-linear least squares (L-BFGS-B algorithm)
- Minimizes squared residuals in log-price space
- Numerically stable Taylor approximations for loading functions near zero

**Output**: Time series of (β₀, β₁, β₂, λ) for each commodity

### 3. **Trading Strategy** (`main.ipynb` — Section 4)

#### Signal Generation

Cross-sectional ranking of commodities on **daily parameter changes**:

1. **Slope Sleeve**: Signal = Δβ₁ (change in slope parameter)
   - Long: commodity with highest Δβ₁  
   - Short: commodity with lowest Δβ₁  
   - Risk Factor: **Slope Spread** = F₁ − F₄

2. **Butterfly Sleeve**: Signal = Δβ₂ (change in curvature parameter)
   - Long: commodity with lowest Δβ₂ (inverted via `BUTTERFLY_SIGN`)  
   - Short: commodity with highest Δβ₂  
   - Risk Factor: **Butterfly Spread** = −F₁ + 2F₂ − F₄

## Interactive Tools

### **Interactive NS Parameter Explorer**

Streamlit app to visualize how β₀, β₁, β₂, λ affect curve shape:

```bash
streamlit run dashboard/interactive_ns_explorer.py
```

- Adjust parameters with sliders
- See curve components (level, slope, curvature) in real-time
- View curve statistics (short/medium/long-term rates, hump location)

---

## Workflow

### 1. Data Preparation
```bash
# Place raw CSV files in data/ folder (not in GitHub)
# Expected columns: contract, price, open_interest, volume
# Indexed by: date
```

### 2. NS Fitting & Analysis
```python
# Run main.ipynb cells 1–3 to:
# - Load data for all commodities
# - Fit NS model daily
# - Plot parameter evolution
```

### 3. Backtest Execution
```python
# Run main.ipynb Section 4 to:
# - Generate daily trading signals
# - Compute vol-budgeted positions
# - Run weekly rebalancing
# - Display NAV and performance stats
```

### 4. Interactive Exploration
```bash
# Explore parameter behavior in real-time
streamlit run dashboard/interactive_ns_explorer.py
```


## Dependencies

See `pyproject.toml` for full dependency list. Key packages:
- **numpy, scipy**: Numerical optimization & linear algebra
- **pandas**: Data manipulation & time series
- **plotly**: Interactive visualizations
- **streamlit**: Dashboard framework
- **scikit-optimize** (optional): Hyperparameter tuning

### Setup with UV

This project uses [uv](https://github.com/astral-sh/uv) — a fast, modern Python package manager.

**Install uv** (if not already installed):
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or install via package manager (homebrew, etc.)
```

**Setup project environment:**
```bash
# Sync dependencies (creates a virtual environment automatically)
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Run main notebook or scripts
uv run python main.ipynb
```

**Install additional dependencies:**
```bash
uv pip install package_name
```

---

## Files NOT Included

The following data and outputs are **not** in the GitHub repository for confidentiality:

- `data/{TICKER}.csv` — Raw commodity futures prices
- `data/{TICKER}_NS_price_spreads.csv` — Pre-fitted NS parameters generated with computeNSspreads.py
- `data/DTB3.csv` — Daily risk-free rates (obtained from FRED)