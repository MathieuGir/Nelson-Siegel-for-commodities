"""
Backtesting script for ML-based trading strategy using walk-forward predictions.
Loads predictions from data/backtest_prediction_tree/ and computes P&L.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

# Configuration
TICKERS = ["KC", "CC", "SB", "CT"]
DATA_DIR = Path("data")
PRED_DIR = DATA_DIR / "backtest_prediction_tree"

HORIZON = 1  # 1-day holding period

# Trading Parameters
SIGNAL_TYPE = "relative"  # "absolute" or "relative" (vs daily median)
LONG_THRESHOLD = 0.0001   # if absolute: pred_ret_h > threshold → long
SHORT_THRESHOLD = -0.0001 # if absolute: pred_ret_h < threshold → short
SCALE_SIZE = 1e13         # position size scale factor
MAX_CONTRACTS = 5000      # max contracts per trade

STARTING_CAPITAL = 1_000_000
RISK_FREE_RATE = 0.02     # for Sharpe computation

VERBOSE = True


def load_predictions(ticker: str) -> pd.DataFrame:
    """Load walk-forward predictions for a ticker."""
    path = PRED_DIR / f"{ticker}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_price_spreads(ticker: str) -> pd.DataFrame:
    """Load original price/spread data for a ticker."""
    path = DATA_DIR / f"{ticker}_NSS_price_spreads.csv"
    if not path.exists():
        raise FileNotFoundError(f"Price spreads not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    return df


def prepare_data(ticker: str) -> pd.DataFrame:
    """Merge predictions with price data and prepare for backtesting."""
    if VERBOSE:
        print(f"  Loading data for {ticker}...")

    # Load data
    pred_df = load_predictions(ticker)
    price_df = load_price_spreads(ticker)

    # Merge on date and contract
    merged = pred_df.merge(
        price_df[["date", "contract", "real_price", "time_to_maturity", "error_bps"]],
        on=["date", "contract"],
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError(f"No matching rows after merge for {ticker}")

    # Sort by contract and date
    merged = merged.sort_values(["contract", "date"]).reset_index(drop=True)

    # Get next day's price for exit
    merged["price_today"] = merged["real_price"]
    merged["price_exit"] = merged.groupby("contract")["real_price"].shift(-HORIZON)
    merged["exit_date"] = merged.groupby("contract")["date"].shift(-HORIZON)

    # Drop rows without exit price
    merged = merged.dropna(subset=["price_exit"]).copy()

    if VERBOSE:
        print(f"    Prepared {len(merged)} trading opportunities")

    return merged


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals based on predictions."""
    out = df.copy()

    if SIGNAL_TYPE == "relative":
        # Relative: vs daily median prediction
        out["pred_median"] = out.groupby("date")["pred_ret_h"].transform("median")
        out["pred_rel"] = out["pred_ret_h"] - out["pred_median"]
        out["direction"] = np.sign(out["pred_rel"])
        # Kill tiny signals
        tiny_thr = 1e-8
        out.loc[out["pred_rel"].abs() < tiny_thr, "direction"] = 0

    elif SIGNAL_TYPE == "absolute":
        # Absolute: vs fixed thresholds
        out["direction"] = 0
        out.loc[out["pred_ret_h"] > LONG_THRESHOLD, "direction"] = 1
        out.loc[out["pred_ret_h"] < SHORT_THRESHOLD, "direction"] = -1
    else:
        raise ValueError(f"Unknown SIGNAL_TYPE: {SIGNAL_TYPE}")

    return out


def size_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Size positions based on signal strength."""
    out = df.copy()

    if SIGNAL_TYPE == "relative":
        qty_raw = SCALE_SIZE * out["pred_rel"].abs()
    else:
        qty_raw = SCALE_SIZE * out["pred_ret_h"].abs()

    out["qty"] = np.floor(qty_raw).clip(0, MAX_CONTRACTS).astype(int)
    # No position if no direction
    out.loc[out["direction"] == 0, "qty"] = 0

    return out


def compute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P&L for each trade."""
    out = df.copy()

    # P&L = direction * qty * (exit_price - entry_price)
    out["pnl_trade"] = (
        out["direction"] * out["qty"] * (out["price_exit"] - out["price_today"])
    )

    return out


def backtest_ticker(ticker: str) -> tuple:
    """Run backtest for a single ticker. Returns (trade_log, portfolio)."""
    if VERBOSE:
        print(f"\n=== Backtesting {ticker} ===")

    # Prepare data
    df = prepare_data(ticker)

    # Generate signals
    df = generate_signals(df)

    # Size positions
    df = size_positions(df)

    # Compute P&L
    df = compute_pnl(df)

    # Filter to trades only
    trade_log = df[df["qty"] > 0].copy()
    trade_log = trade_log.rename(columns={
        "date": "entry_date",
        "price_today": "entry_price",
        "exit_date": "exit_date",
        "price_exit": "exit_price",
    })

    if len(trade_log) == 0:
        if VERBOSE:
            print(f"  No trades generated for {ticker}")
        return trade_log, None

    # Aggregate to daily portfolio level
    portfolio = (
        trade_log
        .groupby("entry_date")["pnl_trade"]
        .sum()
        .to_frame("daily_pnl")
    )
    portfolio["cum_pnl"] = portfolio["daily_pnl"].cumsum()
    portfolio["equity"] = STARTING_CAPITAL + portfolio["cum_pnl"]

    if VERBOSE:
        total_trades = len(trade_log)
        total_pnl = trade_log["pnl_trade"].sum()
        win_count = (trade_log["pnl_trade"] > 0).sum()
        win_rate = 100 * win_count / total_trades if total_trades > 0 else 0
        print(f"  Total trades: {total_trades}")
        print(f"  Total P&L: ${total_pnl:,.2f}")
        print(f"  Win rate: {win_rate:.1f}%")

    return trade_log, portfolio


def compute_stats(portfolio: pd.DataFrame) -> dict:
    """Compute performance statistics."""
    if portfolio is None or len(portfolio) == 0:
        return {}

    daily_ret = portfolio["equity"].pct_change().dropna()
    total_return = portfolio["equity"].iloc[-1] - STARTING_CAPITAL
    total_ret_pct = (total_return / STARTING_CAPITAL) * 100
    annualized_return = (
        (portfolio["equity"].iloc[-1] / STARTING_CAPITAL) ** (252 / len(portfolio)) - 1
    ) * 100
    volatility = daily_ret.std() * np.sqrt(252) * 100
    rf_daily = RISK_FREE_RATE / 252
    sharpe = (
        (daily_ret.mean() - rf_daily) / daily_ret.std() * np.sqrt(252)
        if daily_ret.std() > 0 else 0
    )
    max_dd = (
        (portfolio["equity"].cummax() - portfolio["equity"])
        / portfolio["equity"].cummax()
    ).max() * 100

    return {
        "total_return_$": total_return,
        "total_return_%": total_ret_pct,
        "annualized_return_%": annualized_return,
        "volatility_%": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown_%": max_dd,
    }


def plot_equity_curve(portfolio: pd.DataFrame, title: str):
    """Plot equity curve."""
    if portfolio is None or len(portfolio) == 0:
        print("  No equity curve to plot (no trades).")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio.index,
        y=portfolio["equity"],
        mode="lines",
        name="Equity",
        line=dict(width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_white",
        hovermode="x unified"
    )
    fig.show()


def main():
    print(f"\n{'='*60}")
    print(f"ML Strategy Backtest")
    print(f"Tickers: {TICKERS}")
    print(f"Signal type: {SIGNAL_TYPE}")
    print(f"Horizon: {HORIZON} day(s)")
    print(f"Starting capital: ${STARTING_CAPITAL:,.0f}")
    print(f"{'='*60}\n")

    all_stats = {}
    all_trades = {}

    for ticker in TICKERS:
        try:
            trade_log, portfolio = backtest_ticker(ticker)
            all_trades[ticker] = trade_log
            stats = compute_stats(portfolio)
            all_stats[ticker] = stats

            if stats:
                print(f"\n  Performance for {ticker}:")
                for key, val in stats.items():
                    if "%" in key or "ratio" in key:
                        print(f"    {key}: {val:.2f}")
                    else:
                        print(f"    {key}: ${val:,.2f}" if "$" in key else f"    {key}: {val:.2f}")

        except Exception as e:
            print(f"\n  Error backtesting {ticker}: {e}")

    # Combine all trades across tickers
    if all_trades:
        combined_trades = pd.concat(
            [v for v in all_trades.values() if len(v) > 0],
            axis=0,
            ignore_index=True
        )

        if len(combined_trades) > 0:
            # Portfolio across all tickers
            combined_portfolio = (
                combined_trades
                .groupby("entry_date")["pnl_trade"]
                .sum()
                .to_frame("daily_pnl")
            )
            combined_portfolio["cum_pnl"] = combined_portfolio["daily_pnl"].cumsum()
            combined_portfolio["equity"] = STARTING_CAPITAL + combined_portfolio["cum_pnl"]

            combined_stats = compute_stats(combined_portfolio)

            print(f"\n{'='*60}")
            print(f"Combined Portfolio (All Tickers):")
            print(f"Total trades: {len(combined_trades)}")
            for key, val in combined_stats.items():
                if "%" in key or "ratio" in key:
                    print(f"  {key}: {val:.2f}")
                else:
                    print(f"  {key}: ${val:,.2f}" if "$" in key else f"  {key}: {val:.2f}")
            print(f"{'='*60}\n")

            # Plot combined equity curve
            plot_equity_curve(
                combined_portfolio,
                f"ML Strategy Equity Curve (All Tickers, {SIGNAL_TYPE} signal)"
            )

            # Save results
            results_dir = DATA_DIR / "backtest_results"
            results_dir.mkdir(exist_ok=True)
            combined_trades.to_csv(results_dir / "trades.csv", index=False)
            combined_portfolio.to_csv(results_dir / "portfolio.csv")
            print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
