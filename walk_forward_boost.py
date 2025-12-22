from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import clone
import lightgbm as lgb  # <-- LightGBM

# Configuration
TICKERS = ["KC", "CC", "SB", "CT"]
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "backtest_prediction_tree"

# Optional date filters; set to None to use the full sample
BACKTEST_START = None  # e.g., "1989-01-03"
BACKTEST_END = None

HORIZON = 1
MIN_TRAIN_DAYS = 252
RETRAIN_EVERY = 5
ROLL_WIN = 60
ROLL_MIN = 30
# Only use the most recent N calendar days for training (None = use all history)
TRAIN_WINDOW_DAYS = 3 * 365

# Verbosity flag
VERBOSE = True

FEATURE_COLS = [
    "err", "err_abs", "err_z", "err_z_abs",
    "ttm", "ttm_inv",
    "bucket_front", "bucket_belly", "bucket_back",
    "beta0", "beta1", "beta2", "beta3", "theta1", "theta2",
]
TARGET_COL = "return_h"


def load_price_spreads(ticker: str) -> pd.DataFrame:
    if VERBOSE:
        print(f"  Loading {ticker} price spreads...")
    path = DATA_DIR / f"{ticker}_NSS_price_spreads.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if VERBOSE:
        print(f"    Loaded {len(df)} rows")
    if BACKTEST_START:
        df = df[df.index >= BACKTEST_START]
    if BACKTEST_END:
        df = df[df.index <= BACKTEST_END]
    return df


def prepare_base_df(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if VERBOSE:
        print(f"  Preparing base dataframe...")
    df = df.copy()
    # Ensure date is a column
    df = df.reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    df = df.sort_values(["contract", "date"]).reset_index(drop=True)

    df["price_h"] = df.groupby("contract")["real_price"].shift(-horizon)
    df["return_h"] = (df["price_h"] - df["real_price"]) / df["real_price"]

    df = df.dropna(subset=["price_h", "return_h"]).copy()
    if VERBOSE:
        print(f"    Prepared {len(df)} rows with returns")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if VERBOSE:
        print(f"  Engineering features...")
    out = df.copy()

    out["err"] = out["error_bps"]
    out["err_abs"] = out["error_bps"].abs()
    out["ttm"] = out["time_to_maturity"]
    out["ttm_inv"] = 1.0 / (out["time_to_maturity"] + 1.0)

    group = out.groupby("contract")["error_bps"]
    out["err_mean_roll"] = (
        group.rolling(ROLL_WIN, min_periods=ROLL_MIN).mean().reset_index(level=0, drop=True)
    )
    out["err_std_roll"] = (
        group.rolling(ROLL_WIN, min_periods=ROLL_MIN).std().reset_index(level=0, drop=True)
    )
    valid = out["err_std_roll"] > 0
    out["err_z"] = 0.0
    out.loc[valid, "err_z"] = (
        (out.loc[valid, "error_bps"] - out.loc[valid, "err_mean_roll"]) / out.loc[valid, "err_std_roll"]
    )
    out["err_z_abs"] = out["err_z"].abs()

    out["bucket_front"] = (out["ttm"] < 80).astype(int)
    out["bucket_belly"] = ((out["ttm"] >= 80) & (out["ttm"] < 160)).astype(int)
    out["bucket_back"] = (out["ttm"] >= 160).astype(int)

    out = out.dropna(subset=["err_mean_roll", "err_std_roll"]).copy()
    if VERBOSE:
        print(f"    Created {len(out)} rows with features")
    return out


def walk_forward_train(df_feat: pd.DataFrame) -> pd.DataFrame:
    if VERBOSE:
        print(f"  Starting walk-forward training...")
    missing = [c for c in FEATURE_COLS if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    df_feat = df_feat.copy()
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    unique_dates = df_feat["date"].sort_values().unique()
    total_dates = len(unique_dates)

    if VERBOSE:
        print(f"    Total unique dates: {total_dates}")

    # LightGBM base model
    base_model = lgb.LGBMRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=-1,          # let num_leaves control complexity
        num_leaves=31,
        subsample=0.8,         # row subsampling
        colsample_bytree=0.8,  # feature subsampling
        reg_lambda=1.0,
        objective="regression",
        random_state=42,
        n_jobs=-1,
    )

    model = None
    preds = []

    for i, current_date in tqdm(
        enumerate(unique_dates),
        total=total_dates,
        desc="Walk-forward",
        disable=not VERBOSE
    ):
        # restrict history to a rolling calendar window if configured
        if TRAIN_WINDOW_DAYS:
            cutoff = current_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
            hist_mask = (df_feat["date"] < current_date) & (df_feat["date"] >= cutoff)
        else:
            hist_mask = df_feat["date"] < current_date

        pred_mask = df_feat["date"] == current_date

        if hist_mask.sum() < MIN_TRAIN_DAYS:
            continue
        if pred_mask.sum() == 0:
            continue

        X_train = df_feat.loc[hist_mask, FEATURE_COLS].astype(np.float32)
        y_train = df_feat.loc[hist_mask, TARGET_COL].astype(np.float32)
        X_pred  = df_feat.loc[pred_mask, FEATURE_COLS].astype(np.float32)


        # Train / retrain LightGBM
        if (model is None) or (i % RETRAIN_EVERY == 0):
            model = clone(base_model)
            model.fit(X_train, y_train)

        # Predict for current date
        y_pred = model.predict(X_pred)

        tmp = df_feat.loc[pred_mask, ["date", "contract"]].copy()
        tmp["pred_ret_h"] = y_pred
        preds.append(tmp)

    if not preds:
        if VERBOSE:
            print(f"    No predictions generated.")
        return pd.DataFrame(columns=["date", "contract", "pred_ret_h"])

    pred_df = pd.concat(preds, axis=0).reset_index(drop=True)
    if VERBOSE:
        print(f"    Walk-forward complete: {len(pred_df)} predictions generated")
    return pred_df


def process_ticker(ticker: str) -> Path:
    print(f"\n=== Processing {ticker} ===")
    df = load_price_spreads(ticker)
    df = prepare_base_df(df, horizon=HORIZON)
    df_feat = make_features(df)

    # Ensure NSS parameter columns exist
    needed_params = ["beta0", "beta1", "beta2", "beta3", "theta1", "theta2"]
    for col in needed_params:
        if col not in df_feat.columns:
            raise ValueError(f"Column '{col}' not found in input for {ticker}.")

    pred_df = walk_forward_train(df_feat)
    pred_df["ticker"] = ticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{ticker}_predictions.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"✓ Saved predictions to {out_path} ({len(pred_df)} rows)")
    return out_path


def main():
    print(f"\n{'='*60}")
    print(f"Walk-Forward ML Training (LightGBM)")
    print(f"Tickers: {TICKERS}")
    print(f"Train window: {TRAIN_WINDOW_DAYS} days" if TRAIN_WINDOW_DAYS else f"Train window: All history")
    print(f"Min train days: {MIN_TRAIN_DAYS}")
    print(f"Retrain every: {RETRAIN_EVERY} days")
    print(f"{'='*60}\n")

    for ticker in TICKERS:
        process_ticker(ticker)

    print(f"\n{'='*60}")
    print(f"All tickers processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
