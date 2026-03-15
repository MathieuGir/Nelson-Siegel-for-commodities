from tqdm import tqdm
import pandas as pd
import numpy as np
from helpers.data_helpers import load_calibration_data
from helpers.plot_helpers import plot_term_structure, plot_term_structure_evolution
from helpers.nelson_curve_helpers import NS_rate, NS_estimation, plot_ns_fits



def ns_price_spread_for_date(df: pd.DataFrame, date_str: str, min_maturity: int = 20, max_maturity: int = 500, verbosity=False):
    """
    Return per-contract spreads for a single date using 4-parameter Nelson-Siegel model.
    
    Parameters:
    - df: DataFrame with futures data indexed by date
    - date_str: date string to process
    - min_maturity: minimum time-to-maturity to include
    - max_maturity: maximum time-to-maturity to include
    - verbosity: if True, print fitting details
    
    Returns: list of dicts with contract, prices, spreads, and fitted NS parameters (beta0, beta1, beta2, lambda)
    """
    if date_str not in df.index:
        raise ValueError(f"{date_str} not found in DataFrame index.")
    day_data = df.loc[date_str]
    day_data = day_data[(day_data["time_to_maturity"] >= min_maturity) & (day_data["time_to_maturity"] <= max_maturity)]
    if day_data.empty:
        return []
    maturities = day_data["time_to_maturity"].to_numpy()
    prices = day_data["price"].to_numpy()
    params = NS_estimation(maturities, prices, min_maturity=min_maturity, max_maturity=max_maturity, verbosity=verbosity)
    modeled = np.exp(NS_rate(maturities, params))
    entries = []
    for contract, real_price, model_price in zip(day_data["contract"].to_numpy(), prices, modeled):
        spread = model_price - real_price
        # rel error bps
        rel_error_bps = spread / real_price * 10000 if real_price != 0 else np.nan
        entries.append({
            "contract": contract,
            "real_price": float(real_price),
            "model_price": float(model_price),
            "spread": float(spread),
            "error_bps": float(rel_error_bps),
            # Include fitted NS parameters for this date (4-parameter model)
            "beta0": float(params[0]),    # level
            "beta1": float(params[1]),    # slope
            "beta2": float(params[2]),    # curvature
            "lambda": float(params[3]),   # decay parameter
        })
    return entries



# Apply NS spread for all dates in the steepener window and stack results into one DataFrame

all_commodity_contracts = ['CC', 'KC', 'SB', 'CT']
# all_commodity_contracts = ['CT']
model_spread_start_date = '1980-01-01'
model_spread_end_date = '2025-12-31'

for commo_ticker in all_commodity_contracts:
    steepener_data = load_calibration_data(commo_ticker=commo_ticker, start_date=model_spread_start_date, end_date=model_spread_end_date)

    all_spreads = []
    for date_str in tqdm(steepener_data.index.unique()):
        spreads_for_date = ns_price_spread_for_date(steepener_data, date_str, min_maturity=10, max_maturity=300)
        # print(f"Processed spreads for date: {date_str}, number of contracts: {len(spreads_for_date)}")
        for entry in spreads_for_date:
            all_spreads.append({
                "date": pd.to_datetime(date_str),
                "contract": entry["contract"],
                "time_to_maturity": steepener_data.loc[date_str][steepener_data.loc[date_str]["contract"] == entry["contract"]]["time_to_maturity"].values[0],
                "open_interest": steepener_data.loc[date_str][steepener_data.loc[date_str]["contract"] == entry["contract"]]["open_interest"].values[0],
                "real_price": entry["real_price"],
                "model_price": entry["model_price"],
                "spread": entry["spread"],
                "error_bps": entry["error_bps"],
                # Propagate NS parameters (4-parameter model) for this date across each contract row
                "beta0": entry["beta0"],
                "beta1": entry["beta1"],
                "beta2": entry["beta2"],
                "lambda": entry["lambda"],
            })
    
    NS_model_spread_df = pd.DataFrame(all_spreads)
    NS_model_spread_df.set_index('date', inplace=True)
    print(f"NS model spreads for {commo_ticker}:")
    print(NS_model_spread_df)
    NS_model_spread_df.to_csv(f"data/{commo_ticker}_NS_price_spreads.csv")
