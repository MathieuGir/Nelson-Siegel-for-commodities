import pandas as pd
import datetime
from datetime import date
import numpy as np
import time 



# def load_commodity_prices(file_path, date_col='date', measure_col='measure', measure_value='close', start_date=None, end_date=None):
#     """
#     Load the CSV file and filter rows where the measure column matches the specified value.
    
#     Parameters:
#     - file_path: str, path to the CSV file
#     - date_col: str, name of the date column
#     - measure_col: str, name of the measure column
#     - measure_value: str, value to filter the measure column
    
#     Returns:
#     - pd.DataFrame: filtered DataFrame indexed by date and sorted
#     """
#     # Load the csv file
#     data = pd.read_csv(file_path, sep=',')
    
#     # Filter rows where the measure column matches the specified value, and drop measure column
#     # data = data[data[measure_col] == measure_value].drop(columns=[measure_col, 'time'])
#     data = data[data[measure_col] == measure_value]
    
#     #for each date and contract, keep only latest time entry
#     data = data.sort_values(by=['date', 'time'], ascending=[True, True])
#     data = data.drop_duplicates(subset=['date', 'contract'], keep='last')
#     data = data.drop(columns=[measure_col, 'time'])
    

#     # Rename the value column to price
#     data.rename(columns={'value': 'price'}, inplace=True)
    
#     # Convert date column to datetime format
#     data[date_col] = pd.to_datetime(data[date_col], format='%Y-%m-%d')
    
#     # Index by date and sort it
#     data = data.set_index(date_col).sort_index()

#     # Drop NAN values and values equal to 0
#     zero_count = (data == 0).sum()
#     if zero_count.any():
#         print(f"Dropping rows with zero values:\n{zero_count[zero_count > 0]}")
#     data = data[data['price'] != 0]
#     na_count = data.isna().sum()
#     if na_count.any():
#         print(f"Dropping rows with NA values:\n{na_count[na_count > 0]}")
#     data.dropna(inplace=True)
#     return data


def load_commodity_prices(
    file_path: str,
    date_col: str = "date",
    time_col: str = "time",
    contract_col: str = "contract",
    measure_col: str = "measure",
    value_col: str = "value",
    measures: tuple = ("close", "open_interest", "volume"),
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Load commodity data from a long format file with columns:
      [contract, measure, value, time, date]
    and return a wide DataFrame indexed by date with columns:
      contract, price, open_interest, volume

    Rules:
    - Keep only requested measures (default: close, open_interest, volume)
    - For each (date, contract, measure): keep the latest 'time'
    - Pivot measures -> columns
    - Rename 'close' -> 'price'
    - Drop rows where price is 0 or NA
    """

    df = pd.read_csv(file_path, sep=",")

    # Parse date/time
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 'time' is like "2023-12-07T21:30:09" (ISO-ish) -> parse to datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Optional date filtering (on date column)
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    # Keep only the measures you care about
    df = df[df[measure_col].isin(measures)].copy()

    # Keep latest time per (date, contract, measure)
    df = df.sort_values(by=[date_col, contract_col, measure_col, time_col])
    df = df.drop_duplicates(subset=[date_col, contract_col, measure_col], keep="last")

    # Pivot to wide: one row per (date, contract)
    wide = df.pivot(index=[date_col, contract_col], columns=measure_col, values=value_col).reset_index()

    # Flatten columns (pivot makes measure_col a columns index)
    wide.columns.name = None

    # Rename close -> price
    if "close" in wide.columns:
        wide = wide.rename(columns={"close": "price"})

    # Ensure numeric
    for c in ["price", "open_interest", "volume"]:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")

    # Drop bad prices
    wide = wide.dropna(subset=["price"])
    wide = wide[wide["price"] != 0]

    # Index by date (as you had before), keep contract as a column
    wide = wide.set_index(date_col).sort_index()

    # Optional: if you require open_interest/volume present, uncomment:
    # wide = wide.dropna(subset=["open_interest", "volume"])

    return wide


MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}


def get_time_to_maturity(current_date, contract_name, verbosity=False):
    """
    Returns the number of BUSINESS DAYS until the futures contract expires.
    Expiry = third Wednesday of the delivery month.
    """
    # Convert to Timestamp
    current_date = pd.to_datetime(current_date).normalize()

    # Parse contract
    month_code = contract_name[0]
    year = int(contract_name[1:])
    month = MONTH_MAP[month_code]

    # Generate all Wednesdays of that month
    
    # Only get the 3rd Wednesday of the month
    

    first_day = pd.Timestamp(year, month, 1)
    all_wednesdays = pd.date_range(first_day, first_day + pd.offsets.MonthEnd(0), freq='W-WED')
    expiration_date = all_wednesdays[2]

    # Compute business days difference
    business_days = np.busday_count(current_date.date(), expiration_date.date())

    if verbosity:
        print(f"Expiration date for contract {contract_name} is {expiration_date.date()}")
        print(f"Time to maturity from {current_date.date()} is {business_days} business days.")

    return business_days

def load_short_rate_data(file_path: str = 'data/DTB3.csv', start_date:str = '1980-01-01', end_date:str = date.today().strftime('%Y-%m-%d')) -> pd.DataFrame:
    """
    Returns the DTB3 short rate data (source: FRED) between a given start and end date.
    The rate is converted from percentage to decimal and renamed to 'r_t'.
    """
    short_rate = pd.read_csv(file_path, sep=',') 
    short_rate['observation_date'] = pd.to_datetime(short_rate['observation_date'], format='%Y-%m-%d')
    short_rate = short_rate.set_index('observation_date').sort_index()
    short_rate = short_rate.loc[start_date:end_date]
    short_rate['DTB3'] = short_rate['DTB3'] / 100  # Convert percentage to decimal
    short_rate = short_rate.rename(columns={'DTB3': 'r_t'})
    short_rate.index.name = 'date'
    short_rate.dropna(inplace=True)
    return short_rate


def _expiration_for_contract(contract_name: str) -> pd.Timestamp:
    """Third Wednesday of the delivery month for a given contract like 'H2026'."""
    month_code = contract_name[0]
    year = int(contract_name[1:])
    month = MONTH_MAP[month_code]

    first_day = pd.Timestamp(year, month, 1)
    all_wednesdays = pd.date_range(
        first_day,
        first_day + pd.offsets.MonthEnd(0),
        freq="W-WED"
    )
    return all_wednesdays[2].normalize()  # 3rd Wednesday


def load_calibration_data(
    commo_ticker: str,
    rate_file_path: str = "data/DTB3.csv",
    start_date="1980-01-01",
    end_date="2025-11-13",
):
    """
    Load commodity prices and short rate data for calibration.
    """
    start_loading_time = time.time()
    commodity_data = load_commodity_prices(
        f"data/{commo_ticker}.csv", start_date=start_date, end_date=end_date
    )
    end_loading_time = time.time()
    print(f"Loading commodity data took {end_loading_time - start_loading_time:.2f} seconds.")

    # ---- FAST time-to-maturity ----
    start_maturity_time = time.time()

    # 1) Compute expiration date once per unique contract
    contracts = commodity_data["contract"].astype(str).values
    unique_contracts = pd.unique(contracts)
    exp_map = {c: _expiration_for_contract(c) for c in unique_contracts}

    # 2) Convert index to datetime once
    current_dates = pd.to_datetime(commodity_data.index).normalize().to_numpy()

    # 3) Compute business days (still per-row, but without pandas apply overhead)
    commodity_data["time_to_maturity"] = [
        np.busday_count(pd.Timestamp(cd).date(), exp_map[c].date())
        for cd, c in zip(current_dates, contracts)
    ]

    end_maturity_time = time.time()
    print(f"Calculating time to maturity took {end_maturity_time - start_maturity_time:.2f} seconds.")
    # -------------------------------

    start_rates_time = time.time()
    short_rate_data = load_short_rate_data(rate_file_path, start_date=start_date, end_date=end_date)
    end_rates_time = time.time()
    print(f"Loading short rate data took {end_rates_time - start_rates_time:.2f} seconds.")

    # Merge datasets on date index
    calibration_data = commodity_data.join(short_rate_data, how="inner")
    calibration_data.sort_index(inplace=True)

    return calibration_data