import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px



def plot_term_structure(futures_data: pd.DataFrame, date: pd.Timestamp = None, figsize = (10, 6)):
    if date is None:
        date = futures_data.index[-1]

    date_data = futures_data.loc[date]
    contracts = date_data['contract'].values
    prices = date_data['price'].values
    maturities = date_data['time_to_maturity'].values

    #Plot the term structure with matplotlib
    plt.figure(figsize=figsize)
    plt.plot(maturities, prices, marker='o')
    plt.title(f'Term Structure of Futures Prices on {date.date()}')
    plt.xlabel('Time to Maturity (Business Days)')
    plt.ylabel('Futures Price')
    plt.grid(True)
    plt.show()


def plot_term_structure_evolution(
    futures_data: pd.DataFrame,
    dates: list = None,
    figsize=(10, 6),
    use_plotly: bool = False,
    use_matplotlib: bool = True,
):
    """Plot term structures for selected dates on the same graph."""

    # Determine which dates to plot
    if dates is None:
        selected_dates = futures_data.index.unique()
    else:
        selected_dates = [pd.to_datetime(d) for d in dates]

    if use_plotly:
        fig = px.line(
            title="Term Structure Evolution",
            labels={'x': 'Time to Maturity (Business Days)', 'y': 'Futures Price'}
        )
        for d in selected_dates:
            date_data = futures_data.loc[d]
            date_data = date_data.sort_values("time_to_maturity")  # ensure points connect in maturity order
            prices = date_data['price'].values
            maturities = date_data['time_to_maturity'].values
            fig.add_scatter(x=maturities, y=prices, mode='lines+markers', name=str(pd.to_datetime(d).date()))
        fig.show()
    
    if use_matplotlib:
        plt.figure(figsize=figsize)
        for d in selected_dates:
            date_data = futures_data.loc[d]
            date_data = date_data.sort_values("time_to_maturity")  # order maturities before plotting
            prices = date_data['price'].values
            maturities = date_data['time_to_maturity'].values
            plt.plot(maturities, prices, marker='o', label=str(pd.to_datetime(d).date()))
        plt.title('Term Structure Evolution')
        plt.xlabel('Time to Maturity (Business Days)')
        plt.ylabel('Futures Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        


