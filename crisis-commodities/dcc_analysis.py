import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

# Style settings
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Crisis Periods
CRISIS_PERIODS = {
    "GFC": ("2007-08-01", "2009-03-31"),
    "COVID-19": ("2020-02-01", "2021-12-31"),
    "Russia-Ukraine": ("2022-02-24", "2024-12-31"),}

def load_data_from_excel(ticker_name, folder="raw_data"):
    """
    Load data from Excel and CLEAN IT strictly.
    Fixes issues where header strings are read as data.
    """
    filename = ticker_name.replace(" ", "_").lower() + ".xlsx"
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Run the previous script first.")

    # 1. Read Excel without assuming header format (safest)
    # We read index_col=0 to get dates, but we'll clean them next
    df = pd.read_excel(path, index_col=0)

    # 2. Fix the Index (Remove non-date rows like 'Ticker' or 'Date')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()] # Drop rows where index is not a valid date

    # 3. Select the Price Column ('Adj Close' or 'Close')
    # We check if columns are MultiIndex (Tuples) or standard
    price_series = None





    # Check for 'Adj Close' or 'Close' in column names
    # Note: df.columns might still contain tuples if created by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten columns to string for easier search
        flat_cols = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        if 'Adj Close' in flat_cols:
            idx = flat_cols.index('Adj Close')
            price_series = df.iloc[:, idx]
        elif 'Close' in flat_cols:
            idx = flat_cols.index('Close')
            price_series = df.iloc[:, idx]
    else:
        # Standard columns
        if 'Adj Close' in df.columns:
            price_series = df['Adj Close']
        elif 'Close' in df.columns:
            price_series = df['Close']
        else:
            # Fallback: take the first column if it looks like data
            price_series = df.iloc[:, 0]

    if price_series is None:
        raise ValueError(f"Could not find price column in {filename}")

    # 4. Force Data to Numeric (Crucial Step for your Error)
    # This turns any remaining strings (like column headers) into NaN
    price_series = pd.to_numeric(price_series, errors='coerce')

    # 5. Drop NaNs
    price_series = price_series.dropna()

    return price_series

def fit_garch(returns):
    """Fits a GARCH(1,1) model and returns standardized residuals."""
    try:
        # CHANGE: dist='studentst' respects the "Fat Tail" hypothesis
        am = arch_model(returns * 100, vol='Garch', p=1, o=0, q=1, dist='studentst')
        res = am.fit(disp='off')
        return res.resid / res.conditional_volatility
    except Exception as e:
        print(f" GARCH fit warning: {e}. Using simple volatility.")
        return returns / returns.rolling(30).std().fillna(method='bfill')

def calculate_dcc_ewma(resid1, resid2, lambda_=0.94):
    """Calculates Dynamic Conditional Correlation using EWMA."""
    var1 = np.zeros_like(resid1)
    var2 = np.zeros_like(resid2)
    cov12 = np.zeros_like(resid1)

    var1[0] = np.var(resid1)
    var2[0] = np.var(resid2)
    cov12[0] = np.cov(resid1, resid2)[0, 1]

    for t in range(1, len(resid1)):
        var1[t] = (1 - lambda_) * resid1[t]**2 + lambda_ * var1[t-1]
        var2[t] = (1 - lambda_) * resid2[t]**2 + lambda_ * var2[t-1]
        cov12[t] = (1 - lambda_) * resid1[t] * resid2[t] + lambda_ * cov12[t-1]

    dcc = cov12 / (np.sqrt(var1) * np.sqrt(var2))
    return pd.Series(dcc, index=resid1.index)

def generate_dcc_analysis(asset1_name, asset2_name):
    print(f"\n--- Running DCC Analysis: {asset1_name} vs {asset2_name} ---")

    # 1. Load Data
    try:
        p1 = load_data_from_excel(asset1_name)
        p2 = load_data_from_excel(asset2_name)
    except Exception as e:
        print(f"   Data Load Error: {e}")
        return

    # Align Data
    df = pd.concat([p1, p2], axis=1).dropna()
    df.columns = [asset1_name, asset2_name]





    # Calculate Log Returns
    returns = np.log(df / df.shift(1)).dropna()

    if len(returns) < 100:
        print("   Not enough data points for GARCH analysis.")
        return

    # 2. Fit GARCH
    print(f" Fitting GARCH(1,1) to {asset1_name}...")
    resid1 = fit_garch(returns[asset1_name])

    print(f" Fitting GARCH(1,1) to {asset2_name}...")
    resid2 = fit_garch(returns[asset2_name])

    # 3. Calculate DCC
    print(" Calculating Dynamic Correlation...")
    dcc = calculate_dcc_ewma(resid1, resid2)

    # 4. Generate Figure
    plt.figure(figsize=(12, 6))
    plt.plot(dcc.index, dcc, label='DCC (GARCH-adjusted)', color='darkblue', linewidth=1.5)

    rolling = returns[asset1_name].rolling(252).corr(returns[asset2_name])
    plt.plot(rolling.index, rolling, label='Rolling Window (252d)', color='gray', linestyle='--', alpha=0.6)

    plt.axhline(0, color='black', linewidth=1)

    for name, (start, end) in CRISIS_PERIODS.items():
        plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='red', alpha=0.1)

    plt.title(f'Dynamic Conditional Correlation (DCC): {asset1_name} vs {asset2_name}',
fontsize=14)
    plt.ylabel('Correlation')
    plt.legend()
    plt.tight_layout()

    filename = f"figures/DCC_{asset1_name}_vs_{asset2_name}.png"
    plt.savefig(filename, dpi=300)
    print(f" ✓ Saved Figure: {filename}")

    # 5. Generate Summary Table
    stats = {
        'Period': ['Full Sample'],
        'Mean Corr': [round(dcc.mean(), 3)],
        'Min Corr': [round(dcc.min(), 3)],
        'Max Corr': [round(dcc.max(), 3)],
        'Vol (Std Dev)': [round(dcc.std(), 3)] }

    for name, (start, end) in CRISIS_PERIODS.items():
        mask = (dcc.index >= start) & (dcc.index <= end)
        sub = dcc[mask]
        if len(sub) > 0:
            stats['Period'].append(name)
            stats['Mean Corr'].append(round(sub.mean(), 3))
            stats['Min Corr'].append(round(sub.min(), 3))
            stats['Max Corr'].append(round(sub.max(), 3))
            stats['Vol (Std Dev)'].append(round(sub.std(), 3))

    df_stats = pd.DataFrame(stats)
    csv_name = f"tables/DCC_Stats_{asset1_name}_{asset2_name}.csv"
    df_stats.to_csv(csv_name, index=False)
    print(f" ✓ Saved Table: {csv_name}")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    # Define pairs to analyze
    pairs = [
        ("Gold", "S&P 500"),
        ("Silver", "S&P 500"),
        ("Crude Oil", "S&P 500"),
        ("Copper", "S&P 500")
    ]

    for a1, a2 in pairs: generate_dcc_analysis(a1, a2)
