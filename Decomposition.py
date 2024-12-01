from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def analyze_time_series_components(data):
    # Perform time series decomposition
    decomposition = seasonal_decompose(data, period=365)

    # Get trend, seasonality and residuals
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Calculate variance contribution of each component
    total_var = np.nanvar(data)  # Use np.nanvar to handle potential NaN values
    trend_var = np.nanvar(trend)
    seasonal_var = np.nanvar(seasonal)
    residual_var = np.nanvar(residual)

    # Calculate proportion of each component - ensure scalar values
    trend_prop = float(trend_var / total_var)
    seasonal_prop = float(seasonal_var / total_var)
    residual_prop = float(residual_var / total_var)

    print("Component Variance Proportions:")
    print(f"Trend: {trend_prop:.2%}")
    print(f"Seasonal: {seasonal_prop:.2%}")
    print(f"Residual: {residual_prop:.2%}")

    # Visualize decomposition results
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(data)
    plt.title('Original')
    plt.subplot(412)
    plt.plot(trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(seasonal)
    plt.title('Seasonal')
    plt.subplot(414)
    plt.plot(residual)
    plt.title('Residual')
    plt.tight_layout()
    plt.show()

    return trend_prop, seasonal_prop, residual_prop


def read_tsf_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = {}
    data_start = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#') or len(line) == 0:
            continue
        if line.startswith('@'):
            if line == '@data':
                continue
            try:
                parts = line[1:].split(maxsplit=1)
                if len(parts) >= 2:
                    key, value = parts
                    metadata[key] = value
            except Exception as e:
                print(f"Warning: Could not parse metadata line: {line}")
            continue
        if ':' in line:
            data_start = i
            break

    data_line = lines[data_start].strip()
    parts = data_line.split(':')
    values = parts[2].split(',')
    values = [float(x) for x in values if x.strip()]

    start_date = pd.to_datetime(parts[1])
    dates = pd.date_range(start=start_date, periods=len(values), freq='D')

    df = pd.DataFrame({
        'time': dates,
        'value': values
    })

    return df, metadata


file_path = "saugeenday_dataset.tsf"
df, metadata = read_tsf_file(file_path)
df.set_index('time', inplace=True)

trend_prop, seasonal_prop, residual_prop=analyze_time_series_components(df)