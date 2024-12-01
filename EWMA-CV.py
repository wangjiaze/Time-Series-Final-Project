import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

def seasonal_ewma_with_cv(data, n_splits=5, seasonal_periods=365):
    total_size = len(data)
    test_size = total_size // (n_splits + 2)

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    results = {
        'additive': {'rmse': [], 'r2': [], 'forecasts': []},
        'multiplicative': {'rmse': [], 'r2': [], 'forecasts': []},
        'additive_no_trend': {'rmse': [], 'r2': [], 'forecasts': []}
    }

    fig, axes = plt.subplots(n_splits, 1, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        if len(train) < 2 * seasonal_periods:
            continue

        ax = axes[fold]
        ax.plot(train.index, train, color='blue', alpha=0.6, linewidth=1,
                label='Training Data')
        ax.plot(test.index, test, color='green', alpha=0.6, linewidth=1,
                label='Test Data')

        models = {
            'additive': ExponentialSmoothing(train, trend='add', seasonal='add',
                                             seasonal_periods=seasonal_periods),
            'multiplicative': ExponentialSmoothing(train, trend='add', seasonal='mul',
                                                   seasonal_periods=seasonal_periods),
            'additive_no_trend': ExponentialSmoothing(train, trend=None, seasonal='add',
                                                      seasonal_periods=seasonal_periods)
        }

        colors = {'additive': 'red', 'multiplicative': 'purple', 'additive_no_trend': 'orange'}

        for name, model in models.items():
            try:
                fit = model.fit()
                forecast = fit.forecast(len(test))
                rmse = np.sqrt(mean_squared_error(test, forecast))
                r2 = r2_score(test, forecast)

                results[name]['rmse'].append(rmse)
                results[name]['r2'].append(r2)
                results[name]['forecasts'].append(forecast)

                ax.plot(test.index, forecast, color=colors[name], linestyle='--',
                        label=f'{name} (RMSE={rmse:.2f}, R2={r2:.2f})')

            except:
                print(f"Error fitting {name} model in fold {fold}")
                continue

        ax.set_title(f'Fold {fold + 1}', pad=20, fontsize=9)  # Reduce title font size
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)  # Reduce axis tick font size
        ax.legend(loc='upper left', fontsize=7)  # Reduce legend font size

    plt.tight_layout()
    plt.show()

    avg_results = {}
    for name in results:
        if results[name]['rmse']:
            avg_results[name] = {
                'avg_rmse': np.mean(results[name]['rmse']),
                'avg_r2': np.mean(results[name]['r2']),
                'std_rmse': np.std(results[name]['rmse']),
                'std_r2': np.std(results[name]['r2'])
            }

    return avg_results

def find_best_ewma_model(data, n_splits=5, seasonal_periods=365):
    """
    1. Use CV to find the best model type
    """
    cv_results = seasonal_ewma_with_cv(data, n_splits, seasonal_periods)

    # Select best model type based on average RMSE
    best_model_type = min(cv_results.items(),
                          key=lambda x: x[1]['avg_rmse'])[0]

    print(f"\nBest model type: {best_model_type}")

    """
    2. Parameter tuning for the best model type
    """
    train_size = int(len(data) * 0.8)  # Reserve 20% as final test set
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Parameter grid
    alphas = np.linspace(0.01, 0.99, 20)
    best_alpha = None
    best_rmse = float('inf')
    best_model = None

    # Store alpha tuning results
    tuning_results = []

    for alpha in alphas:
        if best_model_type == 'multiplicative':
            model = ExponentialSmoothing(train_data,
                                         trend='add',
                                         seasonal='mul',
                                         seasonal_periods=seasonal_periods)
        elif best_model_type == 'additive':
            model = ExponentialSmoothing(train_data,
                                         trend='add',
                                         seasonal='add',
                                         seasonal_periods=seasonal_periods)
        else:  # no trend
            model = ExponentialSmoothing(train_data,
                                         trend=None,
                                         seasonal='add',
                                         seasonal_periods=seasonal_periods)

        model_fit = model.fit(smoothing_level=alpha)
        forecast = model_fit.forecast(len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        r2 = r2_score(test_data, forecast)

        tuning_results.append({
            'alpha': alpha,
            'rmse': rmse,
            'r2': r2
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_model = model_fit

    """
    3. Visualize parameter tuning results 
    """
    results_df = pd.DataFrame(tuning_results)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['alpha'], results_df['rmse'], 'o-')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE')
    plt.title(f'Parameter Tuning for {best_model_type} Model')
    plt.grid(True)
    plt.show()

    """
    4. Train final model with best parameters and forecast
    """
    final_model = ExponentialSmoothing(data,  # Use all data to train final model
                                       trend='add' if best_model_type != 'additive_no_trend' else None,
                                       seasonal='mul' if best_model_type == 'multiplicative' else 'add',
                                       seasonal_periods=seasonal_periods)
    final_model_fit = final_model.fit(smoothing_level=best_alpha)

    # Forecast future periods
    future_steps = len(test_data)
    forecast = final_model_fit.forecast(future_steps)
    r2 = r2_score(test_data, forecast)
    print(f"\nBest parameters:")
    print(f"Model type: {best_model_type}")
    print(f"Alpha: {best_alpha:.3f}")
    print(f"Final RMSE: {best_rmse:.2f}")
    print(f"Final R2: {r2:.2f}")

    return {
        'model_type': best_model_type,
        'alpha': best_alpha,
        'model': final_model_fit,
        'forecast': forecast,
        'tuning_results': results_df
    }

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

# Usage example
file_path = "saugeenday_dataset.tsf"

df, metadata = read_tsf_file(file_path)
df.set_index('time', inplace=True)

# Use the model
best_model = find_best_ewma_model(df['value'])

# Split last 20% of data for validation
train_size = int(len(df) * 0.8)
train_data = df['value'][:train_size]
test_data = df['value'][train_size:]

# Use best model to forecast test set
final_model = ExponentialSmoothing(train_data,
                                 trend='add' if best_model['model_type'] != 'additive_no_trend' else None,
                                 seasonal='mul' if best_model['model_type'] == 'multiplicative' else 'add',
                                 seasonal_periods=365)
final_model_fit = final_model.fit(smoothing_level=best_model['alpha'])
forecast = final_model_fit.forecast(len(test_data))

# Calculate forecast performance
rmse = np.sqrt(mean_squared_error(test_data, forecast))
r2 = r2_score(test_data, forecast)

# Visualization
plt.figure(figsize=(15, 6))
plt.plot(train_data.index, train_data, label='Training Data', alpha=0.5, color='blue')
plt.plot(test_data.index, test_data, label='Test Data', alpha=0.5, color='green')
plt.plot(test_data.index, forecast, color='red', linestyle='--',
         label=f'Forecast (RMSE={rmse:.2f}, R2={r2:.2f})')

plt.title(f'Best Model ({best_model["model_type"]}) Forecast Validation')
plt.legend(loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=8)
plt.tight_layout()
plt.show()

print(f"\nBest Model Details:")
print(f"Model Type: {best_model['model_type']}")
print(f"Alpha: {best_model['alpha']:.3f}")
print(f"RMSE on test set: {rmse:.2f}")
print(f"R2 on test set: {r2:.2f}")

'''
def seasonal_ewma_with_cv(data, n_splits=5, seasonal_periods=365):
    total_size = len(data)
    test_size = total_size // (n_splits + 2)

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    results = {
        'additive': {'rmse': [], 'r2': [], 'forecasts': []},
        'multiplicative': {'rmse': [], 'r2': [], 'forecasts': []},
        'additive_no_trend': {'rmse': [], 'r2': [], 'forecasts': []}
    }

    fig, axes = plt.subplots(n_splits, 1, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        if len(train) < 2 * seasonal_periods:
            continue

        ax = axes[fold]
        ax.plot(train.index, train, color='blue', alpha=0.6, linewidth=1, label='Training Data')
        ax.plot(test.index, test, color='green', alpha=0.6, linewidth=1, label='Test Data')

        models = {
            'additive': ExponentialSmoothing(train, trend='add', seasonal='add',
                                             seasonal_periods=seasonal_periods),
            'multiplicative': ExponentialSmoothing(train, trend='add', seasonal='mul',
                                                   seasonal_periods=seasonal_periods),
            'additive_no_trend': ExponentialSmoothing(train, trend=None, seasonal='add',
                                                      seasonal_periods=seasonal_periods)
        }

        colors = {'additive': 'red', 'multiplicative': 'purple', 'additive_no_trend': 'orange'}

        for name, model in models.items():
            try:
                fit = model.fit()
                forecast = fit.forecast(len(test))
                rmse = np.sqrt(mean_squared_error(test, forecast))
                r2 = r2_score(test, forecast)

                results[name]['rmse'].append(rmse)
                results[name]['r2'].append(r2)
                results[name]['forecasts'].append(forecast)

                ax.plot(test.index, forecast, color=colors[name], linestyle='--',
                        label=f'{name} (RMSE={rmse:.2f}, R2={r2:.2f})')

            except:
                print(f"Error fitting {name} model in fold {fold}")
                continue

        ax.set_title(f'Fold {fold + 1}', pad=20, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()

    # 计算和返回平均指标
    avg_results = {}
    for name in results:
        if results[name]['rmse']:
            avg_results[name] = {
                'avg_rmse': np.mean(results[name]['rmse']),
                'avg_r2': np.mean(results[name]['r2']),
                'std_rmse': np.std(results[name]['rmse']),
                'std_r2': np.std(results[name]['r2'])
            }

    return avg_results
'''