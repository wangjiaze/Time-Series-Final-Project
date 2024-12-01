import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


def create_features(data, n_lags=5, seasonal_period=365):
   """Create lag features and seasonal features"""
   # Ensure correct data format
   if isinstance(data, pd.Series):
       df = pd.DataFrame(data)
   else:
       df = pd.DataFrame(data, columns=['value'])

   # Basic lag features
   for i in range(1, n_lags + 1):
       df[f'lag_{i}'] = df['value'].shift(i)

   # Seasonal lag features
   df[f'seasonal_lag_1'] = df['value'].shift(seasonal_period)  # Same period last year
   df[f'seasonal_lag_2'] = df['value'].shift(seasonal_period * 2)  # Same period two years ago

   # Add time features
   #df['day_of_year'] = df.index.dayofyear
   #df['month'] = df.index.month

   # Create periodic features
   #df['yearly_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
   #df['yearly_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

   df = df.dropna()
   return df


def fit_svr_seasonal(data, train_ratio=0.8, n_lags=5):
   # Create features
   features_df = create_features(data, n_lags=n_lags)
   X = features_df.drop('value', axis=1)  # All feature columns
   y = features_df['value']  # Target variable

   # Rest of code remains unchanged
   train_size = int(len(X) * train_ratio)
   X_train, X_test = X[:train_size], X[train_size:]
   y_train, y_test = y[:train_size], y[train_size:]

   # Standardization
   scaler_X = StandardScaler()
   scaler_y = StandardScaler()

   X_train_scaled = scaler_X.fit_transform(X_train)
   X_test_scaled = scaler_X.transform(X_test)

   y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

   # Train SVR model
   print("Training SVR model...")
   print(f"Feature names: {X.columns.tolist()}")
   print(f"Number of features: {X.shape[1]}")
   #linear
   svr = SVR(kernel='rbf', C=100)
   svr.fit(X_train_scaled, y_train_scaled)

   # Prediction
   print("Making predictions...")
   y_pred_scaled = svr.predict(X_test_scaled)
   y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

   # Calculate evaluation metrics
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   r2 = r2_score(y_test, y_pred)

   # Visualization
   plt.figure(figsize=(15, 8))
   plt.plot(y_test.index, y_test.values, label='Actual')
   plt.plot(y_test.index, y_pred, label=f'Predicted (RMSE={rmse:.2f}, R2={r2:.2f})')
   plt.title('SVR Time Series Prediction without Seasonality')
   plt.legend()
   plt.grid(True)
   plt.show()

   return svr, rmse, r2, y_pred


def fit_svr_seasonal_cv(data, n_splits=5, n_lags=5):
   features_df = create_features(data, n_lags=n_lags)
   X = features_df.drop('value', axis=1)
   y = features_df['value']

   print("\nFeatures used:", X.columns.tolist())

   test_size = len(X) // (n_splits + 1)
   tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

   results_df = pd.DataFrame(columns=['Fold', 'RMSE', 'R2'])

   # Further reduce plot height
   fig, axes = plt.subplots(n_splits, 1, figsize=(20, 10))  # Reduce height to 10
   plt.subplots_adjust(hspace=0.2)  # Further reduce spacing

   for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
       X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
       y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

       scaler_X, scaler_y = StandardScaler(), StandardScaler()
       X_train_scaled = scaler_X.fit_transform(X_train)
       X_test_scaled = scaler_X.transform(X_test)
       y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

       svr = SVR(kernel='rbf', C=100)
       svr.fit(X_train_scaled, y_train_scaled)
       y_pred_scaled = svr.predict(X_test_scaled)
       y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

       rmse = np.sqrt(mean_squared_error(y_test, y_pred))
       r2 = r2_score(y_test, y_pred)

       results_df.loc[fold] = [f'Fold {fold + 1}', rmse, r2]

       ax = axes[fold]
       ax.plot(y_train.index, y_train.values,
               color='blue', alpha=0.6, linewidth=1,
               label='Training Data')
       ax.plot(y_test.index, y_test.values,
               color='green', alpha=0.6, linewidth=1,
               label='Test Data')
       ax.plot(y_test.index, y_pred,
               color='red', linestyle='--', linewidth=1.5,
               label='Predictions')

       ax.axvline(X_train.index[-1], color='gray', linestyle=':', alpha=0.5)
       ax.set_title(f'Fold {fold + 1}', pad=5, fontsize=10)  # Reduce padding
       ax.grid(True, alpha=0.3)
       ax.tick_params(axis='both', labelsize=8)

       if fold == 0:
           ax.legend(loc='upper right', fontsize=8)

   plt.tight_layout()
   plt.show()

   # Only add mean, without standard deviation
   results_df.loc[len(results_df)] = ['Mean',
                                      results_df['RMSE'].mean(),
                                      results_df['R2'].mean()]

   # Format values
   results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.2f}")
   results_df['R2'] = results_df['R2'].apply(lambda x: f"{x:.2f}")

   # Create table
   fig, ax = plt.subplots(figsize=(8, 3))  # Reduce table height
   ax.axis('tight')
   ax.axis('off')
   table = ax.table(cellText=results_df.values,
                    colLabels=results_df.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#f2f2f2'] * 3)

   table.auto_set_font_size(False)
   table.set_fontsize(9)
   table.scale(1.2, 1.5)

   plt.title('SVM CV-Result without seasonal features', pad=20)
   plt.tight_layout()
   plt.show()

   return results_df

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


#svr_model, rmse, r2, predictions = fit_svr_seasonal(df['value'])

results_df=fit_svr_seasonal_cv(df['value'])


'''
def fit_svr_seasonal_cv(data, n_splits=5, n_lags=5):
    features_df = create_features(data, n_lags=n_lags)
    X = features_df.drop('value', axis=1)
    y = features_df['value']

    print("\nFeatures used:", X.columns.tolist())

    test_size = len(X) // (n_splits + 1)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    rmse_scores = []
    r2_scores = []

    fig, axes = plt.subplots(n_splits, 1, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        svr = SVR(kernel='rbf', C=100)
        svr.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = svr.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        ax = axes[fold]
        ax.plot(y_train.index, y_train.values,
                color='blue', alpha=0.6, linewidth=1,
                label='Training Data')
        ax.plot(y_test.index, y_test.values,
                color='green', alpha=0.6, linewidth=1,
                label='Test Data')
        ax.plot(y_test.index, y_pred,
                color='red', linestyle='--', linewidth=1.5,
                label='Predictions')

        ax.axvline(X_train.index[-1], color='gray', linestyle=':', alpha=0.5)

        ax.set_title(f'Fold {fold + 1} (RMSE={rmse:.2f}, R2={r2:.2f})',
                     pad=20, fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Value', fontsize=8)
        # 调小坐标轴字体
        ax.tick_params(axis='x', labelsize=8, pad=10)
        ax.tick_params(axis='y', labelsize=8)

        if fold == 0:
            ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print('\nCross-validation scores:')
    print(f'Average RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})')
    print(f'Average R2: {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores):.2f})')

    return rmse_scores, r2_scores
'''
