
import pandas as pd,os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print('Loading...')
data = pd.read_csv('./data/sp500_features.csv')

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

target = 'Close'
features = data.drop(columns=[target, 'Date']).values
X = features
y = data[target].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


def create_sequences(X, y, time_steps=90):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 90
X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)


def moving_average_smoothing(data, window_size=3):
    smoothed_data = np.empty_like(data)
    for col in range(data.shape[1]):
        smoothed_data[:, col] = np.convolve(data[:, col], np.ones(window_size) / window_size, mode='same')
    return smoothed_data


def random_noise(data, noise_factor=0.01):
    noise = noise_factor * np.random.randn(*data.shape)
    return data + noise


def time_series_shift(data, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(data, shift, axis=0)


def data_augmentation(X, y, num_augmentations=5):
    X_augmented = []
    y_augmented = []
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        for _ in range(num_augmentations):
            X_smooth = moving_average_smoothing(X[i])
            X_noisy = random_noise(X_smooth)
            X_shifted = time_series_shift(X_noisy)
            X_augmented.append(X_shifted)
            y_augmented.append(y[i])
    return np.array(X_augmented), np.array(y_augmented)


X_train_full, X_test, y_train_full, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
X_train_full_augmented, y_train_full_augmented = data_augmentation(X_train_full, y_train_full)


def build_model(lstm_layers=1, units=50, dense_layers=1):
    model = Sequential()
    model.add(LSTM(units, return_sequences=(lstm_layers > 1),
                   input_shape=(X_train_full_augmented.shape[1], X_train_full_augmented.shape[2]),
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    for i in range(1, lstm_layers):
        model.add(LSTM(units, return_sequences=(i < lstm_layers - 1), kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
    for _ in range(dense_layers):
        model.add(Dense(units=units, activation='relu'))
    model.add(Dense(1))
    return model
#-------------------------------------------------------------------------------------------------------------------------------
#Customized Cross Validation Settings
#-------------------------------------------------------------------------------------------------------------------------------

# lstm_layer_options = [1, 2, 3]
# unit_options = [50, 100, 200]
# dense_layer_options = [1, 2, 3]
# results = {}
# validation_results = []
# best_model = None
# best_val_mse = float('inf')
# tscv = TimeSeriesSplit(n_splits=5)
#
# for lstm_layers in lstm_layer_options:
#     for units in unit_options:
#         for dense_layers in dense_layer_options:
#             model_name = f"LSTM-{lstm_layers}_Units-{units}_Dense-{dense_layers}"
#             results[model_name] = []
#             print(f"Training {model_name}")
#             val_mse_list, val_rmse_list, val_mae_list, val_r2_list = [], [], [], []
#
#             for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full_augmented)):
#                 X_train, X_val = X_train_full_augmented[train_index], X_train_full_augmented[val_index]
#                 y_train, y_val = y_train_full_augmented[train_index], y_train_full_augmented[val_index]
#
#                 model = build_model(lstm_layers=lstm_layers, units=units, dense_layers=dense_layers)
#                 model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
#                 early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#                 model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
#                           callbacks=[early_stopping], verbose=0)
#
#                 val_predictions = model.predict(X_val)
#                 val_predicted_prices = scaler_y.inverse_transform(val_predictions)
#                 val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))
#
#                 val_mse = mean_squared_error(val_real_prices, val_predicted_prices)
#                 val_rmse = np.sqrt(val_mse)
#                 val_mae = mean_absolute_error(val_real_prices, val_predicted_prices)
#                 val_r2 = r2_score(val_real_prices, val_predicted_prices)
#
#                 val_mse_list.append(val_mse)
#                 val_rmse_list.append(val_rmse)
#                 val_mae_list.append(val_mae)
#                 val_r2_list.append(val_r2)
#
#             avg_val_mse = np.mean(val_mse_list)
#             avg_val_rmse = np.mean(val_rmse_list)
#             avg_val_mae = np.mean(val_mae_list)
#             avg_val_r2 = np.mean(val_r2_list)
#
#             validation_results.append({
#                 'Model': model_name,
#                 'MSE': avg_val_mse,
#                 'RMSE': avg_val_rmse,
#                 'MAE': avg_val_mae,
#                 'R²': avg_val_r2
#             })
#
#             if avg_val_mse < best_val_mse:
#                 best_val_mse = avg_val_mse
#                 best_model = model_name
#
# print(f"Best model based on validation set: {best_model}")
#
# df_val_results = pd.DataFrame(validation_results)
# df_val_results.to_csv(f'./results/models_validation_results.csv', index=False)
#
#
# models = [result['Model'] for result in validation_results]
# mse_values = [result['MSE'] for result in validation_results]
# rmse_values = [result['RMSE'] for result in validation_results]
# mae_values = [result['MAE'] for result in validation_results]
# r2_values = [result['R²'] for result in validation_results]
#
# fig, ax = plt.subplots(figsize=(14, 8))
# bar_width = 0.2
# index = np.arange(len(models))
#
# plt.bar(index, mse_values, bar_width, label='MSE', color='blue')
# plt.bar(index + bar_width, rmse_values, bar_width, label='RMSE', color='green')
# plt.bar(index + 2 * bar_width, mae_values, bar_width, label='MAE', color='orange')
# plt.bar(index + 3 * bar_width, r2_values, bar_width, label='R²', color='red')
#
# plt.yscale('log')
#
# plt.xlabel('Model Complexity', fontsize=12)
# plt.ylabel('Metric Values (Log Scale)', fontsize=12)
# plt.title('Validation Set Evaluation Metrics for Different LSTM Model Complexities', fontsize=14)
#
# plt.xticks(index + bar_width, models, rotation=45, ha='right')
#
# plt.legend()
# plt.tight_layout()
# plt.savefig('./figures/validation_metric_comparison.svg')

# -------------------------------------------------------------------------------------------------------------------
# Final training for the selected best model (no CV) + test evaluation
# -------------------------------------------------------------------------------------------------------------------
best_model_name = 'LSTM-1_Units-200_Dense-1'
lstm_layers, units, dense_layers = map(
    int,
    best_model_name.replace('LSTM-', '')
                   .replace('_Units-', ',')
                   .replace('_Dense-', ',')
                   .split(',')
)
print(f"[Final] Train {best_model_name} without CV")

val_ratio = 0.10
val_size  = max(1, int(len(X_train_full) * val_ratio))
X_train_core, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
y_train_core, y_val = y_train_full[:-val_size], y_train_full[-val_size:]

X_train_aug, y_train_aug = data_augmentation(X_train_core, y_train_core)

model = build_model(lstm_layers=lstm_layers, units=units, dense_layers=dense_layers)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32, verbose=1, callbacks=[es]
)

print("Predicting on test set...")
test_pred = model.predict(X_test)
test_pred_p = scaler_y.inverse_transform(test_pred)
test_real_p = scaler_y.inverse_transform(y_test.reshape(-1, 1))

test_mse  = mean_squared_error(test_real_p, test_pred_p)
test_rmse = float(np.sqrt(test_mse))
test_mae  = mean_absolute_error(test_real_p, test_pred_p)
test_r2   = r2_score(test_real_p, test_pred_p)

print(f"Test Set - MSE: {test_mse:.6f}, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}")

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'{best_model_name} - Train/Val Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
plt.savefig(f'./figures/{best_model_name}_train_val_loss.svg'); plt.close()

pd.DataFrame({
    'Real Prices': test_real_p.flatten(),
    'Predicted Prices': test_pred_p.flatten()
}).to_csv(f'./results/{best_model_name}_test_results.csv', index=False)

plt.figure(figsize=(8, 4))
plt.plot(data['Date'][-len(test_real_p):], test_real_p, label="Real Prices")
plt.plot(data['Date'][-len(test_pred_p):], test_pred_p, label="Predicted Prices", linestyle='--')
plt.title(f'{best_model_name}: Real vs Predicted (Test)')
plt.xlabel('Date'); plt.ylabel('Stock Price'); plt.legend(); plt.tight_layout()
plt.savefig(f'./figures/{best_model_name}_real_vs_predicted.svg'); plt.close()

# ====================== Test-only evaluation & plots ======================
print("Predicting on TEST segment only...")
test_pred = model.predict(X_test)
test_pred_p = scaler_y.inverse_transform(test_pred)
test_real_p = scaler_y.inverse_transform(y_test.reshape(-1, 1))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
test_mse  = mean_squared_error(test_real_p, test_pred_p)
test_rmse = float(np.sqrt(test_mse))
test_mae  = mean_absolute_error(test_real_p, test_pred_p)
test_r2   = r2_score(test_real_p, test_pred_p)

print("\n=== TEST-ONLY METRICS (last 20%) ===")
print(f"Test MSE : {test_mse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Test MAE : {test_mae:.6f}")
print(f"Test R²  : {test_r2:.6f}")

os.makedirs('./results', exist_ok=True)
pd.DataFrame([{
    'Test_MSE': test_mse,
    'Test_RMSE': test_rmse,
    'Test_MAE': test_mae,
    'Test_R2': test_r2
}]).to_csv(f'./results/{best_model_name}_test_metrics.csv', index=False)

dates_test = data['Date'].iloc[-len(test_real_p):].reset_index(drop=True)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(dates_test, test_real_p, label='Real (Test)')
plt.plot(dates_test, test_pred_p, label='Predicted (Test)', linestyle='--')
plt.title(f'{best_model_name}: Real vs Predicted on Test Segment')
plt.xlabel('Date'); plt.ylabel('Target'); plt.legend(); plt.tight_layout()
os.makedirs('./figures', exist_ok=True)
plt.savefig(f'./figures/{best_model_name}_test_real_vs_predicted.svg')
plt.close()

plt.figure(figsize=(4.5, 4.5))
plt.scatter(test_real_p, test_pred_p, s=10, alpha=0.7)
mn = min(test_real_p.min(), test_pred_p.min())
mx = max(test_real_p.max(), test_pred_p.max())
plt.plot([mn, mx], [mn, mx], linewidth=1)
plt.title(f'{best_model_name}: Test Scatter (R²={test_r2:.3f})')
plt.xlabel('Real (Test)'); plt.ylabel('Predicted (Test)')
plt.tight_layout()
plt.savefig(f'./figures/{best_model_name}_test_scatter.svg')
plt.close()

print("Done!")
