
import pandas as pd
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
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

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


def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def moving_average_smoothing(data, window_size=3):
    smoothed_data = np.empty_like(data)
    for col in range(data.shape[1]):
        smoothed_data[:, col] = np.convolve(data[:, col], np.ones(window_size) / window_size, mode='same')
    return smoothed_data

def data_augmentation(X, y, num_augmentations=5):
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):

        X_augmented.append(X[i])
        y_augmented.append(y[i])


        X_smooth = moving_average_smoothing(X[i])

        X_noisy = X_smooth + 0.01 * np.random.randn(*X_smooth.shape)

        X_shifted = np.roll(X_noisy, np.random.randint(-5, 5), axis=0)


        X_augmented.append(X_shifted)
        y_augmented.append(y[i])

    return np.array(X_augmented), np.array(y_augmented)



window_sizes = [10, 30, 60, 90, 120]
metrics = ['MSE', 'RMSE', 'MAE', 'R2']


results = {window_size: {metric: [] for metric in metrics} for window_size in window_sizes}

for window_size in window_sizes:
    print(f'Processing window size: {window_size}')


    X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, window_size)


    X_train_full, X_test, y_train_full, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)


    X_train_full_augmented, y_train_full_augmented = data_augmentation(X_train_full, y_train_full)


    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full_augmented)):
        print(f"Fold {fold + 1} for window size {window_size}")


        X_train, X_val = X_train_full_augmented[train_index], X_train_full_augmented[val_index]
        y_train, y_val = y_train_full_augmented[train_index], y_train_full_augmented[val_index]


        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                       kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))


        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])


        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[early_stopping], verbose=1)


        val_predictions = model.predict(X_val)
        val_predicted_prices = scaler_y.inverse_transform(val_predictions)
        val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))


        val_mse = mean_squared_error(val_real_prices, val_predicted_prices)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(val_real_prices, val_predicted_prices)
        val_r2 = r2_score(val_real_prices, val_predicted_prices)


        results[window_size]['MSE'].append(val_mse)
        results[window_size]['RMSE'].append(val_rmse)
        results[window_size]['MAE'].append(val_mae)
        results[window_size]['R2'].append(val_r2)

results_df = pd.DataFrame(results)
results_df.to_csv('./results/window_size_performance.csv', index=False)


averages = {window_size: {metric: np.mean(results[window_size][metric]) for metric in metrics} for window_size in window_sizes}


log_transformed_results = {window_size: {
    'MSE': np.log1p(averages[window_size]['MSE']),
    'RMSE': np.log1p(averages[window_size]['RMSE']),
    'MAE': averages[window_size]['MAE'],
    'R2': averages[window_size]['R2']
} for window_size in window_sizes}


barWidth = 0.15
r = np.arange(len(metrics))
plt.figure(figsize=(12, 6))

colors = sns.color_palette("Blues", len(window_sizes))

for idx, window_size in enumerate(window_sizes):
    avg_metrics = [log_transformed_results[window_size][metric] for metric in metrics]
    bars = plt.bar(r + idx * barWidth, avg_metrics, width=barWidth, color=colors[idx], label=f'Window Size {window_size}')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')


plt.xlabel('Metrics', fontweight='bold')
plt.ylabel('Log-Transformed / Original Value', fontweight='bold')


plt.xticks([r + barWidth * (len(window_sizes) / 2 - 0.5) for r in np.arange(len(metrics))], metrics)

plt.title('Log-Transformed Evaluation Metrics for Different Window Sizes')
plt.legend()
plt.tight_layout()
plt.savefig(f'./figures/window_size_performance.svg')
