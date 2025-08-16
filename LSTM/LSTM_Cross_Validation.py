
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

print('Loading...')
data = pd.read_csv('./data/sp500_features.csv')

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

target = 'Close'
features = data.drop(columns=[target, 'Date']).values

X = features
# X = data[features].values
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

time_steps = 60
X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)

print('Improving the Data...')


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

print(f"Original training data shape: {X_train_full.shape}")
print(f"Augmented training data shape: {X_train_full_augmented.shape}")

print('Cross Validating...')

tscv = TimeSeriesSplit(n_splits=5)

mse_scores = []
rmse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full_augmented)):
    print(f"Fold {fold + 1}")


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
                        callbacks=[early_stopping])

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label=f'Fold {fold + 1} - Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label=f'Fold {fold + 1} - Validation Loss', color='orange', linestyle='--',
             linewidth=2)
    plt.title(f'Fold {fold + 1} - Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/loss_fold_{fold + 1}.svg')

    val_predictions = model.predict(X_val)

    val_predicted_prices = scaler_y.inverse_transform(val_predictions)
    val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))

    val_mse = mean_squared_error(val_real_prices, val_predicted_prices)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(val_real_prices, val_predicted_prices)
    val_r2 = r2_score(val_real_prices, val_predicted_prices)

    mse_scores.append(val_mse)
    rmse_scores.append(val_rmse)
    mae_scores.append(val_mae)
    r2_scores.append(val_r2)

print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean R²: {np.mean(r2_scores)}")


print('Predicting...')

val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

val_predicted_prices = scaler_y.inverse_transform(val_predictions)
val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))

test_predicted_prices = scaler_y.inverse_transform(test_predictions)
test_real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

test_mse = mean_squared_error(test_real_prices, test_predicted_prices)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_real_prices, test_predicted_prices)
test_r2 = r2_score(test_real_prices, test_predicted_prices)

print(f"Test Set - Mean Squared Error (MSE): {test_mse}")
print(f"Test Set - Root Mean Squared Error (RMSE): {test_rmse}")
print(f"Test Set - Mean Absolute Error (MAE): {test_mae}")
print(f"Test Set - R² Score: {test_r2}")


plt.figure(figsize=(8, 4))

plt.plot(data['Date'][-len(test_real_prices):], test_real_prices, label="Real Prices", color='blue')

plt.plot(data['Date'][-len(test_predicted_prices):], test_predicted_prices, label="Predicted Prices", color='red',
         linestyle='--')

plt.title('Real vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/stock_prices_cv.svg')

predicted_df = pd.DataFrame(test_predicted_prices, columns=["Predicted Close"])
predicted_df.to_csv('./results/predicted_stock_prices_cv.csv', index=False)
