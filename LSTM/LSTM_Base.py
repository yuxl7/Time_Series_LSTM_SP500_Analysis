
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
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

def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)

print('Separating training and testing...')
X_train, X_temp, y_train, y_temp = train_test_split(X_lstm, y_lstm, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)


print('Building LSTM model...')
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

print('Training...')
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

print('Predicting...')
predictions = model.predict(X_test)

predicted_prices = scaler_y.inverse_transform(predictions)
real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_prices, predicted_prices)
r2 = r2_score(real_prices, predicted_prices)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

plt.figure(figsize=(8, 4))

plt.plot(data['Date'][-len(real_prices):], real_prices, label="Real Prices", color='blue')

plt.plot(data['Date'][-len(predicted_prices):], predicted_prices, label="Predicted Prices", color='red', linestyle='--')

plt.title('Real vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/stock_prices.svg')

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/loss.svg')

predicted_df = pd.DataFrame(predicted_prices, columns=["Predicted Close"])
predicted_df.to_csv('./results/predicted_stock_prices.csv', index=False)
