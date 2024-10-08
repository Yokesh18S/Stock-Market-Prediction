# Stock-Market-Prediction
import tensorflow as tf
import pandas as pd
import numpy as np
model = tf.keras.models.load_model(r"/content/stock_prediction_model.h5")
df = pd.read_csv("/content/MSFT.csv")
def prepare_data(data, look_back=60):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
    return np.array(X)
close_prices = df['Close'].values
close_prices = close_prices.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)
X = prepare_data(close_prices_scaled)
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

print("Predictions shape:", predictions.shape)
print("First few predictions:", predictions[:5])
df['Predicted_Close'] = np.nan
df.loc[60:, 'Predicted_Close'] = predictions
df.to_csv("MSFT1.csv", index=False)
print("Predictions saved to MSFT1.csv")
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
df = pd.read_csv('MSFT1.csv')
df['Date'] = pd.to_datetime(df['Date'])
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Actual Close', color='blue')
plt.plot(df['Date'], df['Predicted_Close'], label='Predicted Close', color='red')
plt.title('MSFT Stock Price - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
plt.savefig('MSFT1.png')
print("Plot saved as 'MSFT1.png'")
import pandas as pd
import numpy as np
df = pd.read_csv('MSFT1.csv')
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
df_clean = df.dropna(subset=['Close', 'Predicted_Close'])
mape_value = mape(df_clean['Close'], df_clean['Predicted_Close'])
accuracy = 100 - mape_value

print(f"Mean Absolute Percentage Error (MAPE): {mape_value:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(df_clean['Close'], df_clean['Predicted_Close'])
rmse = np.sqrt(mse)
r2 = r2_score(df_clean['Close'], df_clean['Predicted_Close'])

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(df_clean['Close'], df_clean['Predicted_Close'], alpha=0.5)
plt.plot([df_clean['Close'].min(), df_clean['Close'].max()],
         [df_clean['Close'].min(), df_clean['Close'].max()],
         'r--', lw=2)
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted_scatter.png')
print("Scatter plot saved as 'actual_vs_predicted_scatter.png'")
plt.show()
