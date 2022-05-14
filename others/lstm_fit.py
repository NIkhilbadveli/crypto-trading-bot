# Import required libraries
from binance import Client
from keras.losses import mean_squared_error
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from get_data import get_historical_data
from datetime import datetime, timedelta
from pickle import load

# Initialize the variables
window_size = 100
prediction_window = 48

# Import model .pkl file and use it to predict the price for the next 48 time steps.
model = load_model('bitcoin_model.h5')
scaler = load(open('bitcoin_scaler.pkl', 'rb'))

# Get the historical data for the last 100 time steps
today = datetime.today() - timedelta(days=2)
end_date = today.strftime('%Y-%m-%d')
start_date = (today - timedelta(days=5)).strftime('%Y-%m-%d')
df = pd.read_csv('../data/historical_data_30_min.csv')
# df = get_historical_data('BTC', start_date, end_date, Client.KLINE_INTERVAL_30MINUTE)
# It looks like this method doesn't give us the today's price candles
time_array = df['close_time'].to_numpy()
price_array = df['close'].to_numpy()
price_array = scaler.fit_transform(price_array.reshape(-1, 1))

d = []
for index in range(len(price_array) - window_size):
    d.append(price_array[index: index + window_size])
d = np.array(d)
d = d.reshape((d.shape[0], d.shape[1], 1))

x_data = d[:, :-1, :]
y_data = d[:, -1, :]

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=False)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# print first and last timestamp values
print('First timestamp: ', time_array[0])
print('Last timestamp: ', time_array[-1])

# # We don't have X_test, so we will use the previous prediction and append it to X_test.
# predictions = []
# for i in range(prediction_window):
#     X_test = np.array(price_array[-window_size:])
#     X_test = X_test.reshape((1, window_size, 1))
#     prediction = model.predict(X_test)
#     predictions.append(prediction)
#     price_array = np.append(price_array, prediction)
#
# # Plot the predictions along with the previous price data.
# plt.plot(price_array)
# plt.show()

# todo why is it always going down to the bottom? Looks like this approach is not working.

y_pred = model.predict(X_test)
print(mean_squared_error(Y_test, y_pred))
