import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('../data/historical_data_1_min.csv')
df = data[['close_time', 'close']]
price_array = df.to_numpy()
price_array = MinMaxScaler().fit_transform(price_array)
prediction_period = 360


def simple_moving_average_normal(price_arr, period):
    sma = []
    for i in range(len(price_arr)):
        if i < period:
            sma.append(np.nan)
        else:
            sma.append(np.mean(price_arr[i - period:i, 1]))
    return sma


def simple_moving_average_different(price_arr, period):
    # This will hold all the SMA values for the prediction period
    sma_predictions = []
    actual_prices = []
    for i in range(period, len(price_arr), int(len(price_arr) / 10)):
        # individual SMA values for the prediction period
        sma = []
        input_period = list(price_arr[i - period:i, 1])  # This is like a queue, we append the new avg at the end
        for j in range(prediction_period):
            avg = np.mean(input_period)
            sma.append(avg)  # Saving the SMA value for this period
            input_period.append(avg)
            input_period.pop(0)

        actual_prices.append(price_arr[i - period:i + prediction_period, 1])
        x = price_arr[i - period:i, 1]
        x = np.append(x, sma)
        sma_predictions.append(x)

    return np.array(sma_predictions), np.array(actual_prices)


def plot_sma_preds(sma_preds, actual_prices):
    for i in range(len(sma_preds)):
        plt.plot(sma_preds[i], color='red')
        plt.plot(actual_prices[i], color='blue')
    plt.show()


def calculate_mse(arr1, arr2):
    return ((arr1 - arr2) ** 2).sum() / len(arr1)


a, b = simple_moving_average_different(price_array, 1440)
plot_sma_preds(a, b)
