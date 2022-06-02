import threading
import time
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import yfinance as yf

api_key = 'zlDM7tnQhO1knwggZcoT4IvGxD2qVppkdh02dTJxMHgHgsXpn8mIBLoYO12KQkNB'
api_secret = 'TEb49wDTTYGk0KFEanI4DqShlFV9ZnFh9lLabHmvv7OHA8GSmHm5cdMBuYfn5rcC'
client = Client(api_key, api_secret)

'''
[
    1499040000000,  # Open time
    "0.01634790",  # Open
    "0.80000000",  # High
    "0.01575800",  # Low
    "0.01577100",  # Close
    "148976.11427815",  # Volume
    1499644799999,  # Close time
    "2434.19055334",  # Quote asset volume
    308,  # Number of trades
    "1756.87402397",  # Taker buy base asset volume
    "28.46694368",  # Taker buy quote asset volume
    "17928899.62484339"  # Can be ignored
]'''


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def get_historical_data(cur, base, start_date, end_date, frequency):
    """
    Get historical data from Binance
    """

    # klines = client.get_historical_klines(cur + "USDT", frequency, start_date, end_date)

    def get_klines():
        """
        Get klines from Binance
        :return:
        """
        return client.get_historical_klines(cur + base, frequency, start_date, end_date, limit=1000)

    s = 0
    data_thread = ThreadWithResult(target=get_klines, daemon=True)
    data_thread.start()
    while data_thread.is_alive():
        print('Getting historical data from Binance...', str(timedelta(seconds=s)))
        s += 1
        time.sleep(1)
    print('\n')
    klines = data_thread.result
    assert klines is not None, 'Klines is None'
    print('Getting historical data from Binance... done. Preparing dataframe...')
    df = pd.DataFrame(klines,
                      columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'c1', 'c2', 'c3',
                               'c4',
                               'c5'])
    df.loc[:, ~df.columns.isin(['open_time', 'close_time', 'c2'])] = \
        df.loc[:, ~df.columns.isin(['open_time', 'close_time', 'c2'])].astype(float)
    return df


def get_futures_data(cur, base, start_date, end_date, frequency):
    """
    Get historical data from Binance
    """

    # klines = client.get_historical_klines(cur + "USDT", frequency, start_date, end_date)

    def get_klines():
        """
        Get klines from Binance
        :return:
        """
        return client.futures_historical_klines(cur + base, frequency, start_date, end_date, limit=1000)

    s = 0
    data_thread = ThreadWithResult(target=get_klines, daemon=True)
    data_thread.start()
    while data_thread.is_alive():
        print('Getting futures data from Binance...', str(timedelta(seconds=s)))
        s += 1
        time.sleep(1)
    print('\n')
    klines = data_thread.result
    assert klines is not None, 'Klines is None'
    print('Getting futures data from Binance... done. Preparing dataframe...')
    df = pd.DataFrame(klines,
                      columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'c1', 'c2', 'c3',
                               'c4',
                               'c5'])
    df.loc[:, ~df.columns.isin(['open_time', 'close_time', 'c2'])] = \
        df.loc[:, ~df.columns.isin(['open_time', 'close_time', 'c2'])].astype(float)
    return df


def get_hourly_data(cur, base, start_timestamp, end_timestamp):
    """
    Get data in hourly intervals from Binance starting from a given timestamp
    :param end_timestamp:
    :param cur:
    :param base:
    :param start_timestamp:
    :return:
    """
    # print('Getting hourly data from Binance...')
    klines = client.get_historical_klines(cur + base, Client.KLINE_INTERVAL_1HOUR, start_str=start_timestamp,
                                          end_str=str(end_timestamp), limit=1000)
    df = pd.DataFrame(klines,
                      columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'c1', 'c2', 'c3',
                               'c4',
                               'c5'])
    df.loc[:, ~df.columns.isin(['open_time', 'close_time', 'c2'])] = \
        df.loc[:, ~df.columns.isin(['open_time', 'close_time', 'c2'])].astype(float)
    return df


def get_data_forex(cur, base, start_date, end_date, frequency):
    """
    Get historical data from Binance
    :param base:
    :param cur:
    :param start_date:
    :param end_date:
    :param frequency:
    :return:
    """

    return yf.download(cur + base + '=X', start=start_date, end=end_date, interval=frequency)

# data_df = get_historical_data('ETH', 'USDT', '2020-05-01', '2021-04-30', Client.KLINE_INTERVAL_1HOUR)
# data_df.to_csv('eth_train_1h.csv')
