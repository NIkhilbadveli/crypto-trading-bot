from datetime import datetime
from binance.client import Client
import pandas as pd

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


def get_historical_data(cur, base, start_date, end_date, frequency):
    """
    Get historical data from Binance
    :param base:
    :param cur:
    :param start_date:
    :param end_date:
    :param frequency:
    :return:
    """
    # klines = client.get_historical_klines(cur + "USDT", frequency, start_date, end_date)
    klines = client.get_historical_klines(cur + base, frequency, start_date, end_date, limit=1000)
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


# data_df = get_hourly_data('ETH', 'USDT', 1650997348284, datetime.now().timestamp())
# data_df.to_csv('eth_data_1hr.csv')
