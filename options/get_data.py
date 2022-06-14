import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import yfinance as yf
from dateutil import parser
from dateutil.tz import tz
from deriv_api import DerivAPI

api_key = 'zlDM7tnQhO1knwggZcoT4IvGxD2qVppkdh02dTJxMHgHgsXpn8mIBLoYO12KQkNB'
api_secret = 'TEb49wDTTYGk0KFEanI4DqShlFV9ZnFh9lLabHmvv7OHA8GSmHm5cdMBuYfn5rcC'
client = Client(api_key, api_secret)

deriv_api_key = 'O4WJddI6GWcNskz'
deriv_app_id = 31998

default_date = datetime(2019, 1, 1, tzinfo=tz.gettz('Europe/London'))


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def get_futures_data(cur, base, start_date, end_date, frequency):
    """
    Get historical data from Binance
    """

    # return get_historical_data(cur, base, start_date, end_date, frequency)

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


def get_data_forex(cur, base, start_date, end_date, frequency):
    """
    Get historical data from Binance
    """

    return yf.download(cur + base + '=X', start=start_date, end=end_date, interval=frequency)


async def get_deriv_data(symbol, start_date, end_date, frequency):
    """Use deriv API and download data then convert to df from json"""
    deriv_api = DerivAPI(app_id=deriv_app_id)
    authorize = await deriv_api.authorize(deriv_api_key)

    start_epoch = int(parser.parse(start_date, default=default_date).timestamp())
    end_epoch = int(parser.parse(end_date, default=default_date).timestamp())

    freq_to_secs = {
        '1h': 60 * 60,
        '1m': 60,
        '5m': 5 * 60,
        '15m': 15 * 60,
        '30m': 30 * 60,
        '1d': 1440 * 60
    }

    # def get_klines():
    #     """
    #     Get klines from Binance
    #     :return:
    #     """
    #     data_json = asyncio.run(deriv_api.ticks_history({
    #         "ticks_history": symbol,
    #         "adjust_start_time": 1,
    #         "end": end_epoch,
    #         "start": start_epoch,
    #         "style": "candles",
    #         "granularity": freq_to_secs[frequency]
    #     }))
    #     return data_json

    # s = 0
    # data_thread = ThreadWithResult(target=get_klines, daemon=True)
    # data_thread.start()
    # while data_thread.is_alive():
    #     print('Fetching the data using deriv api...', str(timedelta(seconds=s)))
    #     s += 30
    #     time.sleep(30)
    # print('\n')
    # klines = data_thread.result
    print('Fetching the data using deriv api...')
    all_klines = []
    while end_epoch > start_epoch:
        klines = await deriv_api.ticks_history({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "end": end_epoch,
            "start": start_epoch,
            "style": "candles",
            "granularity": freq_to_secs[frequency]
        })
        all_klines.extend(klines['candles'])
        end_epoch = klines['candles'][0]['epoch']
        # end_epoch = cur_epoch - freq_to_secs[frequency]

    # Get only the klines that are greater than or equal to start_epoch in case more are added.
    all_klines = list(filter(lambda x: x['epoch'] >= start_epoch, all_klines))
    # assert all_klines is not None, 'Klines is None'

    print('Fetching the data using deriv api...Done. Preparing dataframe...')

    # Todo: Why am I getting 5 more values when I try for a month data. I'll have to check the first and last value
    df = pd.DataFrame(data=all_klines, columns=['close', 'epoch', 'high', 'low', 'open'])
    df.drop_duplicates(inplace=True)
    return df


# data_df = asyncio.run(get_deriv_data('R_50', '2021-05-01', '2022-05-01', '1h'))
# print(data_df.head())
# print('Length', len(data_df))
