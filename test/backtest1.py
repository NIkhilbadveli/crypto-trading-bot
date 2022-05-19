import os
import threading
import time
from datetime import timedelta

from binance import Client
from naive_bot import NaiveBot
import pandas as pd
from get_data import ThreadWithResult

time_periods = [('2019-01-01', '2019-12-31'), ('2020-01-01', '2020-12-31'), ('2021-01-01', '2021-12-31')]
currs = ['ETH', 'BTC', 'XRP']
base = 'USDT'
interval = '1h'
columns = ['currency', 'date', 'leverage', 'margin_factor', 'no_of_trades', 'no_of_margin_calls', 'roi']
results = []


def do_the_test():
    for curr in currs:
        for sd, ed in time_periods:
            bot1 = NaiveBot()
            params = bot1.BackTestParams(take_profit_percentage=0.2, short=False,
                                         selling_points=[(1, 0.4 * 5), (3, 1.0 * 5)],
                                         starting_balance=500,
                                         stake_perc=0.1,
                                         compound=True,
                                         enable_forecast=True,
                                         leverage_enabled=True)
            data_file = curr + '_' + base + '_' + '_data_' + interval + '.csv'
            model_file = curr + '_' + base + '_' + '_model_' + interval + '.h5'
            model_scaler_file = curr + '_' + base + '_' + '_scaler_' + interval + '.pkl'
            for fle in [data_file, model_file, model_scaler_file]:
                if os.path.isfile(fle):
                    os.remove(fle)
            print('Testing for {} from {} to {}'.format(curr, sd, ed))
            out = bot1.perform_backtest(currency=curr, base='USDT', start_date=sd, end_date=ed,
                                        interval=Client.KLINE_INTERVAL_1HOUR, params=params)
            results.append(out)
            df = pd.DataFrame(results, columns=columns)
            print(df.to_string(index=True))
            df.to_csv('backtest_results.csv')


s = 0
data_thread = ThreadWithResult(target=do_the_test, daemon=True)
data_thread.start()
while data_thread.is_alive():
    print('Doing the test...', str(timedelta(seconds=s)))
    s += 10
    time.sleep(10)
