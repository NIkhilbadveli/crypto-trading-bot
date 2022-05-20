import itertools
import os
import threading
import time
from datetime import timedelta

from binance import Client
from naive_bot import NaiveBot
import pandas as pd
from get_data import ThreadWithResult

# time_periods = [('2019-01-01', '2019-12-31'), ('2020-01-01', '2020-12-31'), ('2021-01-01', '2021-12-31')]
time_periods = [('2021-01-01', '2021-12-31'), ('2021-01-08', '2022-01-07'), ('2021-01-15', '2022-01-14'),
                ('2021-02-01', '2022-01-31')]
currs = ['BTC', 'XRP', 'ETH']  # Currencies to test
tps = [0.2, 0.5, 1]  # Take profits
mds = [1.2, 1.6]  # Max days
levs = [5, 10, 50]  # Leverages
mfs = [1, 2]  # Margin factors
stps = [0.05, 0.1]  # Stake percentages

groups = [time_periods, currs, tps, mds, levs, mfs, stps]

base = 'USDT'
interval = '1h'
columns = ['currency', 'date', 'leverage', 'margin_factor', 'take_profit', 'max_days', 'stake_perc', 'num_trades',
           'trades_per_day', 'short_perc', 'total_profit', 'profit_per_trade', 'profit_factor', 'no_of_margin_calls',
           'roi']
results = []


def do_the_test():
    """
    This function is used to run the backtest for a given configuration.
    :return:
    """
    print('Total number of possible configurations: ',
          len(currs) * len(tps) * len(mds) * len(levs) * len(mfs) * len(stps) * len(time_periods))
    # Generate all possible configurations using the groups
    for ns in itertools.product(*groups):
        sd, ed = ns[0][0], ns[0][1]
        curr, tp, md, lev, mf, stp = ns[1], ns[2], ns[3], ns[4], ns[5], ns[6]
        bot1 = NaiveBot()
        params = bot1.BackTestParams(take_profit=tp, short=False, max_days=md, starting_balance=500,
                                     stake_perc=stp, compound=True, enable_forecast=True, leverage_enabled=True,
                                     lev_mar=(lev, mf))
        # data_file = curr + '_' + base + '_' + '_data_' + interval + '.csv'
        # model_file = curr + '_' + base + '_' + '_model_' + interval + '.h5'
        # model_scaler_file = curr + '_' + base + '_' + '_scaler_' + interval + '.pkl'
        # for fle in [data_file, model_file, model_scaler_file]:
        #     if os.path.isfile(fle):
        #         os.remove(fle)
        print('Testing for {} from {} to {}'.format(curr, sd, ed))
        out = bot1.perform_backtest(currency=curr, base='USDT', start_date=sd, end_date=ed,
                                    interval=Client.KLINE_INTERVAL_1HOUR, params=params)
        results.append(out)
        df = pd.DataFrame(results, columns=columns)
        # print(df.to_string(index=True))
        df.to_csv('backtest_results.csv')


# Profit factor greater than 3
# Annual drawdown less than 3%
# Annual return greater than 500%
# Maximum daily low of -$1,000
# Avg Daily profit greater than $1,000
# Less than 5,000 trades annually
# Greater than 253 trades annually


s = 0
data_thread = ThreadWithResult(target=do_the_test, daemon=True)
data_thread.start()
while data_thread.is_alive():
    print('Doing the test...', str(timedelta(seconds=s)))
    s += 30
    time.sleep(30)
