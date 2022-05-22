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
time_periods = [('2021-04-21', '2022-04-21'), ('2021-04-22', '2022-04-22'), ('2021-04-23', '2022-04-23'),
                ('2021-04-24', '2022-04-24'), ('2021-04-25', '2022-04-25'), ('2021-04-26', '2022-04-26'),
                ('2021-04-27', '2022-04-27'), ('2021-04-28', '2022-04-28'), ('2021-04-29', '2022-04-29'),
                ('2021-04-30', '2022-04-30'), ('2021-05-01', '2022-05-01'), ('2021-05-02', '2022-05-02'),
                ('2021-05-03', '2022-05-03'), ('2021-05-04', '2022-05-04'), ('2021-05-05', '2022-05-05'),
                ('2021-05-06', '2022-05-06'), ('2021-05-07', '2022-05-07'), ('2021-05-08', '2022-05-08'),
                ('2021-05-09', '2022-05-09'), ('2021-05-10', '2022-05-10'), ('2021-05-11', '2022-05-11'),
                ('2021-05-12', '2022-05-12'), ('2021-05-13', '2022-05-13'), ('2021-05-14', '2022-05-14'),
                ('2021-05-15', '2022-05-15'), ('2021-05-16', '2022-05-16'), ('2021-05-17', '2022-05-17'),
                ('2021-05-18', '2022-05-18'), ('2021-05-19', '2022-05-19'), ('2021-05-20', '2022-05-20'), ]
currs = ['BTC', 'ETH', 'XRP']  # Currencies to test
sbs = [200, 500, 1000, 10000]  # Starting balance
mtas = [10]  # Minimum trade amounts
tps = [0.5]  # Take profits
mds = [1.6]  # Max days
levs = [50]  # Leverages
mfs = [1]  # Margin factors
stps = [0.05]  # Stake percentages

groups = [time_periods, currs, sbs, mtas, tps, mds, levs, mfs, stps]

base = 'USDT'
interval = '1h'
columns = ['currency', 'date', 'starting_balance', 'min_trade_amt', 'leverage', 'margin_factor', 'take_profit',
           'max_days', 'stake_perc', 'num_trades', 'trades_per_day', 'short_perc', 'total_profit', 'total_fee',
           'profit_per_trade', 'profit_factor', 'no_of_margin_calls', 'roi']
results = []


def do_the_test():
    """
    This function is used to run the backtest for a given configuration.
    :return:
    """
    print('Total number of possible configurations: ',
          len(currs) * len(sbs) * len(mtas) * len(tps) * len(mds) * len(levs) * len(mfs) * len(stps) * len(
              time_periods))
    # Generate all possible configurations using the groups
    for ns in itertools.product(*groups):
        sd, ed = ns[0][0], ns[0][1]
        curr, sb, mta, tp, md, lev, mf, stp = ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7], ns[8]
        bot1 = NaiveBot()
        params = bot1.BackTestParams(take_profit=tp, short=False, max_days=md, starting_balance=sb, min_trade_amt=mta,
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
        print(df.to_string(index=True))
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
