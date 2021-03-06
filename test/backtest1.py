import itertools
import time
from datetime import timedelta

from binance import Client
from naive_bot import NaiveBot
import pandas as pd
from get_data import ThreadWithResult

# time_periods = [('2019-01-01', '2019-12-31'), ('2020 - 2-01-01', '2020 - 2-12-31'), ('2021-01-01', '2021-12-31')]
time_periods = [('2022-01-01', '2022-05-31')]

currs = ['BTC']  # Currencies to test
sbs = [500]  # Starting balance
mtas = [10]  # Minimum trade amounts
tps = [0.5]  # Take profits
mds = [1.6]  # Max days
levs = [50]  # Leverages
mfs = [1]  # Margin factors
stps = [0.05]  # Stake percentages

intervals = ['1m']

groups = [time_periods, intervals, currs, sbs, mtas, tps, mds, levs, mfs, stps]

base = 'USDT'
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
        interval, curr, sb, mta, tp, md, lev, mf, stp = ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7], ns[8], ns[9]
        bot1 = NaiveBot()
        params = bot1.BackTestParams(take_profit=tp, short=False, max_days=md, starting_balance=sb, min_trade_amt=mta,
                                     stake_perc=stp, compound=True, enable_forecast=True, leverage_enabled=True,
                                     lev_mar=(lev, mf))
        print('Testing for {} from {} to {}'.format(curr, sd, ed))
        out = bot1.perform_backtest(currency=curr, base=base, start_date=sd, end_date=ed,
                                    interval=interval, params=params)
        results.append(out)
        df = pd.DataFrame(results, columns=columns)
        # print(df.to_string(index=True))
        df.to_csv('backtest_results.csv')


s = 0
data_thread = ThreadWithResult(target=do_the_test, daemon=True)
data_thread.start()
while data_thread.is_alive():
    print('Doing the test...', str(timedelta(seconds=s)))
    s += 30
    time.sleep(30)
