from binance import Client
from naive_bot import NaiveBot
import pandas as pd

year, nMonths = "2021", 12
monthStart = pd.date_range(year, periods=nMonths, freq='MS').strftime("%Y-%m-%d")
monthEnd = pd.date_range(year, periods=nMonths, freq='M').strftime("%Y-%m-%d")

# for sd, ed in zip(monthStart, monthEnd):
#     print(sd, ed)
#     bot1 = NaiveBot()
#     params = bot1.BackTestParams(take_profit_percentage=1, short=False, selling_points=[(1, 2), (3, 5)],
#                                  compound=True,
#                                  enable_forecast=False)
#
#     bot1.perform_backtest(currency='ETH', base='USDT', start_date=sd, end_date=ed,
#                           interval=Client.KLINE_INTERVAL_1MINUTE, params=params)

# time_periods = [('2020-04-01', '2021-03-31'), ('2021-04-01', '2022-03-31')]
# currs = ['BTC', 'ETH']

time_periods = [('2021-01-01', '2021-12-31')]
currs = ['ETH']

for curr in currs:
    for sd, ed in time_periods:
        print('Testing for {} from {} to {}'.format(curr, sd, ed))
        bot1 = NaiveBot()
        params = bot1.BackTestParams(take_profit_percentage=0.2, short=False,
                                     selling_points=[(1, 0.4), (3, 1.0)],
                                     compound=True,
                                     enable_forecast=True,
                                     leverage_enabled=True)

        bot1.perform_backtest(currency=curr, base='USDT', start_date=sd, end_date=ed,
                              interval=Client.KLINE_INTERVAL_1HOUR, params=params)
# bot1.dry_run(currency='ETH', base='USDT', interval=Client.KLINE_INTERVAL_1MINUTE, params=params)
