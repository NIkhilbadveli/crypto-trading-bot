from binance import Client
from naive_bot import NaiveBot

bot1 = NaiveBot()
params = bot1.BackTestParams(take_profit_percentage=1, short=False, selling_points=[(2, 3), (5, 7)], compound=True)

bot1.perform_backtest(currency='BTC', base='USDT', start_date='2021-12-01', end_date='2021-12-31',
                      interval=Client.KLINE_INTERVAL_1MINUTE, params=params)
