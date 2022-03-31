from binance import Client
from naive_bot import NaiveBot

bot1 = NaiveBot()
params = bot1.BackTestParams(take_profit_percentage=1, short=False, selling_points=[(2, 3), (5, 7)], compound=True)

bot1.dry_run(currency='ETH', interval=Client.KLINE_INTERVAL_1MINUTE, params=params)
