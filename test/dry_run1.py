from naive_bot import NaiveBot

curr, sb, mta, tp, md, lev, mf, stp = 'BTC', 500, 10, 0.5, 1.6, 50, 1, 0.05
bot1 = NaiveBot()
params = bot1.BackTestParams(take_profit=tp, short=False, max_days=md, starting_balance=sb, min_trade_amt=mta,
                             stake_perc=stp, compound=True, enable_forecast=True, leverage_enabled=True,
                             lev_mar=(lev, mf))

bot1.dry_run(currency=curr, base='USDT', interval='1m', params=params)
