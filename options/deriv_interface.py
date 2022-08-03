import asyncio
import json
import time

import numpy as np
from deriv_api import DerivAPI
import pandas as pd

app_id = '32172'
api_token = 'zQjGuhLj4w1f8Sv'

symbols = ['R_10', 'R_25', 'R_50', 'R_75', 'R_100']


async def open_proposal(api, symbol, barrier, duration=24, amount=100):
    """Opens a proposal and returns the proposal details"""
    proposal = await api.proposal({"proposal": 1, "amount": amount, "barrier": str(barrier), "basis": "stake",
                                   "contract_type": 'ONETOUCH', "currency": "USD", "duration": duration,
                                   "duration_unit": "h",
                                   "symbol": symbol
                                   })
    # print(json.dumps(proposal, indent=2))
    return proposal


async def buy_contract(api, proposal_id, max_price):
    """Buys a contract based on the proposal id and the max price"""
    buy = await api.buy({"buy": proposal_id, "price": max_price})
    # print(json.dumps(buy, indent=2))
    return buy


async def sell_contract(api, contract_id, min_price=0):
    """Sells a contract based on the contract id and the min price"""
    # If min_price = 0, then it will sell at the current market price
    sell = await api.sell({"sell": contract_id, "price": min_price})
    # print(json.dumps(sell, indent=2))
    return sell


async def get_latest_price(api, symbol):
    """Retrieves the latest price of a symbol from the Deriv API"""
    msg = await api.ticks_history({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 10,
        "end": "latest",
        "start": 1,
        "style": "ticks"
    })

    return msg['history']['prices'][-1]


async def fetch_payout_data():
    """Retrieves the payout data from the Deriv API by opening a proposal and getting the payout"""
    api = DerivAPI(app_id=app_id)
    authorize = await api.authorize(api_token)
    # print(authorize)

    account = await api.balance()
    # print(account)

    active_symbols = await api.active_symbols({"active_symbols": "full"})
    # print(json.dumps(active_symbols, indent=2))

    results = []
    for sym in symbols:
        for bp in np.arange(0.1, 1.0, 0.1):
            print('Fetching Payout (%) for', sym, bp)
            cur_price = await get_latest_price(api, sym)
            payout_perc = (await open_proposal(api, sym, barrier=round(cur_price * (1 + bp / 100), 2)))['proposal'][
                              'payout'] - 100
            results.append((sym, bp, payout_perc))
            time.sleep(1)

    df = pd.DataFrame(results, columns=['Symbol', 'Barrier (%)', 'Payout (%)'])
    df.to_csv('payout_results.csv')


async def open_trade(symbol, barrier, amount, duration=24):
    """First opens a proposal and then buys a contract"""
    api = DerivAPI(app_id=app_id)
    authorize = await api.authorize(api_token)
    # print(authorize)

    proposal = await open_proposal(api, symbol, barrier, duration=duration, amount=amount)
    # print(json.dumps(proposal, indent=2))

    buy = await buy_contract(api, proposal['proposal']['id'], int(proposal['proposal']['spot'] * 1.05))
    # print(json.dumps(buy, indent=2))

    return buy


async def close_trade(contract_id):
    """Closes a contract based on the contract id"""
    api = DerivAPI(app_id=app_id)
    authorize = await api.authorize(api_token)
    # print(authorize)

    sell = await sell_contract(api, contract_id)
    # print(json.dumps(sell, indent=2))

    return sell
# asyncio.run(fetch_payout_data())
