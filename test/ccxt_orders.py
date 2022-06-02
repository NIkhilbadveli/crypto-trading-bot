"""
This file is for managing orders using ccxt library for any exchange.
"""
import ccxt


class CcxtOrders:
    """
    This class is for managing orders using ccxt library for any exchange.
    """

    def __init__(self, symbol, leverage):
        """
        Initialize the class.
        """
        # Api keys for futures test account
        api_key = 'dd05414a31ad8e9e6ce66333a3134e4c871972292153ff63bae3d700d3831f19'
        api_secret = 'da487d0428c7f1de6b149004f853b7ec20a0e292d0c5692e3c56de4139dd734e'

        self.leverage = leverage

        self.symbol = symbol

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # https://github.com/ccxt/ccxt/wiki/Manual#rate-limit
            'options': {
                'defaultType': 'future',  # or 'margin'
            }
        })
        self.exchange.set_sandbox_mode(True)
        self.exchange.verbose = False

        markets = self.exchange.load_markets()  # https://github.com/ccxt/ccxt/wiki/Manual#loading-markets
        market = self.exchange.market(symbol)

        self.exchange.fapiPrivate_post_leverage({  # https://github.com/ccxt/ccxt/wiki/Manual#implicit-api-methods
            'symbol': market['id'],  # https://github.com/ccxt/ccxt/wiki/Manual#symbols-and-market-ids
            'leverage': self.leverage,  # target initial leverage, int from 1 to 125
        })

        # self.exchange.fapiPrivatePostMargintype({
        #     'symbol': market['id'],
        #     'marginType': 'CROSS'
        # })

        # self.exchange.fapiPrivate_post_positionside_dual({
        #     'dualSidePosition': 'true'
        # })

    def get_balance(self):
        """
        Get the balance using ccxt library.
        """
        return float(self.exchange.fetch_balance()['info']['totalWalletBalance'])

    def open_position(self, short, stake):
        """
        Create an order using ccxt library.
        """
        asset_price = self.exchange.fetch_ticker(self.symbol)['close']  # Might be different due to spread in live
        order_amount = stake * self.leverage / asset_price  # After leverage
        order_amount = self.exchange.amount_to_precision(symbol=self.symbol, amount=order_amount)
        print('Order amount:', order_amount)

        if not short:
            order = self.exchange.create_market_buy_order(self.symbol, order_amount, params={
                'positionSide': 'LONG'
            })
        else:
            order = self.exchange.create_market_sell_order(self.symbol, order_amount, params={
                'positionSide': 'SHORT'
            })

        """
        {'info': {'orderId': '860178063',
  'symbol': 'ETHUSDT',
  'status': 'FILLED',
  'clientOrderId': 'x-xcKtGhcube91e6d49bab3b8e30de6',
  'price': '0',
  'avgPrice': '1747.00000',
  'origQty': '0.286',
  'executedQty': '0.286',
  'cumQty': '0.286',
  'cumQuote': '499.64200',
  'timeInForce': 'GTC',
  'type': 'MARKET',
  'reduceOnly': False,
  'closePosition': False,
  'side': 'BUY',
  'positionSide': 'LONG',
  'stopPrice': '0',
  'workingType': 'CONTRACT_PRICE',
  'priceProtect': False,
  'origType': 'MARKET',
  'updateTime': '1653686945494'},
 'id': '860178063',
 'clientOrderId': 'x-xcKtGhcube91e6d49bab3b8e30de6',
 'timestamp': None,
 'datetime': None,
 'lastTradeTimestamp': None,
 'symbol': 'ETH/USDT',
 'type': 'market',
 'timeInForce': 'IOC',
 'postOnly': False,
 'reduceOnly': False,
 'side': 'buy',
 'price': 1747.0,
 'stopPrice': None,
 'amount': 0.286,
 'cost': 499.642,
 'average': 1747.0,
 'filled': 0.286,
 'remaining': 0.0,
 'status': 'closed',
 'fee': None,
 'trades': [],
 'fees': []}"""

        return order

    def get_latest_trade(self):
        """
        Get the latest trade.
        """
        return self.exchange.fetch_my_trades(self.symbol)[-1]

    def close_position(self, short, base_qty):
        """
        Close an order.
        """
        if short:
            order = self.exchange.create_market_buy_order(self.symbol, base_qty, params={
                'positionSide': 'LONG'
            })
        else:
            order = self.exchange.create_market_sell_order(self.symbol, base_qty, params={
                'positionSide': 'SHORT'
            })

        return order

    def check_order_status(self, order_id):
        """
        Check the status of an order.
        """
        return self.exchange.fetch_order(id=order_id, symbol=self.symbol)['info']['status']

    def are_there_open_orders(self):
        """
        Check if there are open orders.
        """
        open_orders = self.exchange.fetch_open_orders(self.symbol)
        return len(open_orders) > 0
