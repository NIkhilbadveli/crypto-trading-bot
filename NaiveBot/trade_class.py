import enum


class TradeStatus(enum.Enum):
    """
    Enum for the trade status
    """
    OPEN_FOR_PROFIT = 1
    OPEN_FOR_LOSS = 2
    CLOSED = 3


class Trade:
    """
    A trade object which contains the time of the trade, the price at which the trade was made and the profit/loss
    """

    def __init__(self, buy_time, buy_price, stake, sell_time=None, sell_price=None, pl_abs=None):
        self.currency = None
        self.interval = None
        self.short = False
        self.compound = False
        self.buy_time = buy_time
        self.buy_price = buy_price
        self.sell_time = sell_time
        self.sell_price = sell_price
        self.stake = stake
        self.pl_abs = pl_abs
        self.trade_status = TradeStatus.OPEN_FOR_PROFIT
        self.opening_balance = None
        self.closing_balance = None