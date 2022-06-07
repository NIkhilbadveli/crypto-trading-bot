import json
import math
import time

from enum import IntEnum

import pprint

import pandas as pd

from datetime import datetime
import websocket
import os
import warnings

from pandas.core.common import SettingWithCopyWarning

from trade_class import Trade, TradeStatus
from get_data import get_historical_data
from telegram_alerts import send_open_alert, send_close_alert, send_socket_disconnect
from forecast_model import ForecastModel
from ccxt_orders import CcxtOrders


class BinanceSocket:
    """
    WSS socket which streams binance kline data in real-time.
    """

    def __init__(self, currency, base, interval, bot):
        self.bot = bot
        contract_type = 'perpetual'
        self.socket_url = "wss://fstream.binance.com/ws/" + (
                currency + base).lower() + '_' + contract_type + "@continuousKline_" + interval
        print('Socket URL :', self.socket_url)
        self.ws = websocket.WebSocketApp(self.socket_url, on_open=lambda ws: self.on_open(ws),
                                         on_close=lambda ws: self.on_close(ws),
                                         on_message=lambda ws, msg: self.on_message(ws, msg))

    def start_listening(self):
        print("Trying to start listening...")
        self.ws.run_forever()

    def on_open(self, ws):
        print("Socket connected!")

    def on_close(self, ws):
        send_socket_disconnect()
        print("Socket disconnected!")

    def on_message(self, ws, message):
        json_message = json.loads(message)
        close_price = float(json_message['k']['c'])
        close_time = int(json_message['k']['T'])
        # print('Close status:', json_message['k']['x'])
        try:
            if json_message['k']['x']:
                print('Current price: ', close_price)
                print('Current time: ', datetime.fromtimestamp(close_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                      close_time)
                self.bot.do_the_magic((close_time, close_price))
        except Exception as e:
            send_socket_disconnect()
            print(e)


class BotMode(IntEnum):
    """
    Enum for the bot mode
    """
    DRY_RUN = 1
    BACK_TEST = 2


class NaiveBot:
    """
    A basic bot which implements gradual selling and with a preset maximum of open trades implicitly determined by the
    starting balance and the starting stake amount.
    """

    class BackTestParams:
        """
        Parameters for the back test
        """

        def __init__(self, take_profit, short, max_days, starting_balance=500, min_trade_amt=10, starting_stake=100,
                     stake_perc=0.05, compound=False, enable_forecast=False, model_retrain=False,
                     leverage_enabled=False, lev_mar=(1, 1)):
            self.starting_balance = starting_balance
            self.starting_stake = starting_stake
            self.stake_percentage = stake_perc
            self.take_profit = take_profit
            self.short = short  # Maybe it's better to remove it from run parameters.
            self.max_days = max_days
            self.compound = compound
            self.currency = None
            self.interval = None
            self.enable_forecast = enable_forecast
            self.model_retrain = model_retrain
            self.leverage_enabled = leverage_enabled
            self.lev_mar = lev_mar
            self.min_trade_amt = min_trade_amt

    def __init__(self):
        self.starting_balance = 500  # Default values
        self.current_balance = 500
        self.price_data = None
        self.total_trading_period = None
        self.run_params = None
        self.trades = []
        self.pwt = None
        # self.current_stake = 100
        self.bot_mode = BotMode.BACK_TEST
        # Ignore warnings
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        self.forecast_model = None
        self.big_loss_trades = []
        self.skip_count = 60
        self.skip_period = 60
        self.margin_factor = 1  # Amount of margin from the balance times the stake including the stake
        self.leverage = 1  # Amount of leverage to open the trade with
        self.fee_perc = 0.04  # Percentage of the fee to be paid for each trade (opening / closing).
        self.trades_file = 'trades.csv'
        self.save_to_file = True
        self.ccxt_orders: CcxtOrders = None
        self.socket_client = None

    def perform_backtest(self, currency, base, start_date, end_date, interval, params: BackTestParams):
        """
        Performs a backtest on the given price data and summarizes the results.
        :param base:
        :param params:
        :param interval:
        :param end_date:
        :param start_date:
        :param currency:
        :return: The trades, the time of the last trade and the price of the last trade.
        """
        if self.forecast_model is None and params.enable_forecast:
            self.forecast_model = ForecastModel(currency, base, back_test=True, start_date=start_date,
                                                end_date=end_date, take_profit=params.take_profit)

        # Try to do some basic validations here
        # Also, maybe cache the previously downloaded data
        file_path = '../data/' + currency + '_' + base + '_' + interval + '_' + start_date + '_' + end_date + '.csv'
        if os.path.isfile(file_path):
            print('Loading data from file')
            self.price_data = pd.read_csv(file_path)[['close_time', 'close']].to_numpy()
        else:
            print("Downloading price data for {}/{} for time period {} and {}...".format(currency, base, start_date,
                                                                                         end_date))
            df = get_historical_data(currency, base, start_date, end_date, interval)
            df.to_csv(file_path)
            self.price_data = df[['close_time', 'close']].to_numpy()
            print("Downloading price data... Done")

        self.run_params = params
        self.run_params.currency = currency
        self.run_params.interval = interval
        self.starting_balance = self.run_params.starting_balance
        self.current_balance = self.starting_balance
        self.total_trading_period = (datetime.fromtimestamp(self.price_data[-1, 0] / 1000.0) - datetime.fromtimestamp(
            self.price_data[0, 0] / 1000.0)).total_seconds() / 86400
        self.save_to_file = True

        if self.run_params.leverage_enabled:
            self.leverage = self.run_params.lev_mar[0]
            self.margin_factor = self.run_params.lev_mar[1]

        self.ccxt_orders = CcxtOrders(symbol=currency + '/' + base, leverage=self.leverage)
        self.current_balance = self.ccxt_orders.get_balance()

        # Delete the trades.csv if already exists
        if os.path.isfile(self.trades_file):
            print('Deleting trades.csv file since it already exists')
            os.remove(self.trades_file)

        print("Performing backtest...")
        for row in self.price_data:
            self.do_the_magic((row[0], row[1]))
        print("Performing backtest... Done")
        print("\n")
        tp = self.summarize_trades()
        # self.month_wise_analysis(tp[0])
        tp.insert(1, start_date)
        return tp

    def month_wise_analysis(self, total_profit):
        """Generates a csv file of monthly returns"""
        if len(self.trades) == 0:
            print("Looks like no trades happened :(")
            return

        # print("\n\nGenerating monthly analysis report...", end='\n')
        df = pd.DataFrame([trd.__dict__ for trd in self.trades])
        assert len(df['buy_time']) != 0
        # Get the maximum of time difference between buy and sell in days
        max_time_diff = max(df['sell_time'] - df['buy_time']) / (1000 * 86400)
        # print("Maximum time difference between buy and sell: {} days".format(max_time_diff))

        # Get the open trades and print the time difference between buy and last timestamp in the data
        open_trades = df[df['trade_status'] == TradeStatus.OPEN_FOR_LOSS]
        open_trades_loss = 0
        if len(open_trades) > 0:
            for i, bt in open_trades.iterrows():
                time_diff = (self.price_data[-1, 0] - bt['buy_time']) / (1000 * 86400)
                loss = (self.price_data[-1, 1] - bt['buy_price']) / bt['buy_price'] * bt['stake']
                if bt['short']:
                    loss = -loss

                # If loss is greater than stake, then max loss is the stake itself
                if abs(loss) > bt['stake']:
                    loss = math.copysign(1, loss) * bt['stake']
                open_trades_loss += loss
                print("Open trade at {} with time difference {} days with loss {}".format(bt['buy_time'], time_diff,
                                                                                          loss))

        roi_adjusted = ((total_profit + open_trades_loss) * 100 / self.starting_balance) * (
                365 / self.total_trading_period)
        print('ROI after accounting for open trades loss: {}'.format(roi_adjusted))
        df['month'] = df['buy_time'].apply(lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m'))
        df_grouped = df.groupby(['currency', 'month'])
        df_final = df_grouped['pl_abs'].sum().to_frame('profit_usd')
        df_final = pd.concat([df_final, df_grouped.size().to_frame('total_trades')], axis=1, join="inner")
        # Cumulative ROI calculation
        df_final['cum_roi'] = df_final['profit_usd'].cumsum() * 100 / self.starting_balance
        df_final.to_csv('monthly_analysis.csv')
        # print("Generating monthly analysis report... done.")

        # Plotting the big loss trades
        # np_arr = np.array(self.big_loss_trades)
        # plt.plot(np_arr[:, 0], np_arr[:, 1], 'ro')
        # plt.xlabel('Open period (hrs)')
        # plt.ylabel('Profit/Loss (%)')
        # plt.show()

    def summarize_trades(self):
        """
        Summarizes the trades and prints the results
        :return:
        """

        if len(self.trades) == 0:
            print("Looks like no trades happened :(")
            return [0] * 17  # Return a list of zeros except for the date column

        # Assumes that total-trading-period is already set

        print('No. of trades', len(self.trades))
        print("Current balance: ", self.current_balance)

        total_profit = 0
        no_of_margin_calls = 0
        short_count = 0
        sum_profits = 0
        sum_losses = 0
        total_fee = 0
        for trade in self.trades:
            total_fee += trade.fee
            if trade.trade_status == TradeStatus.CLOSED or trade.trade_status == TradeStatus.CLOSED_BY_MARGIN_CALL:
                total_profit += trade.pl_abs
                if trade.pl_abs > 0:
                    sum_profits += trade.pl_abs
                else:
                    sum_losses += trade.pl_abs

            if trade.trade_status == TradeStatus.CLOSED_BY_MARGIN_CALL:
                no_of_margin_calls += 1

            if trade.short:
                short_count += 1

        # print('Total profit is {} for the {} days trading period'.format(total_profit, self.total_trading_period))
        roi = (total_profit * 100 / self.starting_balance) * (365 / self.total_trading_period)
        print('Projected yearly ROI is', roi, '%')
        # print('Average profit per day is', total_profit / self.total_trading_period)
        out = [self.run_params.currency, self.run_params.starting_balance, self.run_params.min_trade_amt, self.leverage,
               self.margin_factor, self.run_params.take_profit, self.run_params.max_days,
               self.run_params.stake_percentage]  # Settings
        num_trades = len(self.trades)
        short_perc = short_count / num_trades * 100
        out.extend([num_trades, num_trades / self.total_trading_period, short_perc])  # Trade stats
        sum_losses = abs(sum_losses)
        out.extend(
            [int(total_profit), int(total_fee), int(total_profit) / num_trades, round(sum_profits / sum_losses, 2),
             no_of_margin_calls,
             roi])  # Profit stats
        return out

    def add_params_to_present_trade(self):
        """
        Adds the parameters to the present working trade
        """
        self.pwt.currency = self.run_params.currency
        self.pwt.interval = self.run_params.interval
        self.pwt.compound = self.run_params.compound
        self.pwt.short = self.run_params.short
        self.pwt.leverage = self.leverage

    def get_new_trade_id(self):
        """
        Returns a new trade id
        """
        return None
        # if not os.path.exists(self.trades_file):
        #     return 1
        # df = pd.read_csv(self.trades_file)
        # if len(df) == 0:
        #     return 1
        # else:
        #     return df['id'].max() + 1

    def is_margin_call(self, pl_perc, trade):
        """Checks if a given trade will get a margin call"""
        if not trade:
            return False
        return -pl_perc * trade.leverage * trade.stake / 100 >= 0.98 * trade.margin_amount

    def get_days_diff(self, date1, date2):
        """
        Returns the number of days between two dates
        """
        return (datetime.fromtimestamp(date1 / 1000.0) - datetime.fromtimestamp(
            date2 / 1000.0)).total_seconds() / 86400

    def get_stake_amount(self):
        """
        Sample the stake amount from a normal distribution with min 10 and max 0.1% * B
        :return:
        """
        S = self.current_balance * self.run_params.stake_percentage
        mu = (10 + S) // 2
        sigma = (S - mu) / 3  # 3 Standard deviations will cover 99.7% of the data
        return S  # np.random.normal(mu, sigma)

    def close_trade(self, trade: Trade, idx, close_by_margin_call=False):
        """
        Closes a trade using ccxt_orders
        :param close_by_margin_call:
        :param idx:
        :param trade:
        :return:
        """
        if self.ccxt_orders.are_there_open_orders():
            print("There are existing open orders, not closing the trade")
            return
        print('\nTrying to close the position!')
        # Place order using ccxt_orders
        order = self.ccxt_orders.close_position(short=trade.short, base_qty=trade.base_qty)

        sts = order['info']['status']
        sleep_secs = 0
        # Check order status
        while sts.lower() != 'filled' and sleep_secs <= 180:
            time.sleep(1)
            sleep_secs += 1
            sts = self.ccxt_orders.check_order_status(order_id=order['info']['orderId'])

        # Once again, I'm assuming that the order will get filled within 3 mins.
        sell_fee = order['cost'] * self.fee_perc / 100
        pl_abs = order['cost'] - trade.stake * trade.leverage  # This might be wrong
        if trade.short:
            pl_abs = -pl_abs

        self.current_balance += trade.margin_amount + pl_abs - sell_fee

        self.trades[idx].sell_time = int(order['info']['updateTime'])
        self.trades[idx].sell_price = order['average']
        self.trades[idx].pl_abs = pl_abs
        self.trades[idx].fee += sell_fee
        self.trades[
            idx].trade_status = TradeStatus.CLOSED if not close_by_margin_call else TradeStatus.CLOSED_BY_MARGIN_CALL
        self.trades[idx].closing_balance = self.trades[idx].opening_balance + pl_abs - self.trades[idx].fee
        self.trades[idx].trade_period = self.get_days_diff(self.trades[idx].sell_time,
                                                           self.trades[idx].buy_time)

        # Print the trade details
        if self.bot_mode == BotMode.DRY_RUN:
            print("Present Trade closed at {} with profit USD {}".format(self.trades[idx].sell_time, pl_abs))
            send_close_alert(self.trades[idx])

        # Update trade in the csv file
        if self.save_to_file:
            self.update_trade(self.trades[idx])

    def do_the_magic(self, candle_info):
        """
        This is where all the magic happens.
        :param candle_info:
        :return:
        """
        current_time = candle_info[0]
        current_price = candle_info[1]

        if len(self.trades) >= 10:
            self.socket_client.ws.close()
            return

        if self.pwt is None and not self.ccxt_orders.are_there_open_orders():
            # S = self.get_stake_amount()  # Stake amount to be used for a new trade
            S = self.current_balance * self.run_params.stake_percentage
            if S >= self.run_params.min_trade_amt and self.current_balance >= self.margin_factor * S:
                if self.run_params.enable_forecast:
                    # print('Forecasted short at', current_time, 'for', self.run_params.short)
                    self.run_params.short = self.forecast_model.predict_short(current_time, current_price)

                # Place order using ccxt_orders
                order = self.ccxt_orders.open_position(short=self.run_params.short, stake=S, asset_price=current_price)

                sts = order['info']['status']
                sleep_secs = 0
                # Check order status
                while sts.lower() != 'filled' and sleep_secs <= 180:
                    time.sleep(1)
                    sleep_secs += 1
                    sts = self.ccxt_orders.check_order_status(order_id=order['info']['orderId'])

                # Assuming that the order will get filled in 3 mins.
                buy_fee = order['cost'] * self.fee_perc / 100  # Might not be exact

                # Fee for opening a new trade. Should this include the leverage?
                self.pwt = Trade(buy_time=int(order['info']['updateTime']), buy_price=order['average'],
                                 stake=(order['cost'] / self.leverage))
                self.pwt.id = order['id']
                self.pwt.fee = buy_fee
                self.pwt.base_qty = order['amount']  # Should ideally use order['filled']
                # This might not be full amount as expected.
                self.pwt.opening_balance = self.current_balance
                self.add_params_to_present_trade()
                self.pwt.margin_amount = self.margin_factor * self.pwt.stake

                # reduce the stake amount from current balance
                self.current_balance -= (self.pwt.stake * self.margin_factor + buy_fee)

                # Print the trade details
                if self.bot_mode == BotMode.DRY_RUN:
                    print("\n")
                    print("New trade opened at {} at the buy price of {}".format(self.pwt.buy_time, self.pwt.buy_price))
                    send_open_alert(self.pwt)
                    # Save trades to disk

                if self.save_to_file:
                    self.append_trades_to_csv(self.pwt)

                self.trades.append(self.pwt)

        # Close for profit or convert the trade to a loss and open a new one
        if self.pwt is not None:
            open_period = self.get_days_diff(current_time, self.pwt.buy_time)
            pl_perc = (current_price - self.pwt.buy_price) * 100 / self.pwt.buy_price
            if self.pwt.short:
                pl_perc = -pl_perc
            # print('Open period', open_period)
            # print('P/l percentage', pl_perc)
            idx = self.trades.index(self.pwt)
            # Profit loop. Note that this takes only the current working trade into consideration
            is_margin_call = self.is_margin_call(pl_perc, self.pwt)
            if pl_perc >= self.run_params.take_profit or is_margin_call:
                # Close the trade
                self.close_trade(self.pwt, idx, close_by_margin_call=is_margin_call)
                self.pwt = None
                open_period = 0

            if open_period >= 1:  # If the trade is open for more than 1 day, move it to OPEN_FOR_LOSS
                self.trades[idx].trade_status = TradeStatus.OPEN_FOR_LOSS
                if self.save_to_file:
                    self.update_trade(self.trades[idx])
                # Open a new trade
                self.pwt = None

        # Handle any trades that are open for loss
        for idx, trade in enumerate(self.trades):
            if trade.trade_status == TradeStatus.OPEN_FOR_LOSS:
                open_period = self.get_days_diff(current_time, trade.buy_time)
                pl_perc = (current_price - trade.buy_price) * 100 / trade.buy_price
                if trade.short:
                    pl_perc = -pl_perc
                # print('Open period', open_period)
                # print('P/l percentage', pl_perc)
                is_margin_call = self.is_margin_call(pl_perc, trade)
                if pl_perc >= self.run_params.take_profit or open_period >= self.run_params.max_days or is_margin_call:
                    # Close the trade
                    self.close_trade(trade, idx, close_by_margin_call=is_margin_call)
                    # if not close_status:
                    #     print('Trade close status is not filled!')
                    # else:
                    #     print('Trade closed at', current_time, 'with profit', pl_perc)

    def append_trades_to_csv(self, trade):
        """
        Save the trades to disk
        """
        df = pd.DataFrame([trade.__dict__])
        file_name = self.trades_file
        # Instead of appending everytime, update the corresponding row whenever trade is closed
        with open(file_name, 'a', newline='\n') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

    def update_trade(self, trade):
        """
        Update the closed trade or opened for loss trade in the trades list csv file
        :param trade:
        :return:
        """
        df = pd.read_csv(self.trades_file)
        d = {k: v for k, v in trade.__dict__.items() if
             k in ['currency', 'interval', 'buy_time', 'buy_price', 'short', 'compound']}
        m = (df[list(d)] == pd.Series(d)).all(axis=1)
        df.update(pd.DataFrame(trade.__dict__, index=df.index[m]))
        # df.loc[df['id'] == trade.id, trade.__dict__.keys()] = trade.__dict__.values()
        df.to_csv(self.trades_file, index=False)

    def update_trades_list(self):
        """
        Update the trades list whenever the script restarts by mistake, so that the self.trades is updated
        """
        if not os.path.exists(self.trades_file):
            return
        df = pd.read_csv(self.trades_file)
        trades_list = []
        for _, row in df.iterrows():
            # Todo: Might need to check for other columns as well.
            if row['currency'] != self.run_params.currency or row['interval'] != self.run_params.interval:
                continue
            trade = Trade(buy_time=row['buy_time'], buy_price=row['buy_price'], stake=row['stake'])
            trade.id = row['id']
            trade.currency = row['currency']
            trade.interval = row['interval']
            trade.short = row['short']
            trade.compound = row['compound']
            trade.sell_time = row['sell_time']
            trade.sell_price = row['sell_price']
            trade.pl_abs = row['pl_abs']
            trade.trade_status = row['trade_status']
            trade.opening_balance = row['opening_balance']
            trade.closing_balance = row['closing_balance']
            trade.trade_period = row['trade_period']
            trade.base_qty = row['base_qty']
            trade.fee = row['fee']
            trade.leverage = row['leverage']
            trade.margin_amount = row['margin_amount']

            trades_list.append(trade)
            if trade.trade_status == TradeStatus.OPEN_FOR_PROFIT:
                self.pwt = trade
                self.current_balance = self.pwt.opening_balance - self.pwt.stake
                # self.current_stake = self.pwt.stake

        self.trades = trades_list

    def dry_run(self, currency, base, interval, params: BackTestParams):
        """
        Creates a simulated live-run with demo account. This will also try to email the trades that happen as we go.
        Uses binance socket to listen for the price updates.
        :return:
        """
        if self.forecast_model is None and params.enable_forecast:
            self.forecast_model = ForecastModel(currency, base, back_test=False, take_profit=params.take_profit)

        self.run_params = params
        self.run_params.currency = currency
        self.run_params.interval = interval
        self.starting_balance = self.run_params.starting_balance
        self.current_balance = self.starting_balance
        self.bot_mode = BotMode.DRY_RUN
        if self.run_params.leverage_enabled:
            self.leverage = self.run_params.lev_mar[0]
            self.margin_factor = self.run_params.lev_mar[1]
        self.save_to_file = True

        self.ccxt_orders = CcxtOrders(symbol=currency + '/' + base, leverage=self.leverage)
        self.current_balance = self.ccxt_orders.get_balance()

        self.update_trades_list()

        df = pd.DataFrame([trade.__dict__ for trade in self.trades])
        print('Existing trades from the csv file:-')
        print(df.tail())

        if self.pwt:
            print('\nPresent working trade:-')
            pp = pprint.PrettyPrinter(depth=4)
            pp.pprint(self.pwt.__dict__)
            print('\n')
            print('Current balance:- ', self.current_balance)
            print('\n')

        self.socket_client = BinanceSocket(currency, base, interval, self)
        self.socket_client.start_listening()
