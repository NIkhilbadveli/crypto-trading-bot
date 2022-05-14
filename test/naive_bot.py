import json
import math

from enum import IntEnum

import pprint

import numpy as np
import pandas as pd

from datetime import datetime
import websocket
import os
import warnings

from matplotlib import pyplot as plt
from pandas.core.common import SettingWithCopyWarning

from trade_class import Trade, TradeStatus
from get_data import get_historical_data
from telegram_alerts import send_open_alert, send_close_alert, send_socket_disconnect
from forecast_model import ForecastModel


class BinanceSocket:
    """
    WSS socket which streams binance kline data in real-time.
    """

    def __init__(self, currency, base, interval, bot):
        self.bot = bot
        # See if we can change the update frequency
        self.socket_url = "wss://stream.binance.com:9443/ws/" + (currency + base).lower() + "@kline_" + interval
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
        close_time = int(json_message['k']['t'])
        try:
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

        def __init__(self, take_profit_percentage, short, selling_points, starting_balance=500, starting_stake=100,
                     compound=False, enable_forecast=False, model_retrain=False, leverage_enabled=False):
            self.starting_balance = starting_balance
            self.starting_stake = starting_stake
            self.take_profit_percentage = take_profit_percentage
            self.short = short
            self.selling_points = selling_points
            self.compound = compound
            self.currency = None
            self.interval = None
            self.enable_forecast = enable_forecast
            self.model_retrain = model_retrain
            self.leverage_enabled = leverage_enabled

    def __init__(self):
        self.price_data = None
        self.total_trading_period = None
        self.starting_balance = 500
        self.run_params = None
        self.trades = []
        self.pwt = None
        self.current_balance = self.starting_balance
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
        self.fee_perc = 0  # Percentage of the fee to be paid. Change the logic below so that fees are subtracted after profit with leverage
        self.trades_file = 'trades.csv'

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
                                                end_date=end_date)

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
        self.total_trading_period = (datetime.fromtimestamp(self.price_data[-1, 0] / 1000.0) - datetime.fromtimestamp(
            self.price_data[0, 0] / 1000.0)).total_seconds() / 86400

        if self.run_params.leverage_enabled:
            self.leverage = 5
            self.margin_factor = 3

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
        self.month_wise_analysis(tp)

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
            return

        # Assumes that total-trading-period is already set

        print('No. of trades', len(self.trades))
        print("Current balance: ", self.current_balance)

        total_profit = 0
        for trade in self.trades:
            if trade.trade_status == TradeStatus.CLOSED or trade.trade_status == TradeStatus.CLOSED_BY_MARGIN_CALL:
                total_profit += trade.pl_abs

        # print('Total profit is {} for the {} days trading period'.format(total_profit, self.total_trading_period))
        roi = (total_profit * 100 / self.starting_balance) * (365 / self.total_trading_period)
        print('Projected yearly ROI is', roi, '%')
        # print('Average profit per day is', total_profit / self.total_trading_period)
        return total_profit
        # print('Average trade period is', trades[:, 5].sum() / len(trades))
        # print('Min trade period is', np.min(trades[:, 5]))
        # print('Max trade period is', np.max(trades[:, 5]))

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
        return -(pl_perc - self.fee_perc) * trade.leverage * trade.stake / 100 >= trade.margin_amount

    def do_the_magic(self, candle_info):
        """
        This is where all the magic happens.
        :param candle_info:
        :return:
        """
        current_time = candle_info[0]
        current_price = candle_info[1]
        max_open_period = self.run_params.selling_points[0][0]
        S = self.current_balance * 0.1  # Stake amount to be used for a new trade
        # pwt : Present Working Trade
        # max_loss = self.run_params.selling_points[-1][1]

        # should_open = True
        # self.skip_count += 1
        # if self.run_params.enable_forecast and self.present_working_trade is None and self.skip_count > self.skip_period:
        #     decided, short = self.forecast_model.predict_short(current_time, current_price)
        #     self.skip_count = 0
        #     if decided:
        #         print('Shorting at', current_time, 'for', short)
        #         self.run_params.short = short
        #     should_open = decided

        # Open a new trade if possible
        if self.pwt is None and S >= 10 and self.current_balance >= self.margin_factor * S:
            self.pwt = Trade(buy_time=current_time, buy_price=current_price,
                             stake=S)
            self.pwt.opening_balance = self.current_balance
            self.add_params_to_present_trade()
            self.pwt.margin_amount = self.margin_factor * self.pwt.stake

            # Get uuid and set it as the trade_id
            # self.present_working_trade.id = self.get_new_trade_id()

            # reduce the stake amount from current balance
            self.current_balance -= self.pwt.stake * self.margin_factor

            if self.run_params.enable_forecast:
                self.run_params.short = self.forecast_model.predict_short(current_time, current_price)
                self.pwt.short = self.run_params.short
                # print('Forecasted short at', current_time, 'for', self.run_params.short)

            # Print the trade details
            if self.bot_mode == BotMode.DRY_RUN:
                print("\n")
                print("New trade opened at {} at the buy price of {}".format(current_time, current_price))

            if self.bot_mode == BotMode.DRY_RUN:
                send_open_alert(self.pwt)

            self.trades.append(self.pwt)
            # Save trades to disk
            self.append_trades_to_csv(self.pwt)

        # Close for profit or convert the trade to a loss and open a new one
        if self.pwt is not None:
            open_period = (datetime.fromtimestamp(current_time / 1000.0) - datetime.fromtimestamp(
                self.pwt.buy_time / 1000.0)).total_seconds() / 86400  # in days
            pl_perc = \
                (current_price - self.pwt.buy_price) * 100 / self.pwt.buy_price

            if self.run_params.short:
                pl_perc = -pl_perc

            idx = self.trades.index(self.pwt)
            # Profit loop. Note that this takes only the current working trade into consideration
            if pl_perc >= self.run_params.take_profit_percentage:
                # add the profit amount to stake amount
                pl_abs = self.pwt.stake * (pl_perc - self.fee_perc) / 100
                pl_abs = self.pwt.leverage * pl_abs
                self.current_balance += (self.pwt.margin_amount + pl_abs)
                # new_stake = self.current_balance * 20 / 100
                # if self.run_params.compound and new_stake >= 10:
                #     self.current_stake = new_stake
                # self.current_stake += pl_abs / 2  # This is like re-investing profits

                self.trades[idx].sell_time = current_time
                self.trades[idx].sell_price = current_price
                self.trades[idx].pl_abs = pl_abs
                self.trades[idx].trade_status = TradeStatus.CLOSED
                self.trades[idx].closing_balance = self.trades[idx].opening_balance + pl_abs

                # Print the trade details
                if self.bot_mode == BotMode.DRY_RUN:
                    print("\n")
                    print("Present Trade closed at {} with profit percentage {}".format(current_time, pl_perc))
                    send_close_alert(self.trades[idx])

                self.update_trade(self.trades[idx])

                self.pwt = None
                open_period = 0

            # Margin call checking loop
            if self.run_params.leverage_enabled and self.is_margin_call(pl_perc, self.pwt):
                self.trades[idx].sell_time = current_time
                self.trades[idx].sell_price = current_price
                self.trades[idx].pl_abs = -self.pwt.margin_amount
                self.trades[idx].trade_status = TradeStatus.CLOSED_BY_MARGIN_CALL
                self.trades[idx].closing_balance = self.trades[idx].opening_balance + self.trades[idx].pl_abs

                # Print the trade details
                if self.bot_mode == BotMode.DRY_RUN:
                    print("\n")
                    print("Present Trade closed at {} due to margin call with loss {}".format(current_time,
                                                                                              self.trades[idx].pl_abs))
                    send_close_alert(self.trades[idx])

                self.update_trade(self.trades[idx])

                self.pwt = None
                open_period = 0

            if open_period >= max_open_period:
                self.trades[idx].trade_status = TradeStatus.OPEN_FOR_LOSS
                self.update_trade(self.trades[idx])
                # Open a new trade
                self.pwt = None

        # Handle any trades that are open for loss
        for idx, trade in enumerate(self.trades):
            if trade.trade_status == TradeStatus.OPEN_FOR_LOSS:
                open_period = (datetime.fromtimestamp(current_time / 1000.0) - datetime.fromtimestamp(
                    trade.buy_time / 1000.0)).total_seconds() / 86400
                pl_perc = (current_price - trade.buy_price) * 100 / trade.buy_price
                if trade.short:
                    pl_perc = -pl_perc

                for period, stop_loss_percentage in self.run_params.selling_points:
                    if open_period >= period and -stop_loss_percentage <= pl_perc:
                        pl_abs = trade.stake * (pl_perc - self.fee_perc) / 100
                        pl_abs = self.leverage * pl_abs
                        if -pl_abs >= trade.margin_amount:
                            continue  # Skipping if the loss is more than the margin amount
                        self.current_balance += (trade.margin_amount + pl_abs)
                        # self.current_stake = self.dry_run_params.starting_stake
                        # Update the trade status
                        self.trades[idx].sell_time = current_time
                        self.trades[idx].sell_price = current_price
                        self.trades[idx].pl_abs = pl_abs
                        self.trades[idx].trade_status = TradeStatus.CLOSED
                        self.trades[idx].closing_balance = self.trades[idx].opening_balance + self.trades[idx].pl_abs

                        # Print the trade details
                        if self.bot_mode == BotMode.DRY_RUN:
                            print("\n")
                            print(
                                "Past Trade at {} closed at {} with loss percentage {}".format(trade.buy_time,
                                                                                               current_time,
                                                                                               pl_perc))
                            send_close_alert(self.trades[idx])

                        self.update_trade(self.trades[idx])  # Update the trade to .csv
                        break

                # Margin call checking loop
                if self.run_params.leverage_enabled and self.is_margin_call(pl_perc, trade):
                    self.trades[idx].sell_time = current_time
                    self.trades[idx].sell_price = current_price
                    self.trades[idx].pl_abs = -self.trades[idx].margin_amount
                    self.trades[idx].trade_status = TradeStatus.CLOSED_BY_MARGIN_CALL
                    self.trades[idx].closing_balance = self.trades[idx].opening_balance + self.trades[idx].pl_abs

                    # Print the trade details
                    if self.bot_mode == BotMode.DRY_RUN:
                        print("\n")
                        print("Present Trade closed at {} due to margin call with loss {}".format(current_time,
                                                                                                  self.trades[
                                                                                                      idx].pl_abs))
                        send_close_alert(self.trades[idx])

                    self.update_trade(self.trades[idx])

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
            if row['currency'] != self.run_params.currency or row['interval'] != self.run_params.interval \
                    or row['short'] != self.run_params.short or row['compound'] != self.run_params.compound:
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
        if self.forecast_model is None:
            self.forecast_model = ForecastModel(currency, base)
        self.run_params = params
        self.run_params.currency = currency
        self.run_params.interval = interval
        self.bot_mode = BotMode.DRY_RUN
        self.current_balance = params.starting_balance
        # self.current_stake = params.starting_stake
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
            # print('Current stake:-', self.current_stake)
            print('\n')
        socket_client = BinanceSocket(currency, base, interval, self)
        socket_client.start_listening()
