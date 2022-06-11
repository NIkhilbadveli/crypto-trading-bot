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
from get_data import get_futures_data
from telegram_alerts import send_open_alert, send_close_alert, send_socket_disconnect
from forecast_model import ForecastModel


class BinanceSocket:
    """
    WSS socket which streams binance kline data in real-time.
    """

    def __init__(self, currency, base, interval, bot):
        self.bot = bot
        contract_type = 'perpetual'
        # See if we can change the update frequency
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
                # print('Close price: ', close_price)
                # print('Close time: ', datetime.fromtimestamp(close_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                #       close_time)
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

        def __init__(self, take_profit, short, max_days, tp_days=1, starting_balance=500, min_trade_amt=10,
                     starting_stake=100,
                     stake_perc=0.2, compound=False, enable_forecast=False, model_retrain=False,
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
            self.tp_days = tp_days

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
        self.open_trades_history = []
        self.skip_count = 60
        self.skip_period = 60
        self.margin_factor = 1  # Amount of margin from the balance times the stake including the stake
        self.leverage = 1  # Amount of leverage to open the trade with
        self.fee_perc = 0.02  # Percentage of the fee to be paid for each trade (opening / closing).
        self.trades_file = 'trades.csv'
        self.save_to_file = True

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
        # print('Loading data from file... using it for backtest')
        # self.price_data = pd.read_csv('live_data_tested.csv')[['close_time', 'close']].to_numpy()
        file_path = '../data/' + currency + '_' + base + '_' + interval + '_' + start_date + '_' + end_date + '.csv'
        if os.path.isfile(file_path):
            print('Loading data from file... using it for backtest')
            self.price_data = pd.read_csv(file_path)[['close_time', 'close']].to_numpy()
        else:
            print("Downloading price data for {}/{} for time period {} and {}...".format(currency, base, start_date,
                                                                                         end_date))
            df = get_futures_data(currency, base, start_date, end_date, interval)
            df.to_csv(file_path)
            self.price_data = df[['close_time', 'close']].to_numpy()
            print("Downloading price data... Done")

        self.run_params = params
        self.run_params.currency = currency
        self.run_params.interval = interval
        self.starting_balance = self.run_params.starting_balance
        self.current_balance = self.starting_balance
        self.total_trading_period = self.get_days_diff(self.price_data[-1, 0], self.price_data[0, 0])
        self.save_to_file = True

        if self.run_params.leverage_enabled:
            self.leverage = self.run_params.lev_mar[0]
            self.margin_factor = self.run_params.lev_mar[1]

        # Delete the trades.csv if already exists
        if os.path.isfile(self.trades_file):
            print('Deleting trades.csv file since it already exists')
            os.remove(self.trades_file)

        print("Performing backtest...")
        i = 0
        while i < len(self.price_data):
            row = self.price_data[i]
            # self.close_margin_call_trades(current_time=row[0], current_price=row[1])
            # if (i + 1) % 60 == 0:
            self.do_the_magic((row[0], row[1]))
            i += 1
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
        pl_perc_arr = []
        for trade in self.trades:
            total_fee += trade.fee
            if trade.trade_status == TradeStatus.CLOSED or trade.trade_status == TradeStatus.CLOSED_BY_MARGIN_CALL:
                total_profit += trade.pl_abs
                pl_perc_arr.append(trade.pl_abs * 100 / trade.stake)
                if trade.pl_abs > 0:
                    sum_profits += trade.pl_abs
                else:
                    sum_losses += trade.pl_abs

            if trade.trade_status == TradeStatus.CLOSED_BY_MARGIN_CALL:
                no_of_margin_calls += 1

            if trade.short:
                short_count += 1

        # print('Total profit is {} for the {} days trading period'.format(total_profit, self.total_trading_period))
        roi = ((total_profit - total_fee) * 100 / self.starting_balance) * (365 / self.total_trading_period)
        print('Projected yearly ROI is', roi, '%')
        # print('Max number of trades open at a time', max(self.open_trades_history))
        print('Avg. profit percentage',
              sum([pl for pl in pl_perc_arr if pl > 0]) / (len(self.trades) - no_of_margin_calls))
        print('Avg. loss percentage',
              sum([pl for pl in pl_perc_arr if pl < 0]) / no_of_margin_calls)
        print('Model success rate', (1 - no_of_margin_calls / len(self.trades)) * 100)
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
        return -pl_perc * trade.leverage * trade.stake / 100 >= 0.5 * trade.margin_amount

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

    def close_margin_call_trades(self, current_time, current_price):
        """Close a trade if its margin call at any time in between hours"""
        # filtered_trades = filter(lambda x: x.trade_status in [TradeStatus.OPEN_FOR_PROFIT, TradeStatus.OPEN_FOR_LOSS],
        #                          self.trades)
        closed = False
        for idx, trade in enumerate(self.trades):
            if trade.trade_status == TradeStatus.OPEN_FOR_PROFIT:
                pl_perc = (current_price - trade.buy_price) * 100 / trade.buy_price
                # open_period = self.get_days_diff(current_time, trade.buy_time)
                if trade.short:
                    pl_perc = -pl_perc

                if self.is_margin_call(pl_perc, trade):
                    self.close_trade(current_time, current_price, pl_perc, idx, close_by_margin_call=True)
                    self.pwt = None
                    closed = True

            # if trade.trade_status == TradeStatus.OPEN_FOR_LOSS and self.is_margin_call(pl_perc, trade):
            #     self.close_trade(current_time, current_price, pl_perc, idx, close_by_margin_call=True)
            #     closed = True

        return closed

    def close_trade(self, current_time, current_price, pl_perc, idx, close_by_margin_call=False):
        """
        Closes a trade using ccxt_orders
        """
        if self.trades[idx] is None:
            return False
        # add the profit amount to stake amount
        cur_stake_worth = self.trades[idx].stake * (1 + pl_perc / 100)
        sell_fee = self.fee_perc * cur_stake_worth * self.leverage / 100
        pl_abs = cur_stake_worth - self.trades[idx].stake
        pl_abs = self.trades[idx].leverage * pl_abs  # Is this correct for leverage?

        self.trades[idx].sell_time = current_time
        self.trades[idx].sell_price = current_price
        self.trades[idx].pl_abs = 0.75 * self.trades[idx].margin_amount if pl_abs > 0 else -0.5 * self.trades[
            idx].margin_amount
        self.trades[idx].fee += sell_fee
        self.trades[
            idx].trade_status = TradeStatus.CLOSED if not close_by_margin_call else TradeStatus.CLOSED_BY_MARGIN_CALL
        self.trades[idx].closing_balance = self.trades[idx].opening_balance + self.trades[idx].pl_abs - self.trades[
            idx].fee
        self.trades[idx].trade_period = self.get_days_diff(self.trades[idx].sell_time,
                                                           self.trades[idx].buy_time)

        self.current_balance += self.trades[idx].margin_amount + self.trades[idx].pl_abs - self.trades[idx].fee

        # Print the trade details
        if self.bot_mode == BotMode.DRY_RUN:
            print("\n")
            print("Present Trade closed at {} with profit percentage {}".format(current_time, pl_perc))
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

        # if len(self.trades) > 0:
        #     open_trades_count = 0
        #     for trade in self.trades:
        #         if trade.trade_status in [1, 2]:
        #             open_trades_count += 1
        #     self.open_trades_history.append(open_trades_count)

        if self.pwt is None:
            # S = self.get_stake_amount()  # Stake amount to be used for a new trade
            S = self.current_balance * self.run_params.stake_percentage
            if S >= self.run_params.min_trade_amt and self.current_balance >= self.margin_factor * S:
                buy_fee = self.fee_perc * S * self.leverage / 100
                # Fee for opening a new trade. Should this include the leverage?
                self.pwt = Trade(buy_time=current_time, buy_price=current_price,
                                 stake=S)
                self.pwt.fee = buy_fee
                self.pwt.opening_balance = self.current_balance
                self.add_params_to_present_trade()
                self.pwt.margin_amount = self.margin_factor * self.pwt.stake

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

            idx = self.trades.index(self.pwt)
            # Profit loop. Note that this takes only the current working trade into consideration
            is_margin_call = self.is_margin_call(pl_perc, self.pwt)
            if pl_perc >= self.run_params.take_profit or is_margin_call or open_period >= self.run_params.max_days:
                self.close_trade(current_time, current_price, pl_perc, idx, close_by_margin_call=is_margin_call)
                self.pwt = None

            # if open_period >= self.run_params.tp_days:
            #     self.trades[idx].trade_status = TradeStatus.OPEN_FOR_LOSS
            #     if self.save_to_file:
            #         self.update_trade(self.trades[idx])
            #     self.pwt = None

        # Handle any trades that are open for loss
        # for idx, trade in enumerate(self.trades):
        #     if trade.trade_status == TradeStatus.OPEN_FOR_LOSS:
        #         open_period = self.get_days_diff(current_time, trade.buy_time)
        #         pl_perc = (current_price - trade.buy_price) * 100 / trade.buy_price
        #         if trade.short:
        #             pl_perc = -pl_perc
        #
        #         is_margin_call = self.is_margin_call(pl_perc, trade)
        #         if pl_perc >= self.run_params.take_profit or open_period >= self.run_params.max_days or is_margin_call:
        #             self.close_trade(current_time, current_price, pl_perc, idx, is_margin_call)

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
