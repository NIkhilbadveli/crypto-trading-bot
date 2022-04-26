import json
import pickle
from enum import IntEnum

import pprint

import keras.models
import pandas as pd
from datetime import datetime
import websocket
import os

from sklearn.preprocessing import MinMaxScaler

from trade_class import Trade, TradeStatus
from get_data import get_historical_data
from telegram_alerts import send_open_alert, send_close_alert, send_socket_disconnect

from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands


class BinanceSocket:
    """
    WSS socket which streams binance kline data in real-time.
    """

    def __init__(self, currency, interval, bot):
        self.bot = bot
        # See if we can change the update frequency
        self.socket_url = "wss://stream.binance.com:9443/ws/" + (currency + 'USDT').lower() + "@kline_" + interval
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


class ForecastModel:
    """
    Class for the forecasting model
    """

    def __init__(self, data_file, model_file, model_scaler_file):
        self.interval = '1hr'
        self.data = pd.read_csv(data_file)
        self.scaler = pickle.load(open(model_scaler_file, 'rb'))
        self.model = keras.models.load_model(model_file)
        self.df = None
        self.price_array = None
        self.prepare_input()

    def add_technical_indicators(self, df):
        """
        Add MACD, ADX, RSI, AO, BB to the dataframe
        :return:
        """
        # Create MACD using ta library
        df['macd'] = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9).macd()

        # Create ADX using ta library
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()

        # Create upper and lower bollinger bands using ta library
        indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['upper_bb'] = indicator_bb.bollinger_hband()
        df['lower_bb'] = indicator_bb.bollinger_lband()
        df['bb_perc'] = indicator_bb.bollinger_pband()  # Percentage band

        # Create RSI using ta library
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

        # Create Awesome Osciallator using ta library
        df['ao'] = AwesomeOscillatorIndicator(high=df['high'], low=df['low'], window1=5,
                                              window2=34).awesome_oscillator()
        return df

    def inverse_transform(self, y):
        """
        This function is used to inverse transform the predicted values to the original scale.
        :param y:
        :return:
        """
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler.min_, y_scaler.scale_ = self.scaler.min_[0], self.scaler.scale_[0]
        y = y.reshape(-1, 1)
        y = y_scaler.inverse_transform(y)
        return y

    def prepare_input(self):
        """
        This function is used to prepare the input data for the model.
        :return:
        """
        self.data.dropna(inplace=True)
        self.data['date'] = pd.to_datetime(self.data['open_time'], unit='ms')
        self.data.drop_duplicates(subset='date', keep='first', inplace=True)
        df = self.data[['date', 'close', 'high', 'low']]
        df.set_index('date', inplace=True)

        # Add technical indicators
        df = self.add_technical_indicators(df)

        df = df.dropna()
        df.drop(labels=['high', 'low'], inplace=True, axis=1)
        self.df = df

        self.price_array = df.to_numpy()
        self.price_array = self.scaler.transform(self.price_array)
        self.price_array = self.price_array[:, 1:]

    def predict_short(self, x_timestamp, x_price):
        """
        This function will predict if we should take a short position.
        :param x_timestamp:
        :param x_price:
        :return:
        """
        p, q = 48, 24
        # Find the index of the closest past timestamp
        index = -1
        date_values = self.df.index.values
        x_timestamp = int(x_timestamp)
        for i in range(len(self.df.index.values)):
            cur_timestamp = int(str(pd.Timestamp(date_values[i]).value)[:13])
            # assert len(str(cur_timestamp)) == len(str(x_timestamp))
            if cur_timestamp >= x_timestamp:
                index = i - 1

        if index < p:
            return False  # Going long position

        # Get the past data
        past_data = self.price_array[index - p:index, :]
        past_data = past_data.reshape(1, p, past_data.shape[1])
        y_pred = self.model.predict(past_data)
        y_pred = self.inverse_transform(y_pred)

        print(y_pred)

        # If at least one of the predicted values is greater than the current price by 1%, then we should go long
        for i in range(len(y_pred)):
            diff = (y_pred[i] - x_price) * 100 / x_price
            if diff >= 1:
                return False
            elif diff <= -1:
                return True
        return False


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
                     compound=False):
            self.starting_balance = starting_balance
            self.starting_stake = starting_stake
            self.take_profit_percentage = take_profit_percentage
            self.short = short
            self.selling_points = selling_points
            self.compound = compound
            self.currency = None
            self.interval = None

    def __init__(self):
        self.price_data = None
        self.total_trading_period = None
        self.starting_balance = 500
        self.run_params = None
        self.trades = []
        self.present_working_trade = None
        self.current_balance = self.starting_balance
        self.current_stake = 100
        self.bot_mode = BotMode.BACK_TEST
        self.forecast_model = ForecastModel(data_file='btc_data_1hr.csv', model_file='model.h5',
                                            model_scaler_file='scaler.pkl')

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

        print("Performing backtest...")
        for row in self.price_data:
            self.do_the_magic((row[0], row[1]))
        print("Performing backtest... Done")
        print("\n")
        self.summarize_trades()

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

        open_trades_amount = 0
        total_profit = 0
        for trade in self.trades:
            if trade.trade_status != TradeStatus.CLOSED:
                open_trades_amount += trade.stake
            else:
                total_profit += trade.pl_abs
        print("Sum of current invested amount: ", open_trades_amount)
        print('Total profit is {} for the {} days trading period'.format(total_profit, self.total_trading_period))
        print('Projected yearly ROI is',
              (total_profit * 100 / self.starting_balance) * (365 / self.total_trading_period), '%')
        print('Average profit per day is', total_profit / self.total_trading_period)
        # print('Average trade period is', trades[:, 5].sum() / len(trades))
        # print('Min trade period is', np.min(trades[:, 5]))
        # print('Max trade period is', np.max(trades[:, 5]))

    def add_params_to_present_trade(self):
        """
        Adds the parameters to the present working trade
        """
        self.present_working_trade.currency = self.run_params.currency
        self.present_working_trade.interval = self.run_params.interval
        self.present_working_trade.compound = self.run_params.compound
        self.present_working_trade.short = self.run_params.short

    def get_new_trade_id(self):
        """
        Returns a new trade id
        """
        return None
        # if not os.path.exists('trades.csv'):
        #     return 1
        # df = pd.read_csv('trades.csv')
        # if len(df) == 0:
        #     return 1
        # else:
        #     return df['id'].max() + 1

    def do_the_magic(self, candle_info):
        """
        This is where all the magic happens.
        :param candle_info:
        :return:
        """
        fee_perc = 0.25  # For both buy and sell combined
        current_time = candle_info[0]
        current_price = candle_info[1]

        # Open a new trade if possible
        if self.present_working_trade is None and self.current_balance >= self.current_stake:
            self.present_working_trade = Trade(buy_time=current_time, buy_price=current_price,
                                               stake=self.current_stake)
            self.present_working_trade.opening_balance = self.current_balance
            self.add_params_to_present_trade()
            # Get uuid and set it as the trade_id
            # self.present_working_trade.id = self.get_new_trade_id()
            self.trades.append(self.present_working_trade)
            # reduce the stake amount from current balance
            self.current_balance -= self.present_working_trade.stake

            # Print the trade details
            print("\n")
            print("New trade opened at {} at the buy price of {}".format(current_time, current_price))

            # if self.bot_mode == BotMode.BACK_TEST:
            #     self.run_params.short = self.forecast_model.predict_short(current_time, current_price)

            if self.bot_mode == BotMode.DRY_RUN:
                send_open_alert(self.present_working_trade)

            # Save trades to disk
            self.append_trades_to_csv(self.present_working_trade)

        # Close for profit or convert the trade to a loss and open a new one
        if self.present_working_trade is not None:
            open_period = (datetime.fromtimestamp(current_time / 1000.0) - datetime.fromtimestamp(
                self.present_working_trade.buy_time / 1000.0)).total_seconds() / 86400
            pl_perc = \
                (current_price - self.present_working_trade.buy_price) * 100 / self.present_working_trade.buy_price

            if self.run_params.short:
                pl_perc = -pl_perc

            idx = self.trades.index(self.present_working_trade)
            # Profit loop
            if pl_perc >= self.run_params.take_profit_percentage:
                # add the profit amount to stake amount
                pl_abs = self.present_working_trade.stake * (pl_perc - fee_perc) / 100
                self.current_balance += (self.current_stake + pl_abs)
                if self.run_params.compound:
                    self.current_stake += pl_abs / 2  # This is like re-investing profits

                self.trades[idx].sell_time = current_time
                self.trades[idx].sell_price = current_price
                self.trades[idx].pl_abs = pl_abs
                self.trades[idx].trade_status = TradeStatus.CLOSED
                self.trades[idx].closing_balance = self.current_balance

                # Print the trade details
                print("\n")
                print("Present Trade closed at {} with profit percentage {}".format(current_time, pl_perc))

                if self.bot_mode == BotMode.DRY_RUN:
                    send_close_alert(self.trades[idx])

                self.update_closed_trade(self.trades[idx])

                self.present_working_trade = None
                open_period = 0

            max_open_period = self.run_params.selling_points[0][0]
            if open_period >= max_open_period:
                self.trades[idx].trade_status = TradeStatus.OPEN_FOR_LOSS
                self.update_closed_trade(self.trades[idx])
                # Open a new trade
                self.present_working_trade = None

        # Handle any trades that are open for loss
        for idx, trade in enumerate(self.trades):
            if trade.trade_status == TradeStatus.OPEN_FOR_LOSS:
                open_period = (datetime.fromtimestamp(current_time / 1000.0) - datetime.fromtimestamp(
                    trade.buy_time / 1000.0)).total_seconds() / 86400
                pl_perc = (current_price - trade.buy_price) * 100 / trade.buy_price
                if self.run_params.short:
                    pl_perc = -pl_perc

                for period, stop_loss_percentage in self.run_params.selling_points:
                    if open_period >= period and 0 <= -pl_perc <= stop_loss_percentage:
                        pl_abs = trade.stake * (pl_perc - fee_perc) / 100
                        self.current_balance += (trade.stake + pl_abs)
                        # self.current_stake = self.dry_run_params.starting_stake
                        # Update the trade status
                        self.trades[idx].sell_time = current_time
                        self.trades[idx].sell_price = current_price
                        self.trades[idx].pl_abs = pl_abs
                        self.trades[idx].trade_status = TradeStatus.CLOSED
                        self.trades[idx].closing_balance = self.current_balance

                        # Print the trade details
                        print("\n")
                        print(
                            "Past Trade at {} closed at {} with loss percentage {}".format(trade.buy_time, current_time,
                                                                                           pl_perc))
                        if self.bot_mode == BotMode.DRY_RUN:
                            send_close_alert(self.trades[idx])

                        self.update_closed_trade(self.trades[idx])  # Update the trade to .csv
                        break

    def append_trades_to_csv(self, trade):
        """
        Save the trades to disk
        """
        df = pd.DataFrame([trade.__dict__])
        file_name = 'trades.csv'
        # Instead of appending everytime, update the corresponding row whenever trade is closed
        with open(file_name, 'a', newline='\n') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

    def update_closed_trade(self, trade):
        """
        Update the closed trade or opened for loss trade in the trades list
        :param trade:
        :return:
        """
        df = pd.read_csv('trades.csv')
        d = {k: v for k, v in trade.__dict__.items() if
             k in ['currency', 'interval', 'buy_time', 'buy_price', 'short', 'compound']}
        m = (df[list(d)] == pd.Series(d)).all(axis=1)
        df.update(pd.DataFrame(trade.__dict__, index=df.index[m]))
        # df.loc[df['id'] == trade.id, trade.__dict__.keys()] = trade.__dict__.values()
        df.to_csv('trades.csv', index=False)

    def update_trades_list(self):
        """
        Update the trades list
        """
        if not os.path.exists('trades.csv'):
            return
        df = pd.read_csv('trades.csv')
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
                self.present_working_trade = trade
                self.current_balance = self.present_working_trade.opening_balance - self.present_working_trade.stake
                self.current_stake = self.present_working_trade.stake

        self.trades = trades_list

    def dry_run(self, currency, interval, params: BackTestParams):
        """
        Creates a simulated live-run with demo account. This will also try to email the trades that happen as we go.
        Uses binance socket to listen for the price updates.
        :return:
        """
        self.run_params = params
        self.run_params.currency = currency
        self.run_params.interval = interval
        self.bot_mode = BotMode.DRY_RUN
        self.current_balance = params.starting_balance
        self.current_stake = params.starting_stake
        self.update_trades_list()

        df = pd.DataFrame([trade.__dict__ for trade in self.trades])
        print('Existing trades from the csv file:-')
        print(df.tail())
        if self.present_working_trade:
            print('\nPresent working trade:-')
            pp = pprint.PrettyPrinter(depth=4)
            pp.pprint(self.present_working_trade.__dict__)
            print('\n')
            print('Current balance:- ', self.current_balance)
            print('Current stake:-', self.current_stake)
            print('\n')
        socket_client = BinanceSocket(currency, interval, self)
        socket_client.start_listening()
