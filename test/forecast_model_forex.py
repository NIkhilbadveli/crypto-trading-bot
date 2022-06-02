import os
import random

from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.schedulers.background import BackgroundScheduler
from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from keras.layers import Bidirectional, Activation, Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras.models
import pickle
import pandas as pd

from datetime import datetime, timedelta
from get_data import get_hourly_data, get_historical_data, get_data_forex
from telegram_alerts import send_training_alert


class ForecastModel:
    """
    Class for the forecasting model
    """

    def __init__(self, currency, base, back_test=False, start_date='', end_date='', model_retrain=False,
                 take_profit=1.0):
        self.back_test = back_test
        self.interval = '1h'  # Might be different, but for now it is 1 hour
        self.currency = currency
        self.base = base
        self.df = None
        self.take_profit = take_profit
        self.data_file = currency + '_' + base + '_' + '_data_' + self.interval + '.csv'
        self.model_file = currency + '_' + base + '_' + '_model_' + self.interval + '.h5'
        self.model_scaler_file = currency + '_' + base + '_' + '_scaler_' + self.interval + '.pkl'
        self.scaler = pickle.load(open(self.model_scaler_file, 'rb')) if os.path.isfile(
            self.model_scaler_file) else None
        self.model = keras.models.load_model(self.model_file) if os.path.isfile(self.model_file) else None
        if self.back_test:  # I guess we're using it for dry run as well. Might have to restructure later.
            file_path = '../data/' + currency + '_' + base + '_' + self.interval + '_' + start_date + '_' + end_date + '.csv'
            # Download the file if it doesn't exist
            if os.path.isfile(file_path):
                print('Loading backtest data from file... using it for forecasting by the model')
                df = pd.read_csv(file_path)
            else:
                print("Downloading price data for {}/{} for time period {} and {}...".format(currency, base, start_date,
                                                                                             end_date))
                df = get_data_forex(currency, base, start_date, end_date, self.interval)
                df['close_time'] = df.index.values
                df.to_csv(file_path)
                print("Downloading price data... Done")

            df['close_time'] = pd.to_datetime(df['close_time'])
            # Set close_time as index
            df.set_index('close_time', inplace=True, drop=True)
            self.back_test_data = df

        if self.model is None:
            # Schedule background training once a week
            # print('Training the model for the first time...')
            if not self.back_test:
                self.schedule_background_training()
            else:
                # Get start_date and end_date by subtracting 1 year from each
                st_date = datetime.strptime(start_date, '%Y-%m-%d')
                ed_date = datetime.strptime(end_date, '%Y-%m-%d')
                st_date = (st_date - timedelta(days=365)).strftime('%Y-%m-%d')
                # ed_date = (ed_date - timedelta(days=365)).strftime('%Y-%m-%d')
                print('Training the model for the period {} to {}...'.format(st_date, start_date))
                self.train_model(start_date=st_date, end_date=start_date)
            # print('Training done! Model saved to: ' + self.model_file)

        self.scheduler = None
        self.model_retrain = model_retrain

    def schedule_background_training(self):
        """
        Using apscheduler to schedule the training of the model once a week.
        :return:
        """

        def my_listener(event):
            """
            Listener for the apscheduler.
            :param event:
            """
            if event.exception:
                print('The job crashed :(')
            else:
                print('The job worked :)')

        print('Scheduling background training...')
        scheduler = BackgroundScheduler()
        scheduler.add_listener(my_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        scheduler.start()
        scheduler.add_job(self.train_model, 'interval', weeks=1, id=self.currency + '_' + self.base + '_training')
        for job in scheduler.get_jobs():
            job.modify(next_run_time=datetime.now())
        self.scheduler = scheduler
        self.see_running_jobs()
        print('Scheduling done!')

    def see_running_jobs(self):
        """
        Prints the currently running jobs
        :return:
        """
        print('Currently running jobs:')
        self.scheduler.print_jobs()

    def train_model(self, start_date='', end_date=''):
        """
        Trains the model
        :return:
        """
        # We will use past 'p' time steps to predict the future 'q' time steps
        p = 48
        q = 24
        num_epochs = 100  # These settings are used while training the model
        mini_batch_size = 32
        dropout = 0.2
        verbose = 0

        # Only if the start_date and end_date are empty, train the model for the last 1 year data
        if start_date == '' and end_date == '':
            # Find the timestamp in ms for 1 year back
            end_date = datetime.now().timestamp() * 1000
            # march_timestamp = int(cur_timestamp) - (30 * 24 * 60 * 60 * 1000)
            start_date = int(end_date) - (365 * 24 * 60 * 60 * 1000)

        # Get hourly data
        print('Getting hourly data starting one year back...')
        data = get_data_forex(self.currency, self.base, start_date=start_date, end_date=end_date,
                              frequency=self.interval)
        data.to_csv(self.data_file)  # Saving for later reference
        print('Got hourly data')
        print('Preparing data...')
        price_array, scaler = self.prepare_input(data)
        print('Data prepared')
        print('Splitting data...')
        # Split data into training and test sets using the above function
        x_train, y_train, x_test, y_test = self.make_train_test_data(p, q, price_array, test_size=0.1, shuffle=False)
        print('Shapes of x_train, x_test, y_train, y_test:', x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        print('Splitting data done')
        print('Training model...')
        model = self.get_model(p, q, dropout, x_train)
        # Train model
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=mini_batch_size, shuffle=True,
                            verbose=verbose)
        print('Loss in the last epoch is:', history.history['loss'][-1])
        y_pred = model.predict(x_test)
        # Print mean absolute of percentage error
        # y_test = np.delete(y_test, np.where(y_test == 0), axis=0)
        # mape = np.mean(np.abs((y_test.flatten() - y_pred.flatten()) / y_test.flatten())) * 100
        # print('mean absolute of percentage error:', mape)

        # Rename the model_file if model retrain is enabled
        # if self.model_retrain:
        #     self.model_file = self.model_file.replace('.h5', start_date + '_' + end_date + '.h5')

        # Save the model to disk
        model.save(self.model_file)
        # Save the scaler to disk
        pickle.dump(scaler, open(self.model_scaler_file, 'wb'))

        # Update the model and scaler variables
        self.model = model
        self.scaler = scaler

    def get_model(self, p, q, dropout, x_train):
        """
        Creates the model
        :param p:
        :param q:
        :param dropout:
        :param x_train:
        :return:
        """
        # Create LSTM model
        model = keras.Sequential()
        model.add(
            Bidirectional(LSTM(units=p, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(units=p * 2, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(units=p, return_sequences=False)))
        model.add(Dense(units=q))
        model.add(Activation('linear'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        return model

    def add_technical_indicators(self, df):
        """
        Add MACD, ADX, RSI, AO, BB to the dataframe
        :return:
        """
        # Create MACD using ta library
        df['macd'] = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).macd()

        # Create ADX using ta library
        df['adx'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()

        # Create upper and lower bollinger bands using ta library
        indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['upper_bb'] = indicator_bb.bollinger_hband()
        df['lower_bb'] = indicator_bb.bollinger_lband()
        df['bb_perc'] = indicator_bb.bollinger_pband()  # Percentage band

        # Create RSI using ta library
        df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()

        # Create Awesome Osciallator using ta library
        df['ao'] = AwesomeOscillatorIndicator(high=df['High'], low=df['Low'], window1=5,
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

    def make_train_test_data(self, p, q, p_arr, test_size=0.2, shuffle=True):
        """
        This function creates training and testing data from the input array using p and q values provided above.
        :param p_arr:
        :param test_size:
        :param shuffle:
        :return:
        """
        # Make sequences of 100 previous values and take the next value as the target
        time_offset = p + q
        step = 1
        d = []
        for index in range(0, len(p_arr) - time_offset, step):
            d.append(p_arr[index: index + time_offset, :])
        d = np.array(d)

        x_data = d[:, :-q, 1:]
        y_data = d[:, -q:, 0]
        y_data = y_data.reshape((y_data.shape[0], q, 1))

        # Split data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle,
                                                            random_state=42)

        return X_train, Y_train, X_test, Y_test

    def prepare_input(self, data, prediction_mode=False):
        """
        This function is used to prepare the input data for the model.
        :return:
        """
        data.dropna(inplace=True)
        df = data[['Close', 'High', 'Low']]
        # Add technical indicators
        df = self.add_technical_indicators(df)
        df = df.dropna()
        df.drop(labels=['High', 'Low'], inplace=True, axis=1)
        price_array = df.to_numpy()
        # Preprocess data using MinMaxScaler
        # Todo: Handle first time training manually
        if not prediction_mode:
            scaler = MinMaxScaler(feature_range=(0, 1))
            price_array = scaler.fit_transform(price_array)
        else:
            scaler = self.scaler  # Hopefully this is not None
            price_array = scaler.transform(price_array)
        return price_array, scaler

    def find_index(self, np_time):
        """
        This function is used to find the index of the timestamp in the dataframe.
        :param np_time:
        :return:
        """
        if self.df is None:
            data = self.back_test_data  # This could be None
            data.dropna(inplace=True)
            df = data[['Close', 'High', 'Low']]
            # Add technical indicators
            df = self.add_technical_indicators(df)
            df = df.dropna()
            df.drop(labels=['High', 'Low'], inplace=True, axis=1)
            self.df = df

        # Find the index of the closest past timestamp
        index = -1
        date_values = self.df.index.values
        for i in range(len(date_values)):
            if date_values[i] >= np_time:
                index = i - 1
                break

        price_array = self.df.to_numpy()
        # Preprocess data using MinMaxScaler
        scaler = self.scaler if self.scaler else MinMaxScaler(feature_range=(0, 1))
        price_array = scaler.fit_transform(price_array)
        return index, price_array

    # Convert predictions into trading signals
    def convert_to_signal(self, cur_close, p_o, p_h, p_l, p_c):
        if cur_close > p_o and cur_close > p_h and cur_close > p_l and cur_close > p_c:
            return True
        elif cur_close < p_o and cur_close < p_h and cur_close < p_l and cur_close < p_c:
            return False
        elif cur_close < p_o and cur_close < p_h and cur_close < p_c and cur_close > p_l:
            return True
        elif cur_close > p_o and cur_close > p_l and cur_close > p_c and cur_close < p_h:
            return False
        else:
            return True

    def predict_short(self, x_timestamp, x_price):
        """
        This function will predict if we should take a short position.
        :param x_timestamp:
        :param x_price:
        :return:
        """
        # Return randomly between True and False
        # return random.choice([True, False])
        p, q = 48, 24
        if not self.model:
            if not self.back_test:
                print('Model predicted - Long position! Because we have no model!')
            return False

        if not self.back_test:
            x_dt = datetime.fromtimestamp(x_timestamp / 1000)
            timestamp_48hrs_back = int((x_dt - timedelta(hours=48 + 33)).timestamp() * 1000)

            data = get_hourly_data(self.currency, self.base, start_timestamp=timestamp_48hrs_back,
                                   end_timestamp=x_timestamp)

            price_array, scaler = self.prepare_input(data, prediction_mode=True)
            past_data = price_array[:, 1:]
        else:
            # Using the index method
            index, price_array = self.find_index(x_timestamp)
            # print(index)
            if index < p:
                return False  # Going long position
            # Get the past data
            past_data = price_array[index - p:index, 1:]
        # print(past_data.shape)

        if past_data.shape != (p, 7):  # If the data is not correct dimension, return
            if not self.back_test: print('Model predicted - Long position! Because the data is not in correct shape!')
            return False

        past_data = past_data.reshape(1, p, past_data.shape[1])
        y_pred = self.model.predict(past_data)
        y_pred = self.inverse_transform(y_pred)

        # Send telegram alert
        # send_training_alert()
        # print(y_pred)

        # If at least one of the predicted values is greater than the current price by 1%, then we should go long
        for i in range(len(y_pred)):
            diff = (y_pred[i] - x_price) * 100 / x_price
            if diff >= self.take_profit:
                if not self.back_test: print('Model predicted - Long position! Difference:', diff)
                return False
            elif diff <= -self.take_profit:
                if not self.back_test: print('Model predicted - Short position! Difference:', diff)
                return True
        if not self.back_test: print('Model predicted - Long position! Because there is less than 1% price movement!')
        return False
