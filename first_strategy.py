# This file is for trying out cryptocurrency trading strategies
import requests
import pandas as pd
from datetime import datetime, timedelta

# api_key = '27c2d758c80346a5a4bd3d76fd6e23d157d4ae2ce27c27e35ab83539a937d67b'
api_key = 'e832991cea1d4dc1a710bd9abaf2b556'
upper_bound = round(1 + 1.5 / 100, 3)
lower_bound = round(1 - 5 / 100, 3)


# pd.set_option('display.max_columns', None)

def get_crypto_price(symbol, exchange, start_date, end_date):
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}/{exchange}&interval=5min&apikey={api_key}' \
              f'&start_date={start_date}&end_date={end_date}&order=ASC'
    print(api_url)
    raw = requests.get(api_url).json()
    if raw['status'] == 'error':
        print(raw['message'])

    df = pd.DataFrame(raw['values'])[['datetime', 'close', 'high', 'low']].set_index('datetime')
    df['close'] = df['close'].astype('float64')
    df['high'] = df['high'].astype('float64')
    df['low'] = df['low'].astype('float64')
    # df['max_change'] = (df['high'] - df['low']) * 100 / df['low']
    # df.index = pd.to_datetime(df.index, unit='s')
    return df


def find_buy_sell_times(ada):
    column_names = ['buy_time', 'buy_price', 'sell_time', 'sell_price', 'profit', 'buy_pos']
    out_df = pd.DataFrame(columns=column_names)
    out_row = []

    buy_position = True
    buy_time = ada.index[0]
    out_row.append(buy_time)
    buy_price = ada.iloc[0]['close']
    out_row.append(buy_price)

    i = 0
    for index, row in ada.iterrows():
        if (row['close'] >= buy_price * upper_bound or row['close'] <= buy_price * lower_bound) and buy_position:
            sell_time = ada.index[i]
            out_row.append(sell_time)
            sell_price = ada.iloc[i]['close']
            out_row.append(sell_price)
            out_row.append(
                round((sell_price - buy_price) / buy_price, 4))  # profit in $ for $1 invested for each transaction
            out_row.append(buy_position)
            # print(out_row)
            out_df = out_df.append(pd.Series(out_row, index=column_names), ignore_index=True)

            out_row = []
            if i + 1 < len(ada) - 1 and sell_price > buy_price:
                buy_time = ada.index[i + 1]
                out_row.append(buy_time)
                buy_price = ada.iloc[i + 1]['close']
                out_row.append(buy_price)
            elif i + 1 < len(ada) - 1 and sell_price < buy_price:
                sell_time = ada.index[i + 1]
                out_row.append(sell_time)
                sell_price = ada.iloc[i + 1]['close']
                out_row.append(sell_price)
                buy_position = False
        elif not buy_position and (row['close'] >= sell_price * upper_bound or row[
            'close'] <= sell_price * lower_bound):  # Should swap sell_price and buy_price in the below condition
            buy_time = ada.index[i]
            out_row.append(buy_time)
            buy_price = ada.iloc[i]['close']
            out_row.append(buy_price)
            out_row.append(
                round((-buy_price + sell_price) / sell_price, 4))  # profit in $ for $1 invested for each transaction
            out_row.append(buy_position)
            # print(out_row)
            out_df = out_df.append(pd.Series(out_row, index=column_names), ignore_index=True)

            out_row = []
            if i + 1 < len(ada) - 1 and sell_price > buy_price:
                buy_time = ada.index[i + 1]
                out_row.append(buy_time)
                buy_price = ada.iloc[i + 1]['close']
                out_row.append(buy_price)
                buy_position = True
            elif i + 1 < len(ada) - 1 and sell_price < buy_price:
                sell_time = ada.index[i + 1]
                out_row.append(sell_time)
                sell_price = ada.iloc[i + 1]['close']
                out_row.append(sell_price)
        i += 1

    return out_df


def calc_output(cur1, cur2):
    if upper_bound > 1 and 1 > lower_bound > 0:
        output = find_buy_sell_times(get_crypto_price(cur1, cur2, 1440, 1640197800))
        # print(output)
        profit_perc = round(output['profit'].sum() * 100 / len(output), 2)
        print('Percentage profit per day :- ' + str(profit_perc))
        print('Projected annual returns :- ' + str(profit_perc * 365))
        print('Win rate :- ' + str(len(output.loc[output.profit > 0]) * 100 / len(output)))
    else:
        print('Set proper upper and lower bounds!')


def get_epochs(from_date, to_date):
    delta = to_date - from_date  # as timedelta
    epochs = [int((from_date + timedelta(days=i)).timestamp()) for i in range(delta.days + 1)]
    return epochs


def split_into_days(df):
    df = df[:-1]
    list_of_dfs = []
    start_time = str(df.index[0])
    start = 0
    i = 0
    for index in df.index:
        if start_time.split(' ')[0] != str(index).split(' ')[0]:
            list_of_dfs.append(df.iloc[start: i])
            start_time = str(index)
            start = i
        i += 1
    return list_of_dfs


def buy_sell_logic(cur, amount, start_date, end_date):
    df_main = get_crypto_price(cur, 'USD', start_date, end_date)
    days_df = split_into_days(df_main)
    # epochs = get_epochs(datetime(2021, 12, 18, 5, 30), datetime(2021, 12, 23, 5, 30))
    total_profit = 0
    column_names = ['buy_time', 'buy_price', 'sell_time', 'sell_price', 'profit', 'total_profit']
    out_df = pd.DataFrame(columns=column_names)

    for df in days_df:
        buy_index = int(len(df.index)/2)
        buy_price = df['close'][buy_index]
        buy_time = df.index[buy_index]
        sell_time = 0
        sell_price = 0
        profit = 0

        for index, row in df.iterrows():
            sell_time = index
            sell_price = row['close']
            if row['close'] >= buy_price * upper_bound:
                profit = amount * (sell_price - buy_price) / buy_price
                break
            elif row['close'] <= buy_price * lower_bound:
                profit = amount * (sell_price - buy_price) / buy_price
                break
            elif index == df.index[-1]:
                profit = amount * (sell_price - buy_price) / buy_price

        total_profit += profit
        out_row = [buy_time, buy_price, sell_time, sell_price, profit, total_profit]
        out_df = out_df.append(pd.Series(out_row, index=column_names), ignore_index=True)

    # print(out_df)
    print('Total profit after 30 days for ' + cur + ' is :- ' + str(total_profit))
    return total_profit

# print(get_crypto_price('ETH', 'USD', '2021-11-23', '2021-12-23'))
