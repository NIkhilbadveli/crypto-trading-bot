"""
Generate a trading report that will have the following information for each currency pair:
    - Starting balance
    - Total profit in USD
    - Total number of trades
    - Win/loss ratio
    - Max. Loss in USD
    - Projected yearly ROI
"""
import pandas as pd
from datetime import datetime


def generate_report():
    """
    This function reads trades.csv and generates the report.
    :return:
    """
    df = pd.read_csv('trades.csv')
    # Group by currency, interval, short, compound and then sum the total profit for trade_status == 3
    df_closed = df.loc[df['trade_status'] == 3, :]
    df_grouped = df_closed.groupby(['currency', 'interval', 'short', 'compound'])
    df_final = df_grouped['pl_abs'].sum().to_frame('total_profit')
    df_final['starting_balance'] = 500
    df_final = pd.concat([df_final, df_grouped.size().to_frame('total_trades')], axis=1, join="inner")
    df_final = pd.concat([df_final, df_grouped['buy_time'].min().to_frame('first_buy_time')], axis=1, join="inner")
    df_final = pd.concat([df_final, df_grouped['sell_time'].max().to_frame('last_sell_time')], axis=1, join="inner")
    df_final['first_buy_time'] = pd.to_datetime(df_final['first_buy_time'], unit='ms')
    df_final['last_sell_time'] = pd.to_datetime(df_final['last_sell_time'], unit='ms')
    df_final['trading_period'] = (df_final['last_sell_time'] - df_final['first_buy_time']).dt.total_seconds() / 86400
    df_final['trading_period'] = df_final['trading_period'].round(2)
    df_final['projected_roi'] = df_final['total_profit'] / df_final['starting_balance'] * (
            365 / df_final['trading_period']) * 100

    # Pretty print the dataframe
    print(df_final.to_string(index=True))
    # Print the sum of total profit
    print("\nTotal profit in USD: {}".format(df_final['total_profit'].sum()))
    # Print the projected yearly ROI
    print("Projected yearly ROI (%): {}".format(df_final['projected_roi'].mean()))


generate_report()
