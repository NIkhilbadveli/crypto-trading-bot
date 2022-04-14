"""
This script will create and enable a telegram bot to send alerts of trades.
"""
from trade_class import Trade
import datetime
import telegram

access_token = '5052518752:AAF_jTUmCAiS-89bjop8tfBl4fyFLOCvEmc'
user_id = 5206375205  # Hardcoded for now
bot = telegram.Bot(token=access_token)


def send_close_alert(trade: Trade):
    """
    This function sends an email to the receiver_email
    :param trade:
    """
    today = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    msg = '<b>Trade closed</b> - ' + today + '\n\n'
    msg += "Here are the details of the trade:- \n"
    msg += "Trade Buy time: " + str(trade.buy_time) + "\n"
    msg += "Trade Sell time: " + str(trade.sell_time) + "\n"
    msg += "Trade Buy price: " + str(trade.buy_price) + "\n"
    msg += "Trade Sell price: " + str(trade.sell_price) + "\n"
    msg += "Trade Profit/Loss: " + str(trade.pl_abs) + "\n"

    sent_body = (msg + "\n\n" +
                 "\t\tYour loving bot \u2764 \u2764 \u2764 \u2764\n")

    try:
        bot.send_message(user_id, sent_body)
        print('Alert sent - ' + today)
    except Exception as e:
        print("Some error happened while sending!")
        print(e)


def send_open_alert(trade: Trade):
    """
    This function sends an email to the receiver_email
    :param trade:
    """

    today = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    msg = '<b>Trade opened</b> - ' + today + '\n\n'
    msg += "Here are the details of the trade:- \n"
    msg += "Trade Buy time: " + str(trade.buy_time) + "\n"
    msg += "Trade Buy price: " + str(trade.buy_price) + "\n"
    msg += "Current balance before trade: " + str(trade.opening_balance) + "\n"

    sent_body = (msg + "\n\n" +
                 "\t\tYour loving bot \u2764 \u2764 \u2764 \u2764\n")
    try:
        bot.send_message(user_id, sent_body)
        print('Alert sent - ' + today)
    except Exception as e:
        print("Some error happened while sending!")
        raise e


def send_socket_disconnect():
    """
    This function sends an email to the receiver_email
    """

    today = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    msg = 'Socket disconnected - ' + today + '\n'
    msg += "Check the logs: \n"

    sent_body = (msg + "\n\n" +
                 "\t\tYour loving bot \u2764 \u2764 \u2764 \u2764\n")

    try:
        bot.send_message(user_id, sent_body)
        print('Alert sent - ' + today)
    except Exception as e:
        print("Some error happened while sending!")
        raise e
