import datetime
import smtplib
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from trade_class import Trade

# =============================================================================
# SET EMAIL LOGIN REQUIREMENTS
# =============================================================================
gmail_user = 'nikhilbadveli6@gmail.com'
gmail_app_password = 'nofbzulwstsibyof'
receiver_email = 'nikhilreddy6174@gmail.com'


# =============================================================================
# SET THE INFO ABOUT THE SAID EMAIL
# =============================================================================
def send_close_alert(trade: Trade):
    """
    This function sends an email to the receiver_email
    :param trade:
    """
    return  # Temporarily disabled
    message = MIMEMultipart()
    today = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    message['Subject'] = "Trade closed - " + today
    message['From'] = gmail_user
    message['To'] = receiver_email

    msg = "Here are the details of the trade: \n"
    msg += "Trade Buy time: " + str(trade.buy_time) + "\n"
    msg += "Trade Sell time: " + str(trade.sell_time) + "\n"
    msg += "Trade Buy price: " + str(trade.buy_price) + "\n"
    msg += "Trade Sell price: " + str(trade.sell_price) + "\n"
    msg += "Trade Profit/Loss: " + str(trade.pl_abs) + "\n"

    sent_body = (msg + "\n\n" +
                 "\t\tYour loving bot \u2764 \u2764 \u2764 \u2764\n")
    message.attach(MIMEText(sent_body, "html"))
    msg_body = message.as_string()

    server = SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(message['From'], gmail_app_password)
    try:
        server.sendmail(message['From'], message['To'], msg_body)
        print('Email sent - ' + today)
    except Exception as e:
        print("Some error happened while sending!")
        print(e)
    server.quit()


def send_open_alert(trade: Trade):
    """
    This function sends an email to the receiver_email
    :param trade:
    """
    return  # Temporarily disabled
    message = MIMEMultipart()
    today = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    message['Subject'] = "New Trade opened - " + today
    message['From'] = gmail_user
    message['To'] = receiver_email

    msg = "Here are the details of the trade: \n"
    msg += "Trade Buy time: " + str(trade.buy_time) + "\n"
    msg += "Trade Buy price: " + str(trade.buy_price) + "\n"
    msg += "Current balance before trade: " + str(trade.opening_balance) + "\n"

    sent_body = (msg + "\n\n" +
                 "\t\tYour loving bot \u2764 \u2764 \u2764 \u2764\n")
    message.attach(MIMEText(sent_body, "html"))
    msg_body = message.as_string()

    server = SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(message['From'], gmail_app_password)
    try:
        server.sendmail(message['From'], message['To'], msg_body)
        print('Email sent - ' + today)
    except Exception as e:
        print("Some error happened while sending!")
        raise e
    server.quit()
