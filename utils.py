import json
import os
import shutil
from datetime import datetime, timedelta
import pytz

import alpaca_trade_api as tradeapi

from config import PAPER_ALPACA_API_KEY, PAPER_ALPACA_SECRET_KEY, PAPER_ALPACA_BASE_URL

api = tradeapi.REST(
    key_id=PAPER_ALPACA_API_KEY,
    secret_key=PAPER_ALPACA_SECRET_KEY,
    base_url=PAPER_ALPACA_BASE_URL
)

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S%z'
DATE_FORMAT = '%Y-%m-%d'
US_CENTRAL_TIMEZONE = 'US/Central'

"""
Ticker related utils
"""


def get_all_ticker_names():
    ticker_names = []
    ticker_names += _get_ticker_names('stock_tickers.txt')
    return ticker_names


def _get_ticker_names(file_name):
    ticker_names = []
    exchange_file = open(file_name)
    lines = exchange_file.readlines()
    for line in lines:
        ticker_names.append(line.strip())
    return ticker_names


""""""

"""
Directory related utils
"""


def create_file(path):
    if not os.path.exists(path):
        with open(path, 'w'):
            os.utime(path, None)
        return True
    return False


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_current_directory(dunder_file):
    return os.path.dirname(os.path.realpath(dunder_file))


""""""

"""
Time related utils
"""


def get_current_datetime():
    return datetime.now(pytz.timezone(US_CENTRAL_TIMEZONE))


def get_today_market_open():
    return get_current_datetime().replace(hour=8, minute=30)


def get_alpaca_time_str_format(
        the_datetime  # type: datetime
):
    alpaca_time_str = 'T'.join(the_datetime.strftime(DATETIME_FORMAT).split())
    alpaca_time_str = alpaca_time_str[0:-2] + ':' + alpaca_time_str[-2:]
    return alpaca_time_str


def get_previous_market_open(anchor_time=None):
    if not anchor_time:
        anchor_time = get_current_datetime()

    previous_market_open = anchor_time - timedelta(days=1)
    previous_market_open_date_str = previous_market_open.date().strftime(DATE_FORMAT)

    while not api.get_calendar(start=previous_market_open_date_str,
                               end=previous_market_open_date_str)[0].open:
        previous_market_open = previous_market_open - timedelta(days=1)
        previous_market_open_date_str = previous_market_open.date().strftime(DATE_FORMAT)

    return previous_market_open


""""""


def format_usd(capital):
    capital_formatted = '${:,.2f}'.format(capital)
    return capital_formatted


def update_state(state_file_location, update_func=None, *args, **kwargs):
    created = create_file(state_file_location)
    if created:
        state = {}
    else:
        with open(state_file_location, 'r') as state_file:
            state = json.load(state_file)

    if update_func:
        update_func(state, *args, **kwargs)

    if '_meta' in state:
        if 'first_state_update_call' not in state['_meta']:
            state['_meta']['first_state_update_call'] = datetime.now().strftime(DATETIME_FORMAT)
        state['_meta']['latest_state_update_call'] = datetime.now().strftime(DATETIME_FORMAT)

    with open(state_file_location, 'w') as state_file:
        json.dump(state, state_file)

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)