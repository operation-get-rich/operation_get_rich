import os
from datetime import datetime
from typing import AnyStr

import pandas
import pytz
from pandas import DataFrame

from utils import get_all_ticker_names

GAP_UP_THRESHOLD = 0.10
VOLUME_THRESHOLD = 1e+05
RAW_STOCK_PRICE_DIR = 'raw_data'
GAPED_UP_STOCKS_DIR_NAME = 'gaped_up_stocks_early_volume_1e5_gap_10'

def get_date(time_str):
    # type: (AnyStr) -> datetime
    datetime_obj = datetime.strptime(''.join(time_str.rsplit(':', 1)), '%Y-%m-%d %H:%M:%S%z')
    return datetime(year=datetime_obj.year,
                    month=datetime_obj.month,
                    day=datetime_obj.day,
                    tzinfo=datetime_obj.tzinfo
                    )


def get_date_time(time_str):
    # type: (AnyStr) -> datetime
    return datetime.strptime(''.join(time_str.rsplit(':', 1)), '%Y-%m-%d %H:%M:%S%z')


def get_date_string(time):
    # type: (datetime) -> AnyStr
    return ' '.join(time.isoformat().split('T'))


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def is_gapped_up(open_price, close_price):
    return (open_price - close_price) / close_price > GAP_UP_THRESHOLD

raw_stock_list = os.listdir(RAW_STOCK_PRICE_DIR)
for stock_file in raw_stock_list:
    stock_path = os.path.join(RAW_STOCK_PRICE_DIR, stock_file)

    print("Loading Data:", flush=True)
    stock_price_df = pandas.read_csv(stock_path)  # type: DataFrame

    current_ticker = None
    previous_ticker = None
    open_price = None
    close_price = None
    cummulative_volume = 0
    
    segments = []  # [('AAPL', '01/17/20'), ('AAPL', '01/18/20')]
    print("Starting Segmentation:", flush=True)
    for index in reversed(range(len(stock_price_df.index))):
        row = stock_price_df.loc[index]

        # Reset everything when new ticker arrives
        if current_ticker is None or row.ticker != previous_ticker:
            current_ticker = row.ticker
            open_price = None
            close_price = None
            cummulative_volume = 0
            print("Current Ticker: ", current_ticker, flush=True)
            print("Current Index: ", index, flush=True)

        if get_date_time(row.time).hour < 10 and open_price is None:
            open_price = row.open

        # if close_price is not None:
        #     cummulative_volume += row.volume
        if open_price is not None:
            cummulative_volume += row.volume

        if index - 1 > 0:
            if open_price is not None and get_date(row.time) > get_date(stock_price_df.loc[index - 1].time):
                close_price = stock_price_df.loc[index - 1].open
                if is_gapped_up(open_price, close_price) and cummulative_volume >= VOLUME_THRESHOLD:
                    segments.append((current_ticker, get_date(row.time)))
                    print("Gapped Up: ", current_ticker, get_date(row.time), flush=True)

                open_price = None
                close_price = None
                cummulative_volume = 0

        previous_ticker = current_ticker

    stock_price_df['just_date'] = stock_price_df.apply(lambda row: get_date(row.time), axis=1)

    create_dir('./%s' % GAPED_UP_STOCKS_DIR_NAME)
    for ticker_segment, date_segment in segments:
        print("Writing Ticker: ", ticker_segment, flush=True)
        get_date_string(date_segment)
        the_segment = stock_price_df[
            (stock_price_df.ticker == ticker_segment) &
            (stock_price_df.just_date == date_segment)
            ]  # type: DataFrame
        the_segment = the_segment.drop('just_date', axis=1)
        the_segment = the_segment.drop(the_segment.columns[0], axis=1)

        the_dir = create_dir('./{}/{}'.format(GAPED_UP_STOCKS_DIR_NAME, ticker_segment))

        the_segment.to_csv(path_or_buf='{}/{}_{}'.format(the_dir, ticker_segment, date_segment), index=False)
