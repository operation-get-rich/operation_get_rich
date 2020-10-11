import os
from datetime import datetime

import pandas
import pytz
from pandas import DataFrame

from utils import get_all_ticker_names

filename = 'stock_price_copy_small.csv'
stock_price_df = pandas.read_csv(filename)  # type: DataFrame


def get_date(time):
    datetime_obj = datetime.strptime(''.join(time.rsplit(':', 1)), '%Y-%m-%d %H:%M:%S%z')
    return datetime(year=datetime_obj.year,
                    month=datetime_obj.month,
                    day=datetime_obj.day,
                    tzinfo=datetime_obj.tzinfo
                    )


def is_more_than_ten_am(time):
    datetime_obj = datetime.strptime(''.join(time.rsplit(':', 1)), '%Y-%m-%d %H:%M:%S%z')
    ten_am_obj = datetime(
        year=datetime_obj.year,
        month=datetime_obj.month,
        day=datetime_obj.day,
        hour=10,
        tzinfo=datetime_obj.tzinfo
    )
    return datetime_obj > ten_am_obj  # TODO


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


# tickers = get_all_ticker_names()
tickers = stock_price_df.ticker.unique()

selected_segments = []  # [('AAPL', '01/17/20'), ('AAPL', '01/18/20')]
for current_ticker in tickers:
    is_open_price_found = False
    current_ticker_price_df = stock_price_df[stock_price_df.ticker == current_ticker]  # find all rows of company
    is_first_date = True
    dataframe_length = len(current_ticker_price_df)
    for index, row in current_ticker_price_df.iterrows():
        if index + 1 >= dataframe_length:
            break
        next_date = get_date(current_ticker_price_df.loc[index + 1].time)
        if get_date(row.time) < next_date:
            close_price = row.close
            is_open_price_found = False
            is_first_date = False
        elif is_more_than_ten_am(row.time) and not is_open_price_found and not is_first_date:
            is_open_price_found = True
            open_price = row.open
            if (open_price - close_price) / close_price > 0.15:
                selected_segments.append((current_ticker, get_date(row.date)))
            continue

for ticker_name, the_date in selected_segments:
    the_segment = stock_price_df[
        stock_price_df.ticker == ticker_name and stock_price_df.just_date == the_date]  # type: DataFrame
    the_dir = create_dir('./processed_data/{}'.format(ticker_name))
    the_segment.to_csv(path_or_buf='{}/{}_{}'.format(the_dir, ticker_name, the_date))
