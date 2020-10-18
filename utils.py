import os
from datetime import datetime


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


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


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