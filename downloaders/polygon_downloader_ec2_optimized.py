"""
Downloader that is optimized for S3: It will store all stocks data in 1 file
"""

from datetime import timedelta
import signal
from typing import AnyStr, List

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from dateutil.parser import parse
from pandas import DataFrame

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from directories import DATA_DIR
from utils import get_all_ticker_names, create_dir, get_current_datetime, get_alpaca_time_str_format

SAVE_PATH_DIR = f'{DATA_DIR}/polygon_stock_prices'
START_DATE = '2019-01-01T03:00:00-05:00'
current_date_time = get_alpaca_time_str_format(get_current_datetime())
END_DATE = current_date_time
COMPANY_STEPS = 200

create_dir(SAVE_PATH_DIR)

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)


def main():
    tickers = get_all_ticker_names()

    tickers = ['OPTT', 'EDSA']

    entire_stocks_df = _build_entire_stocks_df(tickers)

    entire_stocks_df.to_csv(path_or_buf=f'{SAVE_PATH_DIR}/stocks.csv', mode='w')


def _build_entire_stocks_df(tickers):
    # type: (List[AnyStr]) -> DataFrame
    date_tuples = _construct_date_tuples(START_DATE, END_DATE)
    start = 0
    total_df = pd.DataFrame()
    while start < len(tickers):
        end = min(len(tickers), start + COMPANY_STEPS)

        print("Current Iteration: ", start, flush=True)
        print("Downloading Tickers: ", tickers[start:end], flush=True)

        to_download_tickers = tickers[start:end]

        data = _download_tickers(to_download_tickers, date_tuples)

        if not data:
            print('No data', start, tickers[start:end])
            start += COMPANY_STEPS
            continue

        data_np = np.array(data)
        df = pd.DataFrame(data_np, columns=['ticker', 'time', 'open', 'close', 'low', 'high', 'volume'])
        total_df = total_df.append(df)

        print("Iteration done downloading: ", start, flush=True)
        start += COMPANY_STEPS
    return total_df


def _construct_date_tuples(start_date, end_date):
    date_tuples = []

    start_datetime = parse(start_date)
    end_datetime = parse(end_date)

    current_datetime = start_datetime
    while current_datetime < end_datetime:
        date_tuples.append(
            (
                str(current_datetime),
                str(current_datetime + timedelta(days=17))
            )
        )
        current_datetime = current_datetime + timedelta(days=18)

    return date_tuples


def _download_tickers(to_download_tickers, date_tuples):
    data = []
    for ticker in to_download_tickers:
        for start_date, end_date in date_tuples:
            print(f'Downloading {ticker}')
            print(f'start_date = {start_date}')
            print(f'end_date = {end_date}\n')

            ticker_aggregate = None
            for _ in range(2):
                try:
                    ticker_aggregate = _download_ticker(ticker, start_date, end_date)
                    break
                except Exception as exc:
                    print(exc)
                    pass

            for prices in ticker_aggregate:
                data.append(
                    [
                        ticker,
                        str(prices.timestamp),
                        prices.open,
                        prices.close,
                        prices.low,
                        prices.high,
                        prices.volume
                    ]
                )
    return data


def _download_ticker(ticker, start_date, end_date):
    def handler(signum, frame):
        raise TimeoutError("Ticker Download is too long")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(30)
    try:
        ticker_aggregate = api.polygon.historic_agg_v2(
            symbol=ticker,
            multiplier=1,
            timespan='minute',
            _from=start_date,
            to=end_date
        )
        return ticker_aggregate
    except Exception:
        raise


main()
