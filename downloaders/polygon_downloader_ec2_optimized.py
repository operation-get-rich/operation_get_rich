"""
Downloader that is optimized for S3: It will store all stocks data in 1 file
"""
import argparse
import os
from typing import AnyStr, List

import alpaca_trade_api as tradeapi
import numpy as np
import pandas
import pandas as pd
from dateutil.parser import parse

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from decorators import retry_with_timeout, RetryTimeoutError
from directories import DATA_DIR
from utils import get_all_ticker_names, create_dir, get_current_datetime, get_alpaca_time_str_format, \
    construct_polygon_date_tuples

SAVE_PATH_DIR = f'{DATA_DIR}/polygon_stock_prices'
START_DATE = '2019-01-01T03:00:00-05:00'
current_datetime_str = get_alpaca_time_str_format(get_current_datetime())
END_DATE = current_datetime_str
COMPANY_STEPS = 200

create_dir(DATA_DIR)
create_dir(SAVE_PATH_DIR)

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

# Arguments
parser = argparse.ArgumentParser(description='TraderGRU Train')

# General Settings
parser.add_argument('--start_index', type=int, default=0, help='The download batch start index')
args = parser.parse_args()


def main():
    tickers = get_all_ticker_names()

    _download_all_tickers_in_batches(tickers)

    _combine_all_stock_batches()


def _download_all_tickers_in_batches(tickers):
    # type: (List[AnyStr]) -> None
    date_tuples = construct_polygon_date_tuples(parse(START_DATE), parse(END_DATE))
    start = args.start_index
    failed_tickers = []
    while start < len(tickers):
        end = min(len(tickers), start + COMPANY_STEPS)

        print("Current Iteration: ", start, flush=True)
        print("Downloading Tickers: ", tickers[start:end], flush=True)

        to_download_tickers = tickers[start:end]
        data, inner_failed_tickers = _download_some_tickers(to_download_tickers, date_tuples)

        failed_tickers += inner_failed_tickers

        if not data:
            print('No data', start, tickers[start:end])
            start += COMPANY_STEPS
            continue

        data_np = np.array(data)
        df = pd.DataFrame(data_np, columns=['ticker', 'time', 'open', 'close', 'low', 'high', 'volume'])

        df.to_csv(path_or_buf=f'{SAVE_PATH_DIR}/stocks_{start}_{end}.csv')

        print("Iteration done downloading: ", start, flush=True)
        start += COMPANY_STEPS

    _log_failed_tickers(failed_tickers)


def _download_some_tickers(to_download_tickers, date_tuples):
    data = []
    failed_tickers = []
    for ticker in to_download_tickers:
        for start_date, end_date in date_tuples:
            print(f'Downloading {ticker}')
            print(f'start_date = {start_date}')
            print(f'end_date = {end_date}\n')

            try:
                ticker_aggregate = _download_ticker(ticker, start_date, end_date)
            except RetryTimeoutError as exc:
                print(exc)
                failed_tickers.append((ticker, start_date, end_date))
                break

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
    return data, failed_tickers


@retry_with_timeout(timeout=30)
def _download_ticker(ticker, start_date, end_date):
    return api.polygon.historic_agg_v2(
        symbol=ticker,
        multiplier=1,
        timespan='minute',
        _from=start_date,
        to=end_date
    )


def _combine_all_stock_batches():
    df = pandas.DataFrame()
    for stock_batch in os.listdir(SAVE_PATH_DIR):
        df = df.append(pandas.read_csv(os.path.join(SAVE_PATH_DIR, stock_batch)))
    df.to_csv(f'{SAVE_PATH_DIR}/stocks_all.csv')


def _log_failed_tickers(failed_tickers):
    print('\n***Failure Report***')
    print(f'\n{len(failed_tickers)} tickers failed')
    for failed_ticker, failed_start_time, failed_end_time in failed_tickers:
        print(f'\n{failed_ticker}')
        print(f'{failed_start_time}')
        print(f'{failed_end_time}')


if __name__ == "__main__":
    main()
