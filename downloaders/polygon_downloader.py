from datetime import datetime, timedelta
import signal

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from dateutil.parser import parse

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils import get_all_ticker_names, create_dir

SAVE_PATH_DIR = '../datas/polygon_stock_prices'
START_DATE = '2019-01-01T03:00:00-05:00'
END_DATE = '2020-11-11T15:00:00-05:00'
COMPANY_STEPS = 10

create_dir(SAVE_PATH_DIR)

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

tickers = get_all_ticker_names()


def main():
    date_tuples = _construct_date_tuples(START_DATE, END_DATE)
    start = 3110
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

        df.to_csv(path_or_buf=f'{SAVE_PATH_DIR}/stock_{start}.csv', mode='w')
        print("Iteration done: ", start, flush=True)
        start += COMPANY_STEPS


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
