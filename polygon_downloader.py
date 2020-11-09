from datetime import datetime

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils import get_all_ticker_names, create_dir

SAVE_PATH_DIR = './stock_prices_polygon'
START_DATE = '2020-10-09T03:00:00-05:00'
END_DATE = '2020-10-12T15:00:00-05:00'
COMPANY_STEPS = 100

create_dir(SAVE_PATH_DIR)

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

tickers = get_all_ticker_names()
start = 0
while start < len(tickers):
    end = min(len(tickers), start + COMPANY_STEPS)
    print("Current Iteration: ", start, flush=True)
    print("Downloading Tickers: ", tickers[start:end], flush=True)

    to_download_tickers = tickers[start:end]

    data = []
    for ticker in to_download_tickers:
        ticker_aggregate = api.polygon.historic_agg_v2(
            symbol=ticker,
            multiplier=1,
            timespan='minute',
            _from=START_DATE,
            to=END_DATE
        )
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

    data_np = np.array(data)
    df = pd.DataFrame(data_np, columns=['ticker', 'time', 'open', 'close', 'low', 'high', 'volume'])

    df.to_csv(path_or_buf=f'{SAVE_PATH_DIR}/stock_{start}.csv', mode='w')
    print("Iteration done: ", start, flush=True)
    start += COMPANY_STEPS
