import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils import get_all_ticker_names, create_dir

SAVE_PATH_DIR = './stock_prices'
START_DATE = '2020-10-05T03:00:00-05:00'
END_DATE = '2020-10-09T15:00:00-05:00'
COMPANY_STEPS = 100

create_dir(SAVE_PATH_DIR)

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

print("Starting Download:", flush=True)

tickers = get_all_ticker_names()
start = 0
while start < len(tickers):
    end = min(len(tickers), start + COMPANY_STEPS)
    print("Current Iteration: ", start, flush=True)
    print("Downloading Tickers: ", tickers[start:end], flush=True)

    barset = api.get_barset(
        symbols=','.join(tickers[start:end]),
        timeframe='1Min',
        start=START_DATE,
        end=END_DATE
    )

    data = []
    for ticker in barset:
        for bar in barset[ticker]:
            data.append([ticker, bar.t, bar.o, bar.c, bar.l, bar.h, bar.v])

    data_np = np.array(data)
    df = pd.DataFrame(data_np, columns=['ticker', 'time', 'open', 'close', 'low', 'high', 'volume'])

    df.to_csv(path_or_buf=f'{SAVE_PATH_DIR}/stock_{start}.csv', mode='w')
    print("Iteration done: ", start, flush=True)
    start += COMPANY_STEPS
