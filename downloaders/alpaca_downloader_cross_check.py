import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils import get_all_ticker_names, create_dir

SAVE_PATH_DIR = '../datas/alpaca_cross_check'
COMPANY_STEPS = 100

create_dir(SAVE_PATH_DIR)

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

print("Starting Download:", flush=True)

the_date = '2020-12-17'
start_datetime = f'{the_date}T03:00:00-05:00'
end_datetime = f'{the_date}T15:00:00-05:00'

tickers = ['TKAT']
for ticker in tickers:
    print("Downloading Tickers: ", ticker, flush=True)

    barset = api.get_barset(
        symbols=ticker,
        timeframe='1Min',
        start=start_datetime,
        end=end_datetime
    )

    data = []
    for ticker in barset:
        for bar in barset[ticker]:
            data.append([ticker, bar.t, bar.o, bar.c, bar.l, bar.h, bar.v])

    data_np = np.array(data)
    df = pd.DataFrame(data_np, columns=['ticker', 'time', 'open', 'close', 'low', 'high', 'volume'])

    df.to_csv(path_or_buf=f'{SAVE_PATH_DIR}/{ticker}_{the_date}.csv', mode='w', index=False)
