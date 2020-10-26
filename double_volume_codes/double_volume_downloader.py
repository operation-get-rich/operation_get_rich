import csv
import os

import alpaca_trade_api as tradeapi

from config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL

api = tradeapi.REST(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

the_date = '2020-10-22'
barset = api.get_barset(
    symbols='SQBG',
    timeframe='1Min',
    start=('%sT08:30:00-05:00' % the_date),
    end=('%sT15:00:00-05:00' % the_date)
)

# create empty dir
root_dir = '../double_volume_stocks'
if not os.path.exists(root_dir):
    os.mkdir(path=('%s' % root_dir))

for ticker in barset:
    data = []
    ticker_file_name = os.path.join(root_dir, f'{the_date}_{ticker}.csv')
    with open(ticker_file_name, mode="a") as ticker_file:
        writer = csv.writer(ticker_file)
        writer.writerow(['date', 'open', 'close', 'low', 'high', 'volume'])
        for bar in barset[ticker]:
            writer.writerow(
                [bar.t, bar.o, bar.c, bar.l, bar.h, bar.v]
            )
